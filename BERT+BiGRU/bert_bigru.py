import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import csv
from collections import defaultdict
import re
import os

# 从环境变量或config.py读取配置
def get_config():
    config = {}
    
    # 尝试从环境变量读取
    if 'DATASET_METHOD' in os.environ:
        config['DATASET_METHOD'] = int(os.environ['DATASET_METHOD'])
    if 'DATASET_PATH' in os.environ:
        config['DATASET_PATH'] = os.environ['DATASET_PATH']
    if 'TRAIN_DATASET_PATH' in os.environ:
        config['TRAIN_DATASET_PATH'] = os.environ['TRAIN_DATASET_PATH']
    if 'TEST_DATASET_PATH' in os.environ:
        config['TEST_DATASET_PATH'] = os.environ['TEST_DATASET_PATH']
    if 'TRAIN_DATASET_NAME' in os.environ:
        config['TRAIN_DATASET_NAME'] = os.environ['TRAIN_DATASET_NAME']
    if 'TEST_DATASET_NAME' in os.environ:
        config['TEST_DATASET_NAME'] = os.environ['TEST_DATASET_NAME']
    
    # 如果环境变量中没有，从config.py读取
    try:
        from config import (
            DATASET_PATH, DATASET_METHOD, TRAIN_DATASET_PATH, TEST_DATASET_PATH,
            TRAIN_RATIO, TEST_RATIO, TRAIN_DATASETS, TEST_DATASETS
        )
        if 'DATASET_METHOD' not in config:
            config['DATASET_METHOD'] = DATASET_METHOD
        if 'DATASET_PATH' not in config:
            config['DATASET_PATH'] = DATASET_PATH
        if 'TRAIN_DATASET_PATH' not in config:
            config['TRAIN_DATASET_PATH'] = TRAIN_DATASET_PATH
        if 'TEST_DATASET_PATH' not in config:
            config['TEST_DATASET_PATH'] = TEST_DATASET_PATH
        config['TRAIN_RATIO'] = TRAIN_RATIO
        config['TEST_RATIO'] = TEST_RATIO
        config['TRAIN_DATASETS'] = TRAIN_DATASETS
        config['TEST_DATASETS'] = TEST_DATASETS
    except ImportError:
        pass
    
    return config

config = get_config()
DATASET_METHOD = config.get('DATASET_METHOD', 1)
DATASET_PATH = config.get('DATASET_PATH')
TRAIN_DATASET_PATH = config.get('TRAIN_DATASET_PATH')
TEST_DATASET_PATH = config.get('TEST_DATASET_PATH')
TRAIN_DATASET_NAME = config.get('TRAIN_DATASET_NAME')
TEST_DATASET_NAME = config.get('TEST_DATASET_NAME')
TRAIN_RATIO = config.get('TRAIN_RATIO', 0.9)
TEST_RATIO = config.get('TEST_RATIO', 0.4)
import os



class CitationDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, label2id, max_len):
        self.dataset = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.dataset.append(item)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label2id[label])
        }


class BertBiGRU(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels, dropout_rate=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(
            self.bert.config.hidden_size,
            hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_out = outputs.last_hidden_state
        sequence_out = self.dropout(sequence_out)

        gru_out, _ = self.gru(sequence_out)
        cls_out = gru_out[:, -1, :]

        logits = self.classifier(cls_out)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return torch.argmax(logits, dim=1)


# ======================
# 评估指标
# ======================
def compute_metrics(true_labels, pred_labels, label_list):
    report = classification_report(true_labels, pred_labels, target_names=label_list, output_dict=True)
    results = {}

    results['整体准确率'] = report['accuracy']
    results['加权平均'] = {
        '精确率': report['weighted avg']['precision'],
        '召回率': report['weighted avg']['recall'],
        'F1分数': report['weighted avg']['f1-score'],
        '支持数': report['weighted avg']['support']
    }

    for label in label_list:
        if label in report:
            results[label] = {
                '精确率': report[label]['precision'],
                '召回率': report[label]['recall'],
                'F1分数': report[label]['f1-score'],
                '支持数': report[label]['support']
            }

    return results


# ======================
# 主函数
# ======================
def main():
    import sys
    import os
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建config.py所在路径（一键运行目录）
    config_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "run", "run_sen_func")
    sys.path.append(config_dir)

    
    # 预训练模型路径：优先读环境变量 BERT_MODEL_PATH（run_all_models.py 会注入，
    # 本机=E:\MODEL\bert-base-cased，服务器=/root/autodl-tmp/model/bert-base-cased）
    model_name = os.environ.get("BERT_MODEL_PATH", "/root/autodl-tmp/model/bert-base-cased")

    # ===== 超参数调整区域 =====
    max_len = 256
    batch_size = 32
    epochs = 5
    lr = 2e-5
    hidden_size = 256
    dropout_rate = 0.1
    clip_grad_norm = 0.5

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # ===== 标签 =====
    # 合并所有数据集的标签以确保标签一致性
    raw_labels = set()
    if DATASET_METHOD == 1:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                raw_labels.add(item['label'])
    else:
        with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                raw_labels.add(item['label'])
        with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                raw_labels.add(item['label'])

    label_list = sorted(raw_labels)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}

    # ===== 数据加载和划分 =====
    if DATASET_METHOD == 1:
        # 方法1：单个数据集按比例划分
        dataset = CitationDataset(DATASET_PATH, tokenizer, label2id, max_len)
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            random_state=42,
            stratify=[dataset[i]['labels'].item() for i in range(len(dataset))]
        )
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
    else:
        # 方法2：训练集和测试集分别使用不同数据集并按比例抽取
        # 加载训练集
        train_full_dataset = CitationDataset(TRAIN_DATASET_PATH, tokenizer, label2id, max_len)
        # 抽取训练集的一部分
        if TRAIN_RATIO < 1.0:
            train_indices, _ = train_test_split(
                range(len(train_full_dataset)),
                train_size=TRAIN_RATIO,
                random_state=42,
                stratify=[train_full_dataset[i]['labels'].item() for i in range(len(train_full_dataset))]
            )
            train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
        else:
            train_dataset = train_full_dataset
        
        # 加载测试集
        test_full_dataset = CitationDataset(TEST_DATASET_PATH, tokenizer, label2id, max_len)
        # 抽取测试集的一部分
        if TEST_RATIO < 1.0:
            val_indices, _ = train_test_split(
                range(len(test_full_dataset)),
                train_size=TEST_RATIO,
                random_state=42,
                stratify=[test_full_dataset[i]['labels'].item() for i in range(len(test_full_dataset))]
            )
            val_dataset = torch.utils.data.Subset(test_full_dataset, val_indices)
        else:
            val_dataset = test_full_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ===== 模型 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBiGRU(model_name, hidden_size, len(label_list), dropout_rate).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ======================
    # 训练
    # ======================
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = model(input_ids, attention_mask, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)

            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # ======================
    # 评估
    # ======================
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            preds = model(input_ids, attention_mask)

            true_labels.extend(labels)
            pred_labels.extend(preds.cpu().numpy())

    # ===== 输出具体识别细节CSV =====
    with open("predictions_vs_true.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["样本序号", "真实标签", "预测标签"])

        for sample_idx, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
            writer.writerow([sample_idx + 1, id2label[true_label], id2label[pred_label]])

    print("\n具体识别细节已保存到 predictions_vs_true.csv")

    # ===== 评估结果 =====
    results = compute_metrics(true_labels, pred_labels, label_list)

    print("\n===== 评估结果 =====")
    for label, metrics in results.items():
        if label == '整体准确率':
            print(f"{label}: {metrics:.4f}")
        else:
            print(f"{label}: P={metrics['精确率']:.4f}, R={metrics['召回率']:.4f}, F1={metrics['F1分数']:.4f}, 支持数={metrics['支持数']}")

    # ===== 保存结果 =====
    with open("classification_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["类别", "精确率", "召回率", "F1分数", "支持数"])

        for label, metrics in results.items():
            if label == '整体准确率':
                writer.writerow([label, metrics, "", "", ""])
            else:
                writer.writerow([label, metrics['精确率'], metrics['召回率'], metrics['F1分数'], metrics['支持数']])

    print("\n结果已保存到 classification_results.csv")


if __name__ == "__main__":
    main()