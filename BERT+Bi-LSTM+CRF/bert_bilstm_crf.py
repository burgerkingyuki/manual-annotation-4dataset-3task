import json
import re
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from torchcrf import CRF
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
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
# ======================
# 数据集
# ======================
class CitationDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id, max_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item['text']
                label = item['label']

                encoding, labels = self.process(text, label)
                self.samples.append((encoding, labels))

    def process(self, text, label):
        pattern = r'\[@([^\]]+)\]'
        matches = list(re.finditer(pattern, text))

        clean_text = ""
        spans = []
        last_end = 0

        for match in matches:
            start, end = match.span()
            span_text = match.group(1)

            clean_text += text[last_end:start]
            span_start = len(clean_text)
            clean_text += span_text
            span_end = len(clean_text)

            spans.append((span_start, span_end))
            last_end = end

        clean_text += text[last_end:]

        encoding = self.tokenizer(
            clean_text,
            return_offsets_mapping=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len
        )

        offsets = encoding['offset_mapping']
        labels = ['O'] * len(offsets)

        for span_start, span_end in spans:
            first = True
            for i, (s, e) in enumerate(offsets):
                if s == e:
                    continue
                if s >= span_end or e <= span_start:
                    continue
                if first:
                    labels[i] = f'B-{label}'
                    first = False
                else:
                    labels[i] = f'I-{label}'

        label_ids = [self.label2id.get(l, 0) for l in labels]
        return encoding, label_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoding, labels = self.samples[idx]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(labels)
        }

# ======================
# 模型
# ======================
class BertBiLSTMCRF(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=256, dropout_rate=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(outputs.last_hidden_state)

        lstm_out, _ = self.lstm(seq_out)
        emissions = self.classifier(lstm_out)

        mask = attention_mask.bool()
        mask[:, 0] = True

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

# ======================
# BIO → span
# ======================
def extract_spans(label_seq):
    spans = []
    start = None
    label_type = None

    for i, tag in enumerate(label_seq):
        if tag.startswith('B-'):
            if start is not None:
                spans.append((start, i - 1, label_type))
            start = i
            label_type = tag[2:]

        elif tag.startswith('I-'):
            continue

        else:
            if start is not None:
                spans.append((start, i - 1, label_type))
                start = None
                label_type = None

    if start is not None:
        spans.append((start, len(label_seq) - 1, label_type))

    return spans

# ======================
# span级评估
# ======================
def compute_span_metrics(true_labels, pred_labels):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_spans = set(extract_spans(true_seq))
        pred_spans = set(extract_spans(pred_seq))

        for span in pred_spans:
            if span in true_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1

        for span in true_spans:
            if span not in pred_spans:
                fn[span[2]] += 1

    results = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_support = 0

    for label in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
        p = tp[label] / (tp[label] + fp[label] + 1e-8)
        r = tp[label] / (tp[label] + fn[label] + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        support = tp[label] + fn[label]

        results[label] = (p, r, f1, support)
        total_tp += tp[label]
        total_fp += fp[label]
        total_fn += fn[label]
        total_support += support

    # 计算整体准确率
    overall_accuracy = total_tp / (total_tp + total_fp + 1e-8) if (total_tp + total_fp) > 0 else 0

    # 计算加权平均
    weighted_p = 0
    weighted_r = 0
    weighted_f1 = 0

    for label in results:
        p, r, f1, support = results[label]
        weighted_p += p * support
        weighted_r += r * support
        weighted_f1 += f1 * support

    weighted_p /= total_support if total_support > 0 else 1
    weighted_r /= total_support if total_support > 0 else 1
    weighted_f1 /= total_support if total_support > 0 else 1

    # 添加整体指标
    results['整体准确率'] = (overall_accuracy, 0, 0, total_tp + total_fp)
    results['加权平均'] = (weighted_p, weighted_r, weighted_f1, total_support)

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
    config_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "run", "run_object")
    sys.path.append(config_dir)
    
    
    model_name = "/root/autodl-tmp/model/bert-base-cased"

    # ===== 超参数调整区域 =====
    max_len = 256
    batch_size = 32
    epochs = 3
    lr = 2e-5
    dropout_rate = 0.1
    hidden_size = 256
    clip_grad_norm = 0.5

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # ===== 标签 =====
    # 合并所有数据集的标签以确保标签一致性
    raw_labels = set()
    if DATASET_METHOD == 1:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                raw_labels.add(json.loads(line)['label'])
    else:
        with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                raw_labels.add(json.loads(line)['label'])
        with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                raw_labels.add(json.loads(line)['label'])

    label_list = ['O']
    for l in raw_labels:
        label_list.append(f'B-{l}')
        label_list.append(f'I-{l}')

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # ===== 数据加载和划分 =====
    # 创建自定义数据集
    class SubsetDataset(Dataset):
        def __init__(self, original_dataset, indices):
            self.original_dataset = original_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.original_dataset[self.indices[idx]]

    if DATASET_METHOD == 1:
        # 方法1：单个数据集按比例划分
        dataset = CitationDataset(DATASET_PATH, tokenizer, label2id, max_len)
        # 首先获取每个样本的标签
        sample_labels = []
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sample_labels.append(item['label'])
        # 使用分层抽样
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            random_state=42,
            stratify=sample_labels
        )
        train_dataset = SubsetDataset(dataset, train_indices)
        val_dataset = SubsetDataset(dataset, val_indices)
    else:
        # 方法2：训练集和测试集分别使用不同数据集并按比例抽取
        # 加载训练集
        train_full_dataset = CitationDataset(TRAIN_DATASET_PATH, tokenizer, label2id, max_len)
        # 获取训练集样本的标签
        train_sample_labels = []
        with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                train_sample_labels.append(item['label'])
        # 抽取训练集的一部分
        if TRAIN_RATIO < 1.0:
            train_indices, _ = train_test_split(
                range(len(train_full_dataset)),
                train_size=TRAIN_RATIO,
                random_state=42,
                stratify=train_sample_labels
            )
            train_dataset = SubsetDataset(train_full_dataset, train_indices)
        else:
            train_dataset = train_full_dataset
        
        # 加载测试集
        test_full_dataset = CitationDataset(TEST_DATASET_PATH, tokenizer, label2id, max_len)
        # 获取测试集样本的标签
        test_sample_labels = []
        with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                test_sample_labels.append(item['label'])
        # 抽取测试集的一部分
        if TEST_RATIO < 1.0:
            val_indices, _ = train_test_split(
                range(len(test_full_dataset)),
                train_size=TEST_RATIO,
                random_state=42,
                stratify=test_sample_labels
            )
            val_dataset = SubsetDataset(test_full_dataset, val_indices)
        else:
            val_dataset = test_full_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBiLSTMCRF(model_name, len(label_list), hidden_size, dropout_rate).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ===== 训练 =====
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader)

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
            pbar.set_postfix({"loss": total_loss/(step+1)})

    # ======================
    # 评估
    # ======================
    model.eval()
    true_labels = []
    pred_labels = []
    tokenized_texts = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask']

            labels = batch['labels'].numpy()
            preds = model(input_ids, attention_mask.to(device))

            for pred, label, mask, input_id in zip(preds, labels, attention_mask, batch['input_ids']):
                seq_pred, seq_true, seq_tokens = [], [], []

                tokens = tokenizer.convert_ids_to_tokens(input_id)
                
                for p, l, m, token in zip(pred, label, mask, tokens):
                    if m == 0:
                        continue
                    seq_pred.append(id2label[p])
                    seq_true.append(id2label[l])
                    seq_tokens.append(token)

                pred_labels.append(seq_pred)
                true_labels.append(seq_true)
                tokenized_texts.append(seq_tokens)

    # ===== 输出具体识别细节CSV =====
    with open("predictions_vs_true.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["样本序号", "位置", "单词", "真实标签", "预测标签"])

        for sample_idx, (true_seq, pred_seq, tokens) in enumerate(zip(true_labels, pred_labels, tokenized_texts)):
            for pos_idx, (true_label, pred_label, token) in enumerate(zip(true_seq, pred_seq, tokens)):
                writer.writerow([sample_idx + 1, pos_idx + 1, token, true_label, pred_label])

    print("\n具体识别细节已保存到 predictions_vs_true.csv")

    # ===== span评估 =====
    results = compute_span_metrics(true_labels, pred_labels)

    print("\n===== Span级评估 =====")
    for label, (p, r, f1, support) in results.items():
        if label == '整体准确率':
            print(f"{label}: {p:.4f}")
        else:
            print(f"{label}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, 支持数={support}")

    # ===== 保存结果 =====
    with open("span_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["类别", "精确率", "召回率", "F1分数", "支持数"])

        for label, (p, r, f1, support) in results.items():
            writer.writerow([label, p, r, f1, support])

    print("\n结果已保存到 span_results.csv")


if __name__ == "__main__":
    main()