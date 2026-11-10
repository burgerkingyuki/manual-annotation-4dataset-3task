# 配置文件
# 数据集路径
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 项目根目录

# 数据集构建方法：
# 1=单个数据集按比例划分
# 2=训练集和测试集分别使用不同数据集并按比例抽取
# 3=N个训练集和N个测试集的N*N批量训练
DATASET_METHOD = 1

# 训练集和测试集使用比例（在自身数据集中抽取）
TRAIN_RATIO = 1    # 训练集使用比例
TEST_RATIO = 1     # 测试集使用比例

# ===== 统一数据集配置接口 =====
# 格式：{自定义名: 路径}
# 方法1：TRAIN_DATASETS和TEST_DATASETS使用相同的数据集（同名）
# 方法2：TRAIN_DATASETS和TEST_DATASETS各包含一个数据集
# 方法3：TRAIN_DATASETS和TEST_DATASETS各包含N个数据集，实现N*N批量训练
TRAIN_DATASETS = {
    "SDCF-引文对象-训练": os.path.join(BASE_DIR, "dataset", "SDCF-引文对象.jsonl"),
    # "ACL-W-引文对象-训练": os.path.join(BASE_DIR, "dataset", "ACL-W-引文对象.jsonl"),
    # "ACTSCI-引文对象-训练": os.path.join(BASE_DIR, "dataset", "ACTSCI-引文对象.jsonl"),
    # "ICLR-引文对象-训练": os.path.join(BASE_DIR, "dataset", "ICLR-引文对象.jsonl"),
}

TEST_DATASETS = {
    "SDCF-引文对象-测试": os.path.join(BASE_DIR, "dataset", "SDCF-引文对象.jsonl"),
    # "ACL-W-引文对象-测试": os.path.join(BASE_DIR, "dataset", "ACL-W-引文对象.jsonl"),
    # "ACTSCI-引文对象-测试": os.path.join(BASE_DIR, "dataset", "ACTSCI-引文对象.jsonl"),
    # "ICLR-引文对象-测试": os.path.join(BASE_DIR, "dataset", "ICLR-引文对象.jsonl"),
}

# 模型配置
MODELS = [
    {
        "name": "BERT+CRF",
        "script_path": os.path.join(BASE_DIR, "model", "BERT+CRF", "bert_crf.py"),
    },
    {
        "name": "BERT+Bi-LSTM+CRF",
        "script_path": os.path.join(BASE_DIR, "model", "BERT+Bi-LSTM+CRF", "bert_bilstm_crf.py"),
    },
    {
        "name": "BERT+BiGRU+CRF",
        "script_path": os.path.join(BASE_DIR, "model", "BERT+BiGRU+CRF", "bert_bigru_crf.py"),
    },
    # {
    #     "name": "BiLSTM+CRF",
    #     "script_path": os.path.join(BASE_DIR, "model", "BiLSTM+CRF", "bilstm_crf.py"),
    # },
    # {
    #     "name": "BiGRU+CRF",
    #     "script_path": os.path.join(BASE_DIR, "model", "BiGRU+CRF", "bigru_crf.py"),
    # }
]

# ===== 旧配置（保留向后兼容，建议使用新配置） =====
# 方法1配置
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "SDCF-引文对象.jsonl")
DATASET_NAME = "SDCF-引文对象"
# 方法2配置
TRAIN_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "SDCF-引文对象-train.jsonl")  # 训练集路径
TEST_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "SDCF-引文对象-test.jsonl")    # 测试集路径
