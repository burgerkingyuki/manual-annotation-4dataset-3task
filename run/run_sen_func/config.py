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
TRAIN_RATIO = 1   # 训练集使用比例
TEST_RATIO = 1    # 测试集使用比例

# ===== 统一数据集配置接口 =====
# 格式：{自定义名: 路径}
# 方法1：TRAIN_DATASETS和TEST_DATASETS使用相同的数据集（同名）
# 方法2：TRAIN_DATASETS和TEST_DATASETS各包含一个数据集
# 方法3：TRAIN_DATASETS和TEST_DATASETS各包含N个数据集，实现N*N批量训练
TRAIN_DATASETS = {
    "SDCF-引文情感-训练": os.path.join(BASE_DIR, "dataset", "TEST1.jsonl"),
    # "ACL-W-引文情感": os.path.join(BASE_DIR, "dataset", "ACL-W-引文情感.jsonl"),
    # "ACTSCI-引文情感": os.path.join(BASE_DIR, "dataset", "ACTSCI-引文情感.jsonl"),
    # "ICLR-引文情感": os.path.join(BASE_DIR, "dataset", "ICLR-引文情感.jsonl"),
}

TEST_DATASETS = {
    "SDCF-引文情感-测试": os.path.join(BASE_DIR, "dataset", "TEST2.jsonl"),
    # "2-SDCF-引文情感-2": os.path.join(BASE_DIR, "dataset", "TEST1.jsonl"),
    # "ACL-W-引文情感": os.path.join(BASE_DIR, "dataset", "ACL-W-引文情感.jsonl"),
    # "ACTSCI-引文情感": os.path.join(BASE_DIR, "dataset", "ACTSCI-引文情感.jsonl"),
    # "ICLR-引文情感": os.path.join(BASE_DIR, "dataset", "ICLR-引文情感.jsonl"),
}

# 模型配置
MODELS = [
    # {
    #     "name": "BiLSTM",
    #     "script_path": os.path.join(BASE_DIR, "model", "BiLSTM", "bilstm.py"),
    # },
    # {
    #     "name": "BiGRU",
    #     "script_path": os.path.join(BASE_DIR, "model", "BiGRU", "bigru.py"),
    # },
    # {
    #     "name": "CNN",
    #     "script_path": os.path.join(BASE_DIR, "model", "CNN", "cnn.py"),
    # },
    {
        "name": "RNN",
        "script_path": os.path.join(BASE_DIR, "model", "RNN", "rnn.py"),
    },
    # {
    #     "name": "BERT",
    #     "script_path": os.path.join(BASE_DIR, "model", "BERT", "bert.py"),
    # },
    # {
    #     "name": "BERT+BiLSTM",
    #     "script_path": os.path.join(BASE_DIR, "model", "BERT+BiLSTM", "bert_bilstm.py"),
    # },
    # {
    #     "name": "BERT+BiGRU",
    #     "script_path": os.path.join(BASE_DIR, "model", "BERT+BiGRU", "bert_bigru.py"),
    # },
    # {
    #     "name": "BERT+CNN",
    #     "script_path": os.path.join(BASE_DIR, "model", "BERT+CNN", "bert_cnn.py"),
    # }
]

# ===== 旧配置（保留向后兼容，建议使用新配置） =====
# 方法1配置
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "SDCF-引文情感.jsonl")
DATASET_NAME = "SDCF-引文情感"
# 方法2配置
TRAIN_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "SDCF-引文情感-train.jsonl")  # 训练集路径
TEST_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "SDCF-引文情感-test.jsonl")    # 测试集路径
