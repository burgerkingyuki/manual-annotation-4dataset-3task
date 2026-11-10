# -*- coding: utf-8 -*-
"""
配置文件：引文对象 —— 多分类任务

和 run_object（引文对象序列标注）、run_sen_func（引文情感/功能分类）完全一致的配置接口，
只是任务换成了「引文对象多分类」，模型换成分类模型。

=========================================================================
一、为什么要单独做一个多分类任务
=========================================================================
引文对象原本是序列标注任务：数据里用 [@...] 标出对象片段，
模型（BERT+CRF 等）把片段转成 BIO 标签训练。

多分类任务没有「片段」概念，一条样本只能有一个类别，所以输入数据必须先换成
和引文情感 / 引文功能完全一致的结构：

    序列标注：{"text": "We use the [@SC-LSTM] proposed by REFLAB", "label": "研究方法"}
    多分类  ：{"text": "We use the SC-LSTM proposed by REFLAB",     "label": "研究方法"}

即把 [@xxx] 还原成 xxx（标记去掉、片段文字保留），label 原样保留。
转换工作由本目录的 build_classification_dataset.py 完成，
生成的数据集放在 dataset/ 下，命名规则：
    <数据集名>-引文对象.jsonl  →  <数据集名>-引文对象多分类.jsonl

=========================================================================
二、数据集构建方法（与使用手册一致）
=========================================================================
1 = 单个数据集按比例划分（TRAIN_DATASETS 与 TEST_DATASETS 用同一个数据集）
2 = 训练集和测试集分别使用不同数据集并按比例抽取
3 = N 个训练集 × N 个测试集的 N*N 批量训练（跨领域/跨数据集实验用这个）

本任务对应「说明-ynt」里引文对象的 4×4 跨数据集实验：
    ACL / ICLR / SDCF / ACTSCI 为训练集，各测试 4 个数据集。
"""

import os

# 项目根目录：.../传统模型-服务器用
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# =============================================================================
# 当前实验：跨数据集（N*N）
#   方法3 = 4 个训练集 × 4 个测试集，两两组合批量训练
#   SKIP_SAME_DATASET = True → 跳过对角线（训练集 == 测试集）的组合：
#     同数据集的结果请用方法1 跑（分层 80/20，训练验证互斥）；
#     方法3 的对角线是同一个文件抽两次（0.8 与 0.2、同一随机种子），
#     测试集必然落在训练集里，数字虚高、没有意义。
#   → 本配置实际跑 4×4-4 = 12 个组合 × 8 个模型 = 96 次训练
#
# 想做同数据集实验：DATASET_METHOD = 1，只启用一个数据集（见文件末尾旧配置示例）
# 想连对角线一起跑（不推荐）：SKIP_SAME_DATASET = False → 16 个组合
# =============================================================================

# 数据集构建方法：1=单个数据集按比例划分；2=训练/测试分别用不同数据集；3=N*N 批量训练
DATASET_METHOD = 3

# 方法3 下这两个比例才生效：训练集文件只抽 TRAIN_RATIO，测试集文件只抽 TEST_RATIO
# （< 1.0 才抽样；设 1.0 表示用该数据集的全量）
# 0.8 / 0.2 与已有的跨域基线（run_object、run_sen_func）保持一致，方便对照；
# 若想让跨域指标更稳（抽样方差更小），可以把 TEST_RATIO 改成 1.0
TRAIN_RATIO = 0.8    # 训练集使用比例
TEST_RATIO = 0.2     # 测试集使用比例

# 方法3 是否跳过「训练集 == 测试集」的对角线组合
SKIP_SAME_DATASET = True

# 任务名：结果根目录为 <BASE_DIR>/引文对象多分类
# 与 run_sen_func 的「引文情感」、run_object 的「引文对象」并列，便于区分统计
TASK_NAME = "引文对象多分类"

# ===== 统一数据集配置接口 =====
# 格式：{自定义名: 路径}
# 方法1：TRAIN_DATASETS 和 TEST_DATASETS 使用相同的数据集（同名）
# 方法2：TRAIN_DATASETS 和 TEST_DATASETS 各包含一个数据集
# 方法3：TRAIN_DATASETS 和 TEST_DATASETS 各包含 N 个数据集，实现 N*N 批量训练
# 注意：这里指向的是「多分类」格式的数据（不含 [@...] 标记），
#      由 build_classification_dataset.py 从同名的「-引文对象.jsonl」转换得到。
# 【跨数据集实验】4 个数据集全部启用
TRAIN_DATASETS = {
    "SDCF-引文对象多分类": os.path.join(BASE_DIR, "dataset", "SDCF-引文对象多分类.jsonl"),
    "ACL-W-引文对象多分类": os.path.join(BASE_DIR, "dataset", "ACL-W-引文对象多分类.jsonl"),
    "ACTSCI-引文对象多分类": os.path.join(BASE_DIR, "dataset", "ACTSCI-引文对象多分类.jsonl"),
    "ICLR-引文对象多分类": os.path.join(BASE_DIR, "dataset", "ICLR-引文对象多分类.jsonl"),
}

# 方法3 需要「测试集」也能轮换，所以这里同样放全 4 个
#（与上面做笛卡尔积 4×4，再由 SKIP_SAME_DATASET 去掉 4 个对角线 → 12 个组合）
TEST_DATASETS = {
    "SDCF-引文对象多分类": os.path.join(BASE_DIR, "dataset", "SDCF-引文对象多分类.jsonl"),
    "ACL-W-引文对象多分类": os.path.join(BASE_DIR, "dataset", "ACL-W-引文对象多分类.jsonl"),
    "ACTSCI-引文对象多分类": os.path.join(BASE_DIR, "dataset", "ACTSCI-引文对象多分类.jsonl"),
    "ICLR-引文对象多分类": os.path.join(BASE_DIR, "dataset", "ICLR-引文对象多分类.jsonl"),
}

# 模型配置：与「引文情感 / 引文功能」完全相同的 8 个分类模型
# （序列标注的 CRF 系列模型不能用于多分类，不要混进来）
MODELS = [
    {
        "name": "BERT",
        "script_path": os.path.join(BASE_DIR, "model", "BERT", "bert.py"),
    },
    {
        "name": "BERT+BiLSTM",
        "script_path": os.path.join(BASE_DIR, "model", "BERT+BiLSTM", "bert_bilstm.py"),
    },
    {
        "name": "BERT+BiGRU",
        "script_path": os.path.join(BASE_DIR, "model", "BERT+BiGRU", "bert_bigru.py"),
    },
    {
        "name": "BERT+CNN",
        "script_path": os.path.join(BASE_DIR, "model", "BERT+CNN", "bert_cnn.py"),
    },
    {
        "name": "BiLSTM",
        "script_path": os.path.join(BASE_DIR, "model", "BiLSTM", "bilstm.py"),
    },
    {
        "name": "BiGRU",
        "script_path": os.path.join(BASE_DIR, "model", "BiGRU", "bigru.py"),
    },
    {
        "name": "CNN",
        "script_path": os.path.join(BASE_DIR, "model", "CNN", "cnn.py"),
    },
    {
        "name": "RNN",
        "script_path": os.path.join(BASE_DIR, "model", "RNN", "rnn.py"),
    },
]

# ===== 预训练 BERT 路径 =====
# 模型脚本里读环境变量 BERT_MODEL_PATH（由 run_all_models.py 注入），本机和服务器都不用改模型代码：
#   - 本机 Windows：E:\MODEL\bert-base-cased
#   - 服务器 autodl：/root/autodl-tmp/model/bert-base-cased
# 也可以在命令行用环境变量强制指定，例如：set BERT_MODEL_PATH=D:\models\bert-base-cased
LOCAL_BERT_PATH = r"E:\MODEL\bert-base-cased"      # ← 本机改这一行
SERVER_BERT_PATH = "/root/autodl-tmp/model/bert-base-cased"
BERT_MODEL_PATH = os.environ.get(
    "BERT_MODEL_PATH",
    LOCAL_BERT_PATH if os.path.isdir(LOCAL_BERT_PATH) else SERVER_BERT_PATH,
)

# ===== 结果目录 =====
# 结果根目录：<BASE_DIR>/引文对象多分类
RESULT_ROOT = os.path.join(BASE_DIR, TASK_NAME)

# 结果汇总文件（长表：训练集 × 测试集 × 模型）
SUMMARY_FILE = os.path.join(RESULT_ROOT, "model_comparison.csv")


# =============================================================================
# 以下两个函数供 run_all_models.py 与 summarize_results.py 共用，
# 保证「跑模型」和「汇总」用的目录规则永远一致（不会出现两边路径对不上的情况）
# =============================================================================
def iter_dataset_pairs():
    """
    返回要运行的数据集组合 [(训练集名, 测试集名), ...]

    方法1/2 只有一组；方法3 是 N*N 全部组合
    （SKIP_SAME_DATASET = True 时去掉训练集与测试集同名的对角线组合）。
    """
    if DATASET_METHOD == 3:
        train_names = list(TRAIN_DATASETS.keys())
        test_names = list(TEST_DATASETS.keys())
        pairs = [(train, test) for train in train_names for test in test_names]
        if SKIP_SAME_DATASET:
            pairs = [(train, test) for train, test in pairs if train != test]
        return pairs

    return [(list(TRAIN_DATASETS.keys())[0], list(TEST_DATASETS.keys())[0])]


def get_output_dir(train_name, test_name, model_name, root=None):
    """
    结果目录：<结果根目录>/<数据集组合>/<模型名>
    结果根目录默认 RESULT_ROOT（即 <BASE_DIR>/引文对象多分类）

    方法1：.../引文对象多分类/<数据集名>/<模型名>
    方法2/3：.../引文对象多分类/<训练集名>-<测试集名>/<模型名>
    """
    root = root or RESULT_ROOT

    if DATASET_METHOD == 1:
        sub_dir = list(TRAIN_DATASETS.keys())[0]
    else:
        sub_dir = f"{train_name}-{test_name}"

    return os.path.join(root, sub_dir, model_name)


def get_summary_file(train_name, test_name, root=None):
    """
    汇总文件路径

    - 方法1（一次只跑一个数据集）：放在该数据集目录下
      <结果根目录>/<数据集名>/model_comparison.csv
      —— 这样 4 个数据集各自一份，换数据集时不会覆盖上一个的汇总
    - 方法2/3（一个或多个组合一起跑）：
      <结果根目录>/model_comparison.csv
    """
    root = root or RESULT_ROOT

    if DATASET_METHOD == 1:
        return os.path.join(root, list(TRAIN_DATASETS.keys())[0], "model_comparison.csv")

    return os.path.join(root, "model_comparison.csv")


# ===== 旧配置（保留向后兼容）=====
# 模型脚本（model/BERT/bert.py 等）在被一键运行脚本调起时，除了环境变量，
# 还会尝试 from config import ...，因此这里保留这些名字，
# 保证单独运行模型脚本、或环境变量缺失时也能读到合理默认值。
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "SDCF-引文对象多分类.jsonl")
DATASET_NAME = "SDCF-引文对象多分类"
TRAIN_DATASET_PATH = DATASET_PATH
TEST_DATASET_PATH = DATASET_PATH
