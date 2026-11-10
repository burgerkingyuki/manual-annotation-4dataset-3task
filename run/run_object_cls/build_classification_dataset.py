# -*- coding: utf-8 -*-
"""
引文对象：序列标注数据 → 多分类数据 转换脚本

为什么需要这一步
----------------
引文对象原始数据是给序列标注（BERT+CRF 等）用的，文本里带有片段标记：

    {"text": "The software package [@REFLAB] seems to be able to ...", "label": "其他"}

[@...] 标记只是「对象片段」的位置信息，标记本身不是文本内容。
多分类模型（BERT / BiLSTM / BiGRU / CNN / RNN / BERT+BiLSTM / BERT+BiGRU / BERT+CNN）
的输入格式和引文情感 / 引文功能完全一致，只认 text + label：

    {"text": "The software package REFLAB seems to be able to ...", "label": "其他"}

所以这里把 [@片段] 还原成 片段（标记去掉、片段文字保留），label 原样保留，
样本数、样本顺序、类别都不变，保证和序列标注实验是同一份数据的两种任务形式。

用法
----
    python build_classification_dataset.py
        # 批量：把 dataset/ 下所有 *-引文对象.jsonl 转成 *-引文对象多分类.jsonl

    python build_classification_dataset.py --stats
        # 批量转换 + 打印每个数据集的类别分布、文本长度统计

    python build_classification_dataset.py -i 源文件.jsonl -o 目标文件.jsonl
        # 只转一个文件（调试用）

    python build_classification_dataset.py --drop-no-span
        # 丢弃不含 [@...] 标记的样本（默认保留，转成纯文本样本）

一键运行脚本（run_all_models.py）发现配置的数据集不存在时会自动调用本脚本。
"""

import os
import json
import glob
import argparse

from config import BASE_DIR

# 引文对象片段标记以 "[@" 开头、以 "]" 结束，例如 "[@REFLAB]" → 片段内容 "REFLAB"
MARKER_OPEN = "[@"
MARKER_CLOSE = "]"

# 数据集目录与命名规则
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SOURCE_SUFFIX = "-引文对象.jsonl"
TARGET_SUFFIX = "-引文对象多分类.jsonl"


def strip_span_markers(text):
    """
    去掉序列标注用的 [@ ] 标记，只保留片段文字本身。

    标记之外的文本原样保留，只把 "[@片段]" 变成 "片段"。

    这里用逐字符扫描，而不是"正则匹配后替换"，原因是源数据里存在极少量畸形标记
    （4 个数据集各 1 条左右），用正则单遍替换会留下残渣：
      - 嵌套：  "... the [@global [@beam-search parsers]#O*] of ..."
                → 正则会把 "[@global [@beam-search parsers]" 当成一个片段，
                  替换后内层的 "[@" 还留在文本里
      - 未闭合："... washing buffer [@50 mM Tris HCl, pH 7.5, ...（后面没有 "]"
                → 正则匹配不到，"[@" 原样留下
    扫描规则：
      - 遇到 "[@"：丢弃这两个字符，片段深度 +1，标记计数 +1
      - 遇到 "]"，且深度 > 0：丢弃该字符，深度 -1
      - 其它字符：原样保留
    对全部正常标记，结果与正则替换逐字节一致（已在 4 个数据集上逐行核对过：
    只有上述 3 条畸形样本发生变化，其余 42033 条完全相同）。

    返回：(清洗后的文本, 标记数量)
    """
    if MARKER_OPEN not in text:
        return text, 0

    pieces = []
    depth = 0
    marker_count = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == '[' and index + 1 < length and text[index + 1] == '@':
            depth += 1
            marker_count += 1
            index += 2
            continue

        if char == ']' and depth > 0:
            depth -= 1
            index += 1
            continue

        pieces.append(char)
        index += 1

    return ''.join(pieces), marker_count


def target_path_of(source_path):
    """由序列标注文件名推出多分类文件名：X-引文对象.jsonl → X-引文对象多分类.jsonl"""
    name = os.path.basename(source_path)

    if name.endswith(SOURCE_SUFFIX):
        return os.path.join(os.path.dirname(source_path),
                            name[:-len(SOURCE_SUFFIX)] + TARGET_SUFFIX)

    # 不符合命名规则时，退化为加后缀
    return os.path.join(os.path.dirname(source_path),
                        os.path.splitext(name)[0] + "-多分类.jsonl")


def convert_file(input_path, output_path, drop_no_span=False, verbose=False):
    """
    把一个序列标注 jsonl 转成多分类 jsonl。

    :param input_path: 源文件，每行 {"text": 含 [@...] 标记, "label": 类别}
    :param output_path: 输出文件，每行 {"text": 纯文本, "label": 类别}
    :param drop_no_span: 是否丢弃不含 [@...] 标记的样本（默认保留）
    :return: 统计信息字典
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"源数据文件不存在: {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    stats = {
        "输入文件": input_path,
        "输出文件": output_path,
        "总行数": 0,
        "写出样本数": 0,
        "丢弃样本数": 0,
        "含标记样本数": 0,
        "无标记样本数": 0,
        "解析失败行数": 0,
        "类别分布": {},
        "文本长度": [],
    }

    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            stats["总行数"] += 1

            try:
                item = json.loads(line)
                text = item['text']
                label = item['label']
            except (json.JSONDecodeError, KeyError) as e:
                stats["解析失败行数"] += 1
                if verbose:
                    print(f"  [警告] 第 {line_no} 行解析失败，已跳过: {e}")
                continue

            clean_text, span_count = strip_span_markers(text)

            if span_count == 0:
                stats["无标记样本数"] += 1
                if drop_no_span:
                    stats["丢弃样本数"] += 1
                    continue
            else:
                stats["含标记样本数"] += 1

            # 与引文情感 / 引文功能数据完全相同的字段结构
            fout.write(json.dumps({"text": clean_text, "label": label},
                                  ensure_ascii=False) + '\n')

            stats["写出样本数"] += 1
            stats["类别分布"][label] = stats["类别分布"].get(label, 0) + 1
            stats["文本长度"].append(len(clean_text))

    return stats


def print_stats(stats):
    """打印单个文件的转换统计"""
    lengths = stats["文本长度"]

    print(f"\n  源文件: {os.path.basename(stats['输入文件'])}")
    print(f"  输出文件: {os.path.basename(stats['输出文件'])}")
    print(f"  样本数: {stats['写出样本数']}（含 [@...] 标记 {stats['含标记样本数']}，"
          f"无标记 {stats['无标记样本数']}）")

    if stats["丢弃样本数"]:
        print(f"  丢弃样本数: {stats['丢弃样本数']}")
    if stats["解析失败行数"]:
        print(f"  解析失败行数: {stats['解析失败行数']}")

    if stats["写出样本数"]:
        dist = "，".join(
            f"{k} {v}（{v / stats['写出样本数'] * 100:.2f}%）"
            for k, v in sorted(stats["类别分布"].items(), key=lambda x: -x[1])
        )
        print(f"  类别分布: {dist}")
    else:
        print("  类别分布: 无")

    if lengths:
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        print(f"  文本长度: 平均 {sum(lengths_sorted) / n:.1f}，"
              f"中位 {lengths_sorted[n // 2]}，最长 {lengths_sorted[-1]}，"
              f"超过 128 字符的占 "
              f"{sum(1 for x in lengths_sorted if x > 128) / n * 100:.2f}%")


def find_source_datasets(dataset_dir=None):
    """找出 dataset/ 下所有序列标注格式的引文对象数据集"""
    dataset_dir = dataset_dir or DATASET_DIR

    sources = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*引文对象*.jsonl"))):
        name = os.path.basename(path)
        # 跳过已经是多分类格式的文件
        if name.endswith(TARGET_SUFFIX) or name.endswith("-多分类.jsonl"):
            continue
        if name.endswith(SOURCE_SUFFIX):
            sources.append(path)

    return sources


def build_all_datasets(dataset_dir=None, drop_no_span=False, verbose=True):
    """
    批量转换 dataset/ 下所有 *-引文对象.jsonl → *-引文对象多分类.jsonl

    返回：[统计信息字典, ...]
    """
    dataset_dir = dataset_dir or DATASET_DIR
    sources = find_source_datasets(dataset_dir)

    if verbose:
        print("=" * 66)
        print("把引文对象序列标注数据转换为多分类数据")
        print(f"数据集目录: {dataset_dir}")
        print(f"找到 {len(sources)} 个序列标注数据集")
        print("=" * 66)

    all_stats = []
    for source in sources:
        target = target_path_of(source)
        stats = convert_file(source, target, drop_no_span=drop_no_span, verbose=verbose)
        all_stats.append(stats)
        if verbose:
            print_stats(stats)

    if verbose:
        total = sum(s["写出样本数"] for s in all_stats)
        print("\n" + "=" * 66)
        print(f"转换完成：{len(all_stats)} 个数据集，共 {total} 条样本")
        print("=" * 66)

    return all_stats


def ensure_datasets_exist(dataset_paths, verbose=True):
    """
    检查一批多分类数据集是否存在，缺失的自动从对应的序列标注数据生成。
    供 run_all_models.py 调用，保证「一键运行」不用手动先转换数据。
    """
    missing = [p for p in dataset_paths if not os.path.exists(p)]

    if not missing:
        return False

    if verbose:
        print("\n以下多分类数据集不存在，将自动从序列标注数据转换生成：")
        for path in missing:
            print(f"  - {path}")

    build_all_datasets(verbose=verbose)

    still_missing = [p for p in dataset_paths if not os.path.exists(p)]
    if still_missing:
        raise FileNotFoundError(
            "以下数据集在转换后仍不存在，请检查 dataset/ 下是否有对应的 "
            "*-引文对象.jsonl：\n  " + "\n  ".join(still_missing)
        )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="把引文对象序列标注数据（[@...] 标记）转换为多分类数据（text + label）"
    )
    parser.add_argument("-i", "--input", default=None,
                        help="源 jsonl 文件；不填则批量转换 dataset/ 下所有 *-引文对象.jsonl")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 jsonl 文件；不填则按命名规则生成 *-引文对象多分类.jsonl")
    parser.add_argument("--drop-no-span", action="store_true",
                        help="丢弃不含 [@...] 标记的样本（默认保留为纯文本样本）")
    parser.add_argument("--stats", action="store_true",
                        help="打印类别分布与文本长度统计")
    args = parser.parse_args()

    if args.input:
        # 单文件模式
        output = args.output or target_path_of(args.input)
        stats = convert_file(args.input, output,
                             drop_no_span=args.drop_no_span, verbose=True)
        print(f"\n转换完成：{stats['总行数']} 行 → {stats['写出样本数']} 条样本")
        print(f"输出文件: {output}")
        if args.stats:
            print_stats(stats)
        return

    # 批量模式
    build_all_datasets(drop_no_span=args.drop_no_span, verbose=True)


if __name__ == "__main__":
    main()
