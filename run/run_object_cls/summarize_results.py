# -*- coding: utf-8 -*-
"""
汇总：引文对象多分类 —— 把所有模型的结果合并成一张对比表

每个模型跑完后会在自己的输出目录下留一份 classification_results.csv：
    类别,精确率,召回率,F1分数,支持数
里面含「整体准确率」「加权平均」以及每个类别（其他 / 研究方法 / 研究问题 / 数据）的指标。

本脚本按 config.py 的 DATASET_METHOD 遍历同样的「训练集 × 测试集 × 模型」组合，
把这些结果汇总成一张长表：

    训练集,测试集,模型名称,整体准确率,加权平均精确率,加权平均召回率,加权平均F1分数,<类别>_精确率,...

输出位置：<BASE_DIR>/引文对象多分类/model_comparison.csv
（方法3 的 N*N 跨数据集实验，直接按「训练集-测试集」筛选即可得每种组合的加权F1）

用法：
    cd model/run/run_object_cls
    python summarize_results.py
    python summarize_results.py --output-root D:/tmp/试跑   # 汇总别处的试跑结果
"""

import os
import csv
import argparse

from config import (
    MODELS,
    RESULT_ROOT,
    iter_dataset_pairs,
    get_output_dir,
    get_summary_file,
)

# 长表里固定排在前面的列
BASE_COLUMNS = ["训练集", "测试集", "模型名称",
                "整体准确率", "加权平均精确率", "加权平均召回率", "加权平均F1分数"]


def extract_metrics(classification_results_file):
    """
    从 classification_results.csv 中提取指标

    文件格式：类别,精确率,召回率,F1分数,支持数
    """
    metrics = {}

    if not os.path.exists(classification_results_file):
        return metrics

    with open(classification_results_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["类别"]

            if category == "整体准确率":
                metrics["整体准确率"] = float(row["精确率"])
            elif category == "加权平均":
                metrics["加权平均精确率"] = float(row["精确率"])
                metrics["加权平均召回率"] = float(row["召回率"])
                metrics["加权平均F1分数"] = float(row["F1分数"])
            else:
                # 每个类别的精确率 / 召回率 / F1 / 支持数
                metrics[f"{category}_精确率"] = float(row["精确率"])
                metrics[f"{category}_召回率"] = float(row["召回率"])
                metrics[f"{category}_F1分数"] = float(row["F1分数"])
                metrics[f"{category}_支持数"] = float(row["支持数"])

    return metrics


def summarize(output_root=None, models=None, dataset_pairs=None, summary_file=None):
    """
    汇总所有结果，写长表 CSV，并在终端打印对比表

    :param output_root: 结果根目录（默认 config.RESULT_ROOT）
    :param models: 模型列表（默认 config.MODELS）
    :param dataset_pairs: 数据集组合（默认按 config.DATASET_METHOD 生成）
    :param summary_file: 汇总文件路径（默认 config.get_summary_file：
                         方法1 放在 <数据集名>/model_comparison.csv，其余放在根目录）
    """
    models = models or MODELS
    dataset_pairs = dataset_pairs or iter_dataset_pairs()
    output_root = output_root or RESULT_ROOT

    if summary_file is None:
        first_train, first_test = dataset_pairs[0]
        summary_file = get_summary_file(first_train, first_test, root=output_root)

    print(f"\n开始汇总所有模型结果（{len(dataset_pairs)} 个数据集组合 × {len(models)} 个模型）...\n")

    all_results = []
    missing_count = 0

    for train_name, test_name in dataset_pairs:
        print(f"[{train_name} → {test_name}]")

        for model_config in models:
            model_name = model_config["name"]

            # 结果目录：与 run_all_models.py 共用 config.get_output_dir，规则不会错位
            output_dir = get_output_dir(train_name, test_name, model_name, root=output_root)

            classification_results_file = os.path.join(output_dir, "classification_results.csv")
            metrics = extract_metrics(classification_results_file)

            if metrics:
                all_results.append({
                    "训练集": train_name,
                    "测试集": test_name,
                    "模型名称": model_name,
                    **metrics,
                })
                print(f"  成功提取模型 {model_name} 的结果")
            else:
                missing_count += 1
                print(f"  未找到模型 {model_name} 的结果文件（跳过）")

    if not all_results:
        print("\n没有提取到任何模型结果")
        return all_results

    # ===== 列顺序：固定列 + 其余按字母序 =====
    all_columns = set()
    for result in all_results:
        all_columns.update(result.keys())
    extra_columns = sorted(all_columns - set(BASE_COLUMNS))
    fieldnames = BASE_COLUMNS + extra_columns

    os.makedirs(os.path.dirname(os.path.abspath(summary_file)), exist_ok=True)
    with open(summary_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\n结果汇总完成，共 {len(all_results)} 条记录，已保存到: {summary_file}")
    if missing_count:
        print(f"（另有 {missing_count} 个「训练集×测试集×模型」组合没有结果文件，未计入）")

    print_comparison_tables(all_results)

    return all_results


def print_comparison_tables(all_results):
    """打印两张对比表：明细（按加权F1排序）与各模型跨组合平均加权F1"""
    sep = "=" * 100

    # ===== 表1：明细，按 训练集/测试集 分组、加权F1 降序 =====
    print("\n表1：各数据集组合下的模型对比（按加权平均F1降序）")
    print(sep)
    print(f"{'训练集':<22}{'测试集':<22}{'模型名称':<14}{'整体准确率':<12}{'加权F1':<10}")
    print(sep)

    pairs = []
    for result in all_results:
        key = (result["训练集"], result["测试集"])
        if key not in pairs:
            pairs.append(key)

    for train_name, test_name in pairs:
        rows = [r for r in all_results
                if r["训练集"] == train_name and r["测试集"] == test_name]
        rows.sort(key=lambda x: x.get("加权平均F1分数", 0), reverse=True)
        for result in rows:
            print(f"{result['训练集']:<22}{result['测试集']:<22}{result['模型名称']:<14}"
                  f"{result.get('整体准确率', 0):<12.4f}"
                  f"{result.get('加权平均F1分数', 0):<10.4f}")
        print("-" * 100)

    print(sep)

    # ===== 表2：每个模型在所有组合上的平均加权F1 =====
    model_scores = {}
    for result in all_results:
        model_scores.setdefault(result["模型名称"], []).append(
            result.get("加权平均F1分数", 0)
        )

    print(f"\n表2：各模型在全部 {len(pairs)} 个数据集组合上的平均加权F1（降序）")
    print(sep)
    print(f"{'模型名称':<16}{'平均加权F1':<14}{'组合数':<10}")
    print(sep)
    for model_name, scores in sorted(model_scores.items(),
                                     key=lambda x: sum(x[1]) / len(x[1]), reverse=True):
        print(f"{model_name:<16}{sum(scores) / len(scores):<14.4f}{len(scores):<10}")
    print(sep)


def summarize_all_datasets(output_root=None, models=None, out_file=None):
    """
    把结果根目录下「所有数据集目录」的结果合并成一张总表

    适用场景：方法1 下分数据集、一个一个跑（4 个数据集 × 8 个模型），
    每次跑完只在各自数据集目录里留一份 model_comparison.csv；
    本函数扫描全部数据集目录，生成一张横向对比总表：

        <结果根目录>/model_comparison_all_datasets.csv
        （数据集, 模型名称, 整体准确率, 加权平均F1, 各类别指标...）

    并在终端打印「模型 × 数据集」的加权F1 透视表。
    """
    output_root = output_root or RESULT_ROOT
    models = models or MODELS
    out_file = out_file or os.path.join(output_root, "model_comparison_all_datasets.csv")

    if not os.path.isdir(output_root):
        print(f"结果根目录不存在（还没跑过任何模型）: {output_root}")
        return []

    dataset_names = sorted(d for d in os.listdir(output_root)
                           if os.path.isdir(os.path.join(output_root, d)))
    if not dataset_names:
        print(f"结果根目录下还没有任何数据集目录: {output_root}")
        return []

    print(f"\n扫描 {len(dataset_names)} 个数据集目录 × {len(models)} 个模型 ...")

    rows = []
    for dataset_name in dataset_names:
        found = 0
        for model_config in models:
            model_name = model_config["name"]
            result_file = os.path.join(output_root, dataset_name, model_name,
                                       "classification_results.csv")
            metrics = extract_metrics(result_file)
            if metrics:
                rows.append({"数据集": dataset_name, "模型名称": model_name, **metrics})
                found += 1
        print(f"  {dataset_name}: 找到 {found} 个模型的结果")

    if not rows:
        print("\n没有提取到任何结果")
        return []

    # ===== 写总表 =====
    fixed = ["数据集", "模型名称", "整体准确率", "加权平均精确率", "加权平均召回率", "加权平均F1分数"]
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())
    fieldnames = fixed + sorted(all_columns - set(fixed))

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n总表已保存到: {out_file}（{len(rows)} 条记录）")

    # ===== 打印「模型 × 数据集」加权F1 透视表 =====
    model_names = [m["name"] for m in models]
    lookup = {(row["数据集"], row["模型名称"]): row.get("加权平均F1分数", 0) for row in rows}

    cell_width = max(12, max(len(d) * 2 for d in dataset_names) + 2)
    print("\n加权平均F1（行=模型，列=数据集）")
    sep = "=" * (16 + cell_width * len(dataset_names))
    print(sep)
    print(f"{'模型':<16}" + "".join(f"{d[:cell_width // 2]:<{cell_width}}" for d in dataset_names))
    print(sep)
    for model_name in model_names:
        line = f"{model_name:<16}"
        for dataset_name in dataset_names:
            value = lookup.get((dataset_name, model_name))
            line += f"{'—' if value is None else f'{value:.4f}':<{cell_width}}"
        print(line)
    print(sep)

    return rows


def main():
    parser = argparse.ArgumentParser(description="引文对象多分类：汇总所有模型结果")
    parser.add_argument("--output-root", default=None,
                        help="结果根目录（默认 <BASE_DIR>/引文对象多分类）")
    parser.add_argument("--all-datasets", action="store_true",
                        help="扫描结果根目录下所有数据集目录，生成跨数据集总表（4 数据集 × 8 模型用这个）")
    args = parser.parse_args()

    if args.all_datasets:
        summarize_all_datasets(output_root=args.output_root)
    else:
        summarize(output_root=args.output_root)


if __name__ == "__main__":
    main()
