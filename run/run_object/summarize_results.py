import os
import csv
from config import MODELS, SUMMARY_FILE


def extract_metrics(span_results_file):
    """
    从span_results.csv文件中提取指标
    """
    metrics = {}

    if not os.path.exists(span_results_file):
        return metrics

    with open(span_results_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["类别"]
            if category in ["整体准确率", "加权平均"]:
                metrics[category] = {
                    "精确率": float(row["精确率"]),
                    "召回率": float(row["召回率"]),
                    "F1分数": float(row["F1分数"])
                }
            else:
                if "类别指标" not in metrics:
                    metrics["类别指标"] = []
                metrics["类别指标"].append({
                    "类别": category,
                    "精确率": float(row["精确率"]),
                    "召回率": float(row["召回率"]),
                    "F1分数": float(row["F1分数"]),
                    "支持数": int(row["支持数"])
                })

    return metrics


def summarize_results():
    """
    汇总所有模型的结果
    """
    print("开始汇总模型结果...")

    # 创建结果列表
    summary = []

    for model_config in MODELS:
        model_name = model_config["name"]
        output_dir = model_config["output_dir"]

        # 检查结果文件是否存在
        span_results_file = os.path.join(output_dir, "span_results.csv")
        if not os.path.exists(span_results_file):
            print(f"模型 {model_name} 结果文件不存在")
            continue

        # 提取指标
        metrics = extract_metrics(span_results_file)

        # 添加到汇总
        result = {
            "模型名称": model_name
        }

        if "整体准确率" in metrics:
            result["整体准确率"] = metrics["整体准确率"]["精确率"]

        if "加权平均" in metrics:
            result["加权平均精确率"] = metrics["加权平均"]["精确率"]
            result["加权平均召回率"] = metrics["加权平均"]["召回率"]
            result["加权平均F1分数"] = metrics["加权平均"]["F1分数"]

        summary.append(result)

    # 保存汇总结果
    if summary:
        os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["模型名称", "整体准确率", "加权平均精确率", "加权平均召回率", "加权平均F1分数"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for row in summary:
                writer.writerow(row)

        print(f"结果汇总已保存到 {SUMMARY_FILE}")
    else:
        print("没有可汇总的结果")


if __name__ == "__main__":
    summarize_results()