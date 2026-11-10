import os
import csv
import json
from config import MODELS, SUMMARY_FILE


def extract_metrics(classification_results_file):
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
                metrics[f"{category}_精确率"] = float(row["精确率"])
                metrics[f"{category}_召回率"] = float(row["召回率"])
                metrics[f"{category}_F1分数"] = float(row["F1分数"])
    return metrics


def main():
    print("\n开始汇总所有模型结果...\n")
    
    all_results = []
    
    for model_config in MODELS:
        model_name = model_config["name"]
        output_dir = model_config["output_dir"]
        
        classification_results_file = os.path.join(output_dir, "classification_results.csv")
        
        print(f"正在处理模型: {model_name}")
        
        metrics = extract_metrics(classification_results_file)
        
        if metrics:
            result = {
                "模型名称": model_name,
                **metrics
            }
            all_results.append(result)
            print(f"成功提取模型 {model_name} 的结果")
        else:
            print(f"无法提取模型 {model_name} 的结果")
    
    # 保存汇总结果到CSV
    if all_results:
        # 获取所有列名
        all_columns = set()
        for result in all_results:
            all_columns.update(result.keys())
        all_columns = sorted(all_columns)
        
        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        
        print(f"\n结果汇总完成，已保存到: {SUMMARY_FILE}")
        
        # 打印对比表格
        print("\n" + "="*120)
        print(f"{'模型名称':<15} {'整体准确率':<12} {'加权平均精确率':<15} {'加权平均召回率':<15} {'加权平均F1分数':<15}")
        print("="*120)
        
        for result in sorted(all_results, key=lambda x: x["整体准确率"], reverse=True):
            print(f"{result['模型名称']:<15} {result.get('整体准确率', 0):<12.4f} {result.get('加权平均精确率', 0):<15.4f} {result.get('加权平均召回率', 0):<15.4f} {result.get('加权平均F1分数', 0):<15.4f}")
        
        print("="*120)
    else:
        print("\n没有提取到任何模型结果")


if __name__ == "__main__":
    main()
