# -*- coding: utf-8 -*-
"""
一键运行：引文对象 —— 多分类任务

和 run_object / run_sen_func 的 run_all_models.py 是同一套逻辑、同一套配置接口，
只是任务换成「引文对象多分类」、模型换成 8 个分类模型。

做的事情：
  1. 检查多分类数据集，缺失时自动从 dataset/ 下的 *-引文对象.jsonl 转换生成；
  2. 按 config.DATASET_METHOD 生成数据集组合（方法3 = N*N 全部组合）；
  3. 逐个模型运行，数据集通过环境变量注入给模型脚本
     （DATASET_METHOD / DATASET_PATH / TRAIN_DATASET_PATH / TEST_DATASET_PATH ...）；
  4. 回收每个模型的 predictions_vs_true.csv 与 classification_results.csv，
     放到 <BASE_DIR>/引文对象多分类/<数据集组合>/<模型>/ 下；
  5. 全部跑完后自动汇总成 model_comparison.csv（可用 --skip-summary 关闭）。

用法：
    cd model/run/run_object_cls
    python run_all_models.py                    # 按 config.py 一键跑完
    python run_all_models.py --list             # 只看会跑哪些组合和模型
    python run_all_models.py --models RNN,CNN   # 只跑指定模型（试跑用）
    python run_all_models.py --output-root D:/tmp/试跑   # 结果写到别处
    python run_all_models.py --skip-build       # 不自动生成数据集（缺了直接报错）
    python run_all_models.py --skip-summary     # 只跑模型不汇总
"""

import os
import sys
import time
import shutil
import argparse
import subprocess

from config import (
    MODELS,
    DATASET_METHOD,
    TRAIN_DATASETS,
    TEST_DATASETS,
    TASK_NAME,
    RESULT_ROOT,
    BERT_MODEL_PATH,
    iter_dataset_pairs,
    get_output_dir,
)

# 添加当前目录到 PYTHONPATH：模型脚本会 from config import ...（读取 TRAIN_RATIO / TEST_RATIO）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def get_dataset_config(train_name, test_name):
    """
    根据数据集方法获取要传给模型脚本的环境变量配置

    与 run_object / run_sen_func 完全一致：
      方法1 → DATASET_PATH（单数据集按比例划分）
      方法2 → TRAIN_DATASET_PATH + TEST_DATASET_PATH
      方法3 → 与方法2相同，只是由外层按 N*N 组合逐个传入
    """
    if DATASET_METHOD == 1:
        # 方法1：单个数据集按比例划分
        dataset_name = list(TRAIN_DATASETS.keys())[0]
        dataset_path = TRAIN_DATASETS[dataset_name]
        return {
            "DATASET_METHOD": 1,
            "DATASET_PATH": dataset_path,
            "DATASET_NAME": dataset_name,
            "TRAIN_DATASET_NAME": dataset_name,
            "TEST_DATASET_NAME": dataset_name,
        }
    elif DATASET_METHOD == 2:
        # 方法2：训练集和测试集分别使用不同数据集
        train_dataset_name = list(TRAIN_DATASETS.keys())[0]
        test_dataset_name = list(TEST_DATASETS.keys())[0]
        return {
            "DATASET_METHOD": 2,
            "TRAIN_DATASET_PATH": TRAIN_DATASETS[train_dataset_name],
            "TEST_DATASET_PATH": TEST_DATASETS[test_dataset_name],
            "TRAIN_DATASET_NAME": train_dataset_name,
            "TEST_DATASET_NAME": test_dataset_name,
        }
    elif DATASET_METHOD == 3:
        # 方法3：N*N批量训练
        return {
            "DATASET_METHOD": 2,  # 模型内部使用方法2的逻辑
            "TRAIN_DATASET_PATH": TRAIN_DATASETS[train_name],
            "TEST_DATASET_PATH": TEST_DATASETS[test_name],
            "TRAIN_DATASET_NAME": train_name,
            "TEST_DATASET_NAME": test_name,
        }
    else:
        raise ValueError(f"未知的 DATASET_METHOD: {DATASET_METHOD}（只支持 1/2/3）")


def build_env(train_name, test_name):
    """
    构造传给模型脚本的环境变量

    - PYTHONPATH：让模型脚本能 import 到本目录的 config.py（读 TRAIN_RATIO/TEST_RATIO）；
      这里用 os.pathsep 拼接，Windows 是 ";"、Linux 是 ":"，服务器上也能生效。
    - DATASET_*：数据集配置
    - PYTHONIOENCODING：统一子进程输出编码，避免中文日志乱码
    """
    env = os.environ.copy()

    if 'PYTHONPATH' in env and env['PYTHONPATH']:
        env['PYTHONPATH'] = os.pathsep.join([current_dir, env['PYTHONPATH']])
    else:
        env['PYTHONPATH'] = current_dir

    for key, value in get_dataset_config(train_name, test_name).items():
        env[key] = str(value)

    # 预训练 BERT 路径（本机 E:\MODEL / 服务器 /root/autodl-tmp/model 自动切换）
    env['BERT_MODEL_PATH'] = BERT_MODEL_PATH

    env['PYTHONIOENCODING'] = 'utf-8'

    return env


def describe_device():
    """
    说明模型脚本会用什么设备跑

    模型脚本里是 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')，
    有 GPU 就自动用 GPU，不需要改代码。这里只把它提前打印出来，
    免得在没装 CUDA 版 torch 的机器上白跑几小时。
    """
    try:
        import torch
    except Exception as e:
        return f"未知（torch 导入失败: {e}）"

    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
        except Exception:
            gpu_name, count = "未知型号", 0
        return f"GPU（{gpu_name}，共 {count} 张卡，CUDA {torch.version.cuda}）"
    return f"CPU（torch {torch.__version__}，未检测到可用 CUDA）"


def run_model(model_config, train_name, test_name, output_root=None):
    """
    运行单个模型（某训练集 × 某测试集）
    """
    model_name = model_config["name"]
    script_path = model_config["script_path"]

    # 构建输出目录名（规则与 summarize_results.py 共用 config.get_output_dir）
    output_dir = get_output_dir(train_name, test_name, model_name, root=output_root)

    print(f"\n{'=' * 66}")
    print(f"开始运行模型: {model_name}")
    print(f"训练集: {train_name}")
    print(f"测试集: {test_name}")
    print(f"输出目录: {output_dir}")
    print(f"{'=' * 66}")

    if not os.path.exists(script_path):
        print(f"[跳过] 模型脚本不存在: {script_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # 保存当前工作目录
    original_cwd = os.getcwd()
    # 切换到模型脚本所在目录（模型脚本把 CSV 写在当前目录）
    script_dir = os.path.dirname(script_path)

    start_time = time.time()
    try:
        os.chdir(script_dir)

        result = subprocess.run(
            [sys.executable, os.path.basename(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=36000,  # 10 小时超时
            env=build_env(train_name, test_name)
        )

        elapsed = time.time() - start_time

        # 保存标准输出和错误输出
        with open(os.path.join(output_dir, "output.log"), "w", encoding="utf-8") as f:
            f.write(f"模型: {model_name}\n")
            f.write(f"训练集: {train_name}\n")
            f.write(f"测试集: {test_name}\n")
            f.write(f"脚本: {script_path}\n")
            f.write(f"耗时: {elapsed:.2f} 秒\n")
            f.write(f"返回码: {result.returncode}\n\n")
            f.write("标准输出:\n")
            f.write(result.stdout or "")
            f.write("\n\n错误输出:\n")
            f.write(result.stderr or "")

        # 复制生成的CSV文件到输出目录
        csv_files = ["predictions_vs_true.csv", "classification_results.csv"]
        copied = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                shutil.copy(csv_file, os.path.join(output_dir, csv_file))
                os.remove(csv_file)  # 删除原文件
                copied.append(csv_file)

        if result.returncode != 0:
            print(f"[失败] 模型 {model_name} 返回码 {result.returncode}，耗时 {elapsed:.2f} 秒")
            print(f"        错误详情见 {os.path.join(output_dir, 'output.log')}")
            return False

        if not copied:
            print(f"[警告] 模型 {model_name} 未产出结果文件，耗时 {elapsed:.2f} 秒")
            print(f"        详情见 {os.path.join(output_dir, 'output.log')}")
            return False

        print(f"模型 {model_name} 运行完成，耗时: {elapsed:.2f} 秒")
        return True

    except subprocess.TimeoutExpired:
        print(f"模型 {model_name} 运行超时（超过 10 小时）")
        return False
    except Exception as e:
        print(f"模型 {model_name} 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 切换回原工作目录
        os.chdir(original_cwd)


def select_models(model_names):
    """按名称筛选模型，保持 config 中的顺序"""
    if not model_names:
        return MODELS

    wanted = [n.strip() for n in model_names if n.strip()]
    known = {m["name"] for m in MODELS}
    unknown = [n for n in wanted if n not in known]
    if unknown:
        print(f"[警告] 配置里没有这些模型，已忽略: {', '.join(unknown)}")
        print(f"        可选模型: {', '.join(m['name'] for m in MODELS)}")

    selected = [m for m in MODELS if m["name"] in wanted]
    if not selected:
        raise SystemExit("没有匹配到任何可运行的模型")
    return selected


def main():
    parser = argparse.ArgumentParser(description="引文对象多分类：一键运行所有分类模型")
    parser.add_argument("--models", default=None,
                        help="只运行指定模型，逗号分隔，例如: RNN,CNN,BERT")
    parser.add_argument("--output-root", default=None,
                        help="结果根目录（默认 <BASE_DIR>/引文对象多分类），试跑时可指向临时目录")
    parser.add_argument("--skip-build", action="store_true",
                        help="不自动生成多分类数据集（数据集缺失时直接报错）")
    parser.add_argument("--skip-summary", action="store_true",
                        help="只运行模型，不自动汇总")
    parser.add_argument("--list", action="store_true",
                        help="只打印数据集组合与模型清单，不运行")
    args = parser.parse_args()

    models = select_models(args.models.split(",") if args.models else None)
    dataset_pairs = iter_dataset_pairs()
    output_root = args.output_root

    # ===== 参数总览 =====
    print(f"\n{TASK_NAME} —— 一键运行")
    print(f"数据集构建方法: {DATASET_METHOD}"
          f"（{'单个数据集按比例划分' if DATASET_METHOD == 1 else '训练/测试分别指定' if DATASET_METHOD == 2 else 'N*N 批量训练'}）")
    print(f"训练集数量: {len(TRAIN_DATASETS)}，测试集数量: {len(TEST_DATASETS)}")
    print(f"数据集组合数量: {len(dataset_pairs)}")
    print(f"模型数量: {len(models)}")
    print(f"结果根目录: {output_root or RESULT_ROOT}")
    # 预训练模型路径：本机/服务器自动切换，路径不对时在这里就能看出来
    print(f"预训练模型: {BERT_MODEL_PATH}"
          f" {'（路径存在）' if os.path.isdir(BERT_MODEL_PATH) else '（！！路径不存在，请改 config.py 的 LOCAL_BERT_PATH）'}")
    print(f"运行设备: {describe_device()}")

    if args.list:
        print("\n数据集组合:")
        for index, (train_name, test_name) in enumerate(dataset_pairs, 1):
            print(f"  {index}. {train_name} → {test_name}")
        print("\n模型:")
        for index, model_config in enumerate(models, 1):
            print(f"  {index}. {model_config['name']:<14} {model_config['script_path']}")
        return

    # ===== 步骤1：准备多分类数据集 =====
    if not args.skip_build:
        from build_classification_dataset import ensure_datasets_exist
        all_dataset_paths = list(dict.fromkeys(
            list(TRAIN_DATASETS.values()) + list(TEST_DATASETS.values())
        ))
        print("\n" + "-" * 66)
        print("步骤1/3：检查多分类数据集")
        print("-" * 66)
        ensure_datasets_exist(all_dataset_paths, verbose=True)

    # ===== 步骤2：运行所有模型 =====
    print("\n" + "-" * 66)
    print(f"步骤2/3：运行 {len(dataset_pairs)} 个数据集组合 × {len(models)} 个模型")
    print("-" * 66)

    total_start_time = time.time()
    results = []
    total_tasks = len(dataset_pairs) * len(models)
    current_task = 0
    completed, failed = 0, 0

    for pair_index, (train_name, test_name) in enumerate(dataset_pairs, 1):
        print(f"\n{'#' * 66}")
        print(f"数据集组合进度: {pair_index}/{len(dataset_pairs)} —— "
              f"训练集: {train_name}，测试集: {test_name}")
        print(f"{'#' * 66}")

        for model_config in models:
            current_task += 1
            print(f"\n[总进度 {current_task}/{total_tasks}]")
            success = run_model(model_config, train_name, test_name, output_root)
            results.append((model_config["name"], train_name, test_name, success))
            if success:
                completed += 1
            else:
                failed += 1

    total_elapsed = time.time() - total_start_time

    print(f"\n{'=' * 66}")
    print("所有模型运行完成")
    print(f"总任务数: {total_tasks}，完成: {completed} 个，失败: {failed} 个")
    print(f"总耗时: {total_elapsed:.2f} 秒")
    print(f"{'=' * 66}")

    # ===== 运行结果明细 =====
    failed_items = [r for r in results if not r[3]]
    if not failed_items:
        print("\n运行结果: 全部成功")
    else:
        print(f"\n运行结果: 失败 {len(failed_items)} 项（详见各自的 output.log）")
        for model_name, train_name, test_name, _ in failed_items:
            print(f"  [失败] {model_name} [{train_name} -> {test_name}]")

    # ===== 步骤3：汇总结果 =====
    if args.skip_summary:
        print("\n已跳过汇总。可稍后执行: python summarize_results.py")
        return

    print("\n" + "-" * 66)
    print("步骤3/3：汇总所有模型结果")
    print("-" * 66)

    from summarize_results import summarize
    summarize(output_root=output_root, models=models, dataset_pairs=dataset_pairs)


if __name__ == "__main__":
    main()
