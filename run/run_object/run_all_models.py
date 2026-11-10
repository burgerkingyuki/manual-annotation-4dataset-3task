import os
import subprocess
import time
import shutil
import sys
from config import (
    MODELS, 
    DATASET_METHOD, 
    TRAIN_DATASETS, 
    TEST_DATASETS,
    # TRAIN_DATASET_PATH,
    # TEST_DATASET_PATH,
    # DATASET_PATH
)

# 添加当前目录到PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def get_dataset_config(train_name, test_name):
    """
    根据数据集方法获取配置
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


def run_model(model_config, train_name, test_name, base_dir):
    """
    运行单个模型
    """
    model_name = model_config["name"]
    script_path = model_config["script_path"]
    
    # 构建输出目录名
    if DATASET_METHOD == 1:
        dataset_name = list(TRAIN_DATASETS.keys())[0]
        output_dir = os.path.join(base_dir, "引文对象", dataset_name, model_name)
    elif DATASET_METHOD == 2:
        train_dataset_name = list(TRAIN_DATASETS.keys())[0]
        test_dataset_name = list(TEST_DATASETS.keys())[0]
        output_dir = os.path.join(base_dir, "引文对象", f"{train_dataset_name}-{test_dataset_name}", model_name)
    else:  # 方法3
        output_dir = os.path.join(base_dir, "引文对象", f"{train_name}-{test_name}", model_name)

    print(f"\n{'='*50}")
    print(f"开始运行模型: {model_name}")
    print(f"训练集: {train_name}")
    print(f"测试集: {test_name}")
    print(f"{'='*50}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存当前工作目录
    original_cwd = os.getcwd()

    # 切换到模型脚本所在目录
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

    # 运行模型
    start_time = time.time()
    try:
        # 设置环境变量，添加当前目录到PYTHONPATH
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{current_dir};{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = current_dir
        
        # 设置数据集配置的环境变量
        dataset_config = get_dataset_config(train_name, test_name)
        for key, value in dataset_config.items():
            env[key] = str(value)
        
        result = subprocess.run(
            ["python", os.path.basename(script_path)],
            capture_output=True,
            text=True,
            timeout=36000,  # 10小时超时
            env=env
        )

        # 保存标准输出和错误输出
        with open(os.path.join(output_dir, "output.log"), "w", encoding="utf-8") as f:
            f.write("标准输出:\n")
            f.write(result.stdout)
            f.write("\n\n错误输出:\n")
            f.write(result.stderr)

        # 复制生成的CSV文件到输出目录
        csv_files = ["predictions_vs_true.csv", "span_results.csv"]
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                shutil.copy(csv_file, os.path.join(output_dir, csv_file))
                os.remove(csv_file)  # 删除原文件

        end_time = time.time()
        print(f"模型 {model_name} 运行完成，耗时: {end_time - start_time:.2f} 秒")

        return True
    except subprocess.TimeoutExpired:
        print(f"模型 {model_name} 运行超时")
        return False
    except Exception as e:
        print(f"模型 {model_name} 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 切换回原工作目录
        os.chdir(original_cwd)


def main():
    """
    主函数
    """
    print("开始一键运行所有模型...")
    
    # 记录开始时间
    total_start_time = time.time()
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 生成训练集和测试集组合
    if DATASET_METHOD == 3:
        # 方法3：N*N批量训练
        train_names = list(TRAIN_DATASETS.keys())
        test_names = list(TEST_DATASETS.keys())
        dataset_pairs = [(train, test) for train in train_names for test in test_names]
    else:
        # 方法1和方法2：单组训练
        train_name = list(TRAIN_DATASETS.keys())[0]
        test_name = list(TEST_DATASETS.keys())[0]
        dataset_pairs = [(train_name, test_name)]
    
    print(f"数据集组合数量: {len(dataset_pairs)}")
    print(f"模型数量: {len(MODELS)}")
    
    # 运行所有模型和数据集组合
    results = []
    total_tasks = len(dataset_pairs) * len(MODELS)
    current_task = 0
    
    for train_name, test_name in dataset_pairs:
        current_task += 1
        print(f"\n{'='*60}")
        print(f"进度: {current_task}/{len(dataset_pairs)} - 训练集: {train_name}, 测试集: {test_name}")
        print(f"{'='*60}")
        
        for model_config in MODELS:
            success = run_model(model_config, train_name, test_name, base_dir)
            results.append((model_config["name"], train_name, test_name, success))
    
    # 输出运行结果
    print(f"\n{'='*50}")
    print("所有模型运行完成")
    print(f"{'='*50}")
    
    total_end_time = time.time()
    print(f"总耗时: {total_end_time - total_start_time:.2f} 秒")
    
    # 显示每个模型的运行结果
    print("\n运行结果:")
    for model_name, train_name, test_name, success in results:
        status = "成功" if success else "失败"
        print(f"{model_name} [{train_name} -> {test_name}]: {status}")


if __name__ == "__main__":
    main()
