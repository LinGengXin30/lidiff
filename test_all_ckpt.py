import os
import subprocess
import re
import json
from datetime import datetime

# 配置参数
EXPERIMENT_ID = "prob10_5p0reg"
CKPT_DIR = f"lidiff/experiments/{EXPERIMENT_ID}/checkpoints"
CONFIG_FILE = "lidiff/config/config.yaml"
RESULT_FILE = f"test_results_{EXPERIMENT_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# 确保结果目录存在
os.makedirs(os.path.dirname(RESULT_FILE) or '.', exist_ok=True)

# 收集所有ckpt文件
ckpt_files = []
for file in os.listdir(CKPT_DIR):
    if file.endswith('.ckpt') and 'epoch=' in file:
        ckpt_files.append(os.path.join(CKPT_DIR, file))

# 按epoch排序
ckpt_files.sort(key=lambda x: int(re.search(r'epoch=(\d+)', x).group(1)))

print(f"找到 {len(ckpt_files)} 个ckpt文件，开始测试...")

# 测试结果
results = []

# 遍历测试每个ckpt文件
for ckpt_file in ckpt_files:
    # 提取epoch编号
    epoch_match = re.search(r'epoch=(\d+)', ckpt_file)
    if not epoch_match:
        continue
    epoch = int(epoch_match.group(1))
    
    print(f"\n测试 Epoch {epoch}...")
    print(f"文件: {ckpt_file}")
    
    # 构建测试命令
    # 检查hparams.yaml文件是否存在
    hparams_path = os.path.join(os.path.dirname(ckpt_file), '..', 'lightning_logs', 'version_11', 'hparams.yaml')
    if not os.path.exists(hparams_path):
        # 尝试其他version目录
        lightning_logs_dir = os.path.join(os.path.dirname(ckpt_file), '..', 'lightning_logs')
        if os.path.exists(lightning_logs_dir):
            versions = [d for d in os.listdir(lightning_logs_dir) if d.startswith('version_')]
            if versions:
                # 使用最新的version目录
                versions.sort(key=lambda x: int(x.split('_')[1]))
                hparams_path = os.path.join(lightning_logs_dir, versions[-1], 'hparams.yaml')
    
    # 构建命令
    cmd = [
        'python', 'lidiff/train.py',
        '--config', CONFIG_FILE,
        '--weights', ckpt_file,
        '--test'
    ]
    
    # 执行命令并捕获输出
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, cwd='.')
        
        # 解析输出中的指标
        cd_mean_match = re.search(r'CD Mean: ([\d.]+)', output)
        cd_std_match = re.search(r'CD Std: ([\d.]+)', output)
        precision_match = re.search(r'Precision: ([\d.]+)', output)
        recall_match = re.search(r'Recall: ([\d.]+)', output)
        fscore_match = re.search(r'F-Score: ([\d.]+)', output)
        
        # 提取指标值
        cd_mean = float(cd_mean_match.group(1)) if cd_mean_match else None
        cd_std = float(cd_std_match.group(1)) if cd_std_match else None
        precision = float(precision_match.group(1)) if precision_match else None
        recall = float(recall_match.group(1)) if recall_match else None
        fscore = float(fscore_match.group(1)) if fscore_match else None
        
        # 保存结果
        result = {
            'epoch': epoch,
            'ckpt_file': ckpt_file,
            'cd_mean': cd_mean,
            'cd_std': cd_std,
            'precision': precision,
            'recall': recall,
            'fscore': fscore
        }
        results.append(result)
        
        # 打印结果
        print(f"Epoch {epoch} 测试完成:")
        print(f"  CD Mean: {cd_mean}")
        print(f"  CD Std: {cd_std}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
        print(f"  F-Score: {fscore}")
        
    except subprocess.CalledProcessError as e:
        print(f"测试 Epoch {epoch} 失败:")
        print(e.output)
        
        # 保存失败结果
        result = {
            'epoch': epoch,
            'ckpt_file': ckpt_file,
            'error': str(e),
            'output': e.output[:500]  # 只保存部分输出
        }
        results.append(result)

# 保存所有结果到JSON文件
with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n所有测试完成，结果保存到: {RESULT_FILE}")

# 分析最佳模型
if results:
    # 过滤掉失败的结果
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        # 按CD Mean排序（越小越好）
        best_cd = min(valid_results, key=lambda x: x['cd_mean'])
        # 按F-Score排序（越大越好）
        best_fscore = max(valid_results, key=lambda x: x['fscore'])
        
        print("\n最佳模型分析:")
        print(f"按 CD Mean 最佳: Epoch {best_cd['epoch']}")
        print(f"  CD Mean: {best_cd['cd_mean']}")
        print(f"  F-Score: {best_cd['fscore']}")
        print(f"  文件: {best_cd['ckpt_file']}")
        
        print(f"\n按 F-Score 最佳: Epoch {best_fscore['epoch']}")
        print(f"  F-Score: {best_fscore['fscore']}")
        print(f"  CD Mean: {best_fscore['cd_mean']}")
        print(f"  文件: {best_fscore['ckpt_file']}")
        
        # 打印所有结果表格
        print("\n所有测试结果:")
        print("Epoch | CD Mean  | F-Score  | Precision | Recall")
        print("-" * 60)
        for r in sorted(valid_results, key=lambda x: x['epoch']):
            print(f"{r['epoch']:5d} | {r['cd_mean']:.4f} | {r['fscore']:.4f} | {r['precision']:.4f} | {r['recall']:.4f}")
    else:
        print("\n没有有效的测试结果。")
else:
    print("\n没有测试结果。")