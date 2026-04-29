import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_clean_data(file_path, method_type, view_type):
    """加载并清洗数据"""
    try:
        df = pd.read_csv(file_path)
        
        # 去除重复行（基于filename）
        df = df.drop_duplicates(subset=['filename'], keep='first')
        
        # 添加方法类型和视图类型列
        df['method'] = method_type
        df['view'] = view_type
        
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def create_single_boxplot(df, title, save_path):
    """为单个数据集创建包含Dice和HD95的箱线图"""
    if df is None:
        print(f"无法为 {title} 创建箱线图：数据为空")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Dice系数箱线图
    dice_data = df['dice']
    box1 = ax1.boxplot(dice_data, patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    ax1.set_title(f'{title} - Dice Coefficient', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Dice Coefficient', fontsize=12)
    ax1.set_xticklabels(['Dice'], fontsize=12)
    
    # 添加Dice统计信息
    dice_mean = dice_data.mean()
    dice_std = dice_data.std()
    ax1.text(0.5, 0.95, f'Mean: {dice_mean:.3f}\nStd: {dice_std:.3f}', 
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # HD95箱线图
    hd95_data = df['hd95']
    box2 = ax2.boxplot(hd95_data, patch_artist=True, 
                      boxprops=dict(facecolor='lightgreen', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    ax2.set_title(f'{title} - HD95 Distance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('HD95 Distance', fontsize=12)
    ax2.set_xticklabels(['HD95'], fontsize=12)
    
    # 添加HD95统计信息
    hd95_mean = hd95_data.mean()
    hd95_std = hd95_data.std()
    ax2.text(0.5, 0.95, f'Mean: {hd95_mean:.3f}\nStd: {hd95_std:.3f}', 
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"箱线图已保存到: {save_path}")

def calculate_summary_statistics(data_list, method_names):
    """计算汇总统计信息"""
    summary_data = []
    
    for i, df in enumerate(data_list):
        if df is not None:
            method_name = method_names[i]
            dice_mean = df['dice'].mean()
            dice_std = df['dice'].std()
            hd95_mean = df['hd95'].mean()
            hd95_std = df['hd95'].std()
            
            summary_data.append({
                'method': method_name,
                'dice_mean': round(dice_mean,2),
                'dice_std': round(dice_std,2),
                'hd95_mean': round(hd95_mean,2),
                'hd95_std': round(hd95_std,2),
                'sample_size': len(df)
            })
    
    return pd.DataFrame(summary_data)

def main():
    # 文件路径配置
    files_config = [
        {
            'path': 'results/diagnosis/longitudinal/evaluation_results.csv',
            'method': 'traditional',
            'view': 'longitudinal',
            'name': 'Longitudinal Traditional',
            'save_name': 'longitudinal_traditional'
        },
        {
            'path': 'results/diagnosis/longitudinal/dl_evaluation_results.csv',
            'method': 'deep_learning',
            'view': 'longitudinal',
            'name': 'Longitudinal Deep Learning',
            'save_name': 'longitudinal_deep_learning'
        },
        {
            'path': 'results/diagnosis/transverse/evaluation_results.csv',
            'method': 'traditional',
            'view': 'transverse',
            'name': 'Transverse Traditional',
            'save_name': 'transverse_traditional'
        },
        {
            'path': 'results/diagnosis/transverse/dl_evaluation_results.csv',
            'method': 'deep_learning',
            'view': 'transverse',
            'name': 'Transverse Deep Learning',
            'save_name': 'transverse_deep_learning'
        }
    ]
    
    # 加载所有数据
    all_data = []
    method_names = []
    
    for config in files_config:
        print(f"加载文件: {config['path']}")
        df = load_and_clean_data(config['path'], config['method'], config['view'])
        if df is not None:
            all_data.append(df)
            method_names.append(config['name'])
            print(f"  - 成功加载 {len(df)} 条数据")
        else:
            all_data.append(None)
            method_names.append(config['name'])
            print(f"  - 加载失败")
    
    # 创建输出目录
    os.makedirs('results/diagnosis', exist_ok=True)
    
    # 为每个数据集创建单独的箱线图
    for i, config in enumerate(files_config):
        if all_data[i] is not None:
            save_path = f"results/diagnosis/{config['save_name']}_boxplot.png"
            create_single_boxplot(all_data[i], config['name'], save_path)
    
    # 计算汇总统计信息
    summary_df = calculate_summary_statistics(all_data, method_names)
    
    # 保存汇总结果
    summary_df.to_csv('results/diagnosis/sum.csv', index=False)
    print(f"汇总结果已保存到: results/diagnosis/sum.csv")
    
    # 打印汇总结果
    print("\n=== 汇总统计结果 ===")
    print(summary_df.to_string(index=False))
    
    # 额外分析：深度学习方法 vs 传统方法
    print("\n=== 深度学习方法 vs 传统方法 ===")
    dl_data = [df for df, name in zip(all_data, method_names) if 'Deep Learning' in name]
    traditional_data = [df for df, name in zip(all_data, method_names) if 'Traditional' in name]
    
    if dl_data and traditional_data:
        dl_combined = pd.concat(dl_data, ignore_index=True)
        traditional_combined = pd.concat(traditional_data, ignore_index=True)
        
        print(f"深度学习方法平均 Dice: {dl_combined['dice'].mean():.4f} ± {dl_combined['dice'].std():.4f}")
        print(f"传统方法平均 Dice: {traditional_combined['dice'].mean():.4f} ± {traditional_combined['dice'].std():.4f}")
        print(f"深度学习方法平均 HD95: {dl_combined['hd95'].mean():.4f} ± {dl_combined['hd95'].std():.4f}")
        print(f"传统方法平均 HD95: {traditional_combined['hd95'].mean():.4f} ± {traditional_combined['hd95'].std():.4f}")

if __name__ == "__main__":
    main()