import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Set paths
BASE_DIR = r"d:\通过拉曼光谱预测物及其浓度"
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures", "midterm_defense")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set font for matplotlib (support Chinese and look professional)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_model_comparison():
    print("Generating Model Comparison Bar Chart (Chinese)...")
    tasks = ['Task1\n(混合复杂度)', 'Task2\n(福美双存在性)', 'Task3\n(MG存在性)', 
             'Task4\n(福美双浓度等级)', 'Task5\n(MG浓度等级)', 'Task6\n(MBA浓度等级)']
    
    # Macro-F1 scores from baseline_report_revised.md
    rf = [0.784, 0.879, 0.639, 0.589, 0.403, 0.637]
    ridge = [0.750, 0.893, 0.670, 0.570, 0.391, 0.586]
    plsda = [0.683, 0.810, 0.583, 0.478, 0.398, 0.513]
    cnn = [0.579, 0.798, 0.519, 0.496, 0.395, 0.478]
    
    x = np.arange(len(tasks))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - 1.5*width, rf, width, label='随机森林 (RF)', color='#2F4F4F')
    rects2 = ax.bar(x - 0.5*width, ridge, width, label='岭回归分类 (Ridge)', color='#4682B4')
    rects3 = ax.bar(x + 0.5*width, plsda, width, label='偏最小二乘-判别分析 (PLS-DA)', color='#CD853F')
    rects4 = ax.bar(x + 1.5*width, cnn, width, label='一维卷积神经网络 (1D-CNN)', color='#B22222')
    
    ax.set_ylabel('Macro-F1 得分', fontsize=12)
    ax.set_title('各模型在不同分类任务上的交叉验证表现对比 (3折 SGKF)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), ncol=4)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    # 如果需要可取消注释标注数字
    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)
    # autolabel(rects4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, '1_model_comparison_bar.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_feature_importance():
    print("Generating Feature Importance Spectra Plot (Chinese)...")
    # Load data
    try:
        wavenumber = np.load(os.path.join(PROCESSED_DIR, "wavenumber.npy"))
        X = np.load(os.path.join(PROCESSED_DIR, "X_p1.npy"))
        meta = pd.read_csv(os.path.join(DATA_DIR, "metadata_v1.csv"))
    except Exception as e:
        print(f"Error loading data for feature importance: {e}")
        return
        
    y = meta['has_thiram'].astype(int).values
    
    # Train a quick RF
    print("Training Random Forest on entire dataset to extract temporal feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    # Smooth importances slightly for better visualization
    window = 5
    importances_smooth = np.convolve(importances, np.ones(window)/window, mode='same')
    
    mean_spectrum_pos = np.mean(X[y == 1], axis=0)
    mean_spectrum_neg = np.mean(X[y == 0], axis=0)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot spectra on left axis
    ax1.plot(wavenumber, mean_spectrum_pos, color='#B22222', alpha=0.8, label='含有福美双 (Mean Spectrum)')
    ax1.plot(wavenumber, mean_spectrum_neg, color='#4682B4', alpha=0.8, label='未含福美双 (Mean Spectrum)')
    ax1.set_xlabel('拉曼频移 (cm$^{-1}$)', fontsize=12)
    ax1.set_ylabel('相对强度 (标准化后)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot feature importance on right axis
    ax2 = ax1.twinx()
    # Normalize importance for better visual scale
    imp_norm = importances_smooth / np.max(importances_smooth)
    ax2.fill_between(wavenumber, 0, imp_norm, color='orange', alpha=0.3, label='随机森林特征重要性得分')
    ax2.plot(wavenumber, imp_norm, color='darkorange', linewidth=1.5)
    ax2.set_ylabel('归一化特征重要性', color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')
    
    plt.title('拉曼光谱特征峰捕捉：特征重要性得分投影热力映射 (任务: 判断福美双存在性)', fontsize=14)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, '2_feature_importance_spectra.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_folder_distribution():
    print("Generating Folder Distribution Chart (Chinese)...")
    try:
        meta = pd.read_csv(os.path.join(DATA_DIR, "metadata_v1.csv"))
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
        
    # Group by folder_name
    folder_meta = meta.groupby('folder_name').agg({
        'c_thiram': 'first',
        'c_mg': 'first',
        'mixture_order': 'first'
    }).reset_index()
    
    thiram_counts = folder_meta['c_thiram'].value_counts().sort_index()
    mg_counts = folder_meta['c_mg'].value_counts().sort_index()
    
    labels = [f"0 ppm", f"4 ppm", f"5 ppm", f"6 ppm"]
    
    # For missing categories add 0
    t_vals = [thiram_counts.get(k, 0) for k in [0, 4, 5, 6]]
    m_vals = [mg_counts.get(k, 0) for k in [0, 4, 5, 6]]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Thiram
    bars1 = axes[0].bar(x, t_vals, width, color=['#A9A9A9', '#FFE4B5', '#FFA07A', '#CD5C5C'])
    axes[0].set_title('以独立 Folder 计：福美双 (Thiram) 样本分布', fontsize=13)
    axes[0].set_ylabel('独立 Folder 数量 (即完全独立样本数)', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, max(max(t_vals), max(m_vals)) + 2)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')
        
    # Highlight the 6ppm structural deficiency
    if t_vals[-1] <= 3:
        axes[0].annotate('关键数据极度匮乏\n(仅含3组实验批次)', xy=(x[-1], t_vals[-1]), xytext=(x[-1]-1.5, t_vals[-1]+10),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                     fontsize=10, color='red', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    # MG
    bars2 = axes[1].bar(x, m_vals, width, color=['#A9A9A9', '#ADD8E6', '#87CEFA', '#4682B4'])
    axes[1].set_title('以独立 Folder 计：孔雀石绿 (MG) 样本分布', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, max(max(t_vals), max(m_vals)) + 2)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')

    plt.suptitle('数据底层匮乏结构剖析：高浓度独立训练样本稀缺导致“受限结果”', fontsize=15, y=1.05)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, '3_folder_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_model_comparison()
    plot_feature_importance()
    plot_folder_distribution()
    print("\nAll defense visualizations generated successfully in figures/midterm_defense/")
