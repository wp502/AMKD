import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

# =======================
# 数据
# =======================

# 1️⃣ Temperature
T = np.array([0.5,1,2,3,4,5,6])
micro_T = np.array([65.34,66.02,66.44,66.21,67.48,66.41,66.25])
macro_T = np.array([58.77,60.30,60.22,60.98,62.57,60.93,60.98])

# 2️⃣ Weight
W = np.array([0.01,0.05,0.1,0.5,1,2,5])
micro_W = np.array([55.00,61.23,63.51,65.71,67.48,66.67,66.85])
macro_W = np.array([45.40,54.57,57.61,60.27,62.57,61.75,61.81])

# 3️⃣ Dimension
D = np.array([64,128,256,512,1024,2048])
micro_D = np.array([64.76,65.37,64.71,67.48,66.72,66.68])
macro_D = np.array([59.27,59.95,59.43,62.57,60.55,62.25])

# =======================
# 绘图
# =======================

fig, axes = plt.subplots(1, 3, figsize=(15,4))

def plot_subplot(ax, x, y1, y2, xlabel):
    ax.plot(x, y1, marker='o', linewidth=2, label='Micro-F1')
    ax.plot(x, y2, marker='s', linewidth=2, label='Macro-F1')
    
    # 标注最优点
    best_idx = np.argmax(y1)
    ax.scatter(x[best_idx], y1[best_idx], s=80)
    
    best_idx2 = np.argmax(y2)
    ax.scatter(x[best_idx2], y2[best_idx2], s=80)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylim(45, 70)
    ax.grid(True, linestyle='--', alpha=0.4)

# 子图
plot_subplot(axes[0], T, micro_T, macro_T, 'T')
plot_subplot(axes[1], W, micro_W, macro_W, 'Weight')
plot_subplot(axes[2], D, micro_D, macro_D, 'Dimension')

axes[0].set_ylabel('F1 Score', fontsize=12)

# 统一图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11)

plt.tight_layout(rect=[0,0.1,1,1])

# 保存高质量PDF
plt.savefig("MMIMDB_parameter_sensitivity.pdf", dpi=300, bbox_inches='tight')

plt.show()