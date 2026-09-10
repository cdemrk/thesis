import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ΟΡΙΣΜΟΣ ΣΕΝΑΡΙΩΝ ΚΑΙ ΡΥΘΜΙΣΕΩΝ ΜΠΑΡΑΣ
# ==========================================
scenarios = ['Abundance', 'Marginal', 'Mild\nContention', 'Heavy\nContention']
x = np.arange(len(scenarios))

# ΑΛΛΑΓΗ: Αντιστροφή της σειράς των K (από μικρό σε μεγάλο)
configs = ['1', '2', '4', '6', '8', '10', '20']
num_configs = len(configs)
total_width = 0.8
width = total_width / num_configs

# ΑΛΛΑΓΗ: Αντιστροφή των χρωμάτων για να διατηρηθεί η αντιστοίχιση χρώματος-K
colors = ['#C0392B', '#E67E22', '#F4D03F', '#A9DFBF', '#5499C7', '#2980B9', '#1A5276']

# ==========================================
# 2. ΕΙΣΑΓΩΓΗ ΔΕΔΟΜΕΝΩΝ (ΑΠΟ ΠΡΑΓΜΑΤΙΚΟ ΠΕΙΡΑΜΑ PRUNING)
# ==========================================
# (a) Service Rate (%)
data_sr = {
    '20': [100.0, 100.0, 83.6, 42.2],
    '10': [100.0, 100.0, 83.6, 39.3],
    '8':  [100.0, 100.0, 85.2, 39.3],
    '6':  [100.0, 100.0, 85.2, 41.5],
    '4':  [100.0, 100.0, 82.0, 43.7],
    '2':  [100.0, 100.0, 86.9, 40.0],
    '1':  [100.0, 100.0, 85.2, 40.0]
}

# (b) Average Execution Cost per Task ($)
data_cost = {
    '20': [0.00202, 0.05408, 0.06171, 0.04576],
    '10': [0.00201, 0.05967, 0.05984, 0.04919],
    '8':  [0.00201, 0.06087, 0.05907, 0.04932],
    '6':  [0.00203, 0.05890, 0.06112, 0.05435],
    '4':  [0.00203, 0.06426, 0.05913, 0.04903],
    '2':  [0.00203, 0.05100, 0.05698, 0.05351],
    '1':  [0.00200, 0.05996, 0.05697, 0.05513]
}

# (c) Average Idle Time per Task (ms)
data_idle = {
    '20': [107043, 621573, 502134, 410013],
    '10': [107022, 373277, 421893, 298175],
    '8':  [106954, 340023, 211673, 309665],
    '6':  [106938, 553204, 214089, 523939],
    '4':  [106935, 245786, 448189, 135024],
    '2':  [106921, 247496, 230840,  97875],
    '1':  [106886, 259989, 348010,  96295]
}

# (d) Average Latency (ms)
data_lat = {
    '20': [108973, 627679, 509455, 416757],
    '10': [108912, 379186, 429425, 305320],
    '8':  [108822, 345432, 219888, 316394],
    '6':  [108830, 558707, 221736, 533148],
    '4':  [108826, 251459, 455524, 144045],
    '2':  [108807, 253124, 238762, 105633],
    '1':  [108767, 265529, 356126, 104340]
}

# (e) Peak GPU Utilization (%)
data_gpu = {
    '20': [50.0, 95.4, 99.4, 100.0],
    '10': [50.0, 95.4, 99.4,  99.4],
    '8':  [50.0, 95.4, 98.9,  99.4],
    '6':  [50.0, 95.4, 98.9,  99.4],
    '4':  [50.0, 95.4, 98.3,  99.4],
    '2':  [50.0, 95.4, 100.0, 100.0],
    '1':  [50.0, 95.4, 98.9, 100.0]
}

# (f) Cumulative System Reward
data_reward = {
    '20': [458.3, 649.4,  136.4, -3172.8],
    '10': [458.4, 652.4,  139.0, -3432.9],
    '8':  [458.4, 653.2,  211.6, -3430.5],
    '6':  [458.4, 651.6,  208.6, -3240.2],
    '4':  [458.4, 653.8,   79.9, -3034.6],
    '2':  [458.4, 657.2,  278.3, -3358.1],
    '1':  [458.5, 657.5,  214.1, -3356.3]
}

# Decision Time (Για το 2ο ξεχωριστό γράφημα)
data_decision_time = {
    '20': [16.02, 13.31, 11.00, 5.16],
    '10': [4.75,  4.63,  3.86,  1.80],
    '8':  [3.56,  3.47,  2.98,  1.36],
    '6':  [2.42,  2.45,  2.05,  0.95],
    '4':  [1.52,  1.54,  1.25,  0.63],
    '2':  [0.80,  0.81,  0.69,  0.30],
    '1':  [0.48,  0.51,  0.41,  0.18]
}

# Κοινή συνάρτηση σχεδίασης
def plot_metric(ax, data_dict, title, ylabel, is_percentage=False):
    for i, k_val in enumerate(configs):
        lbl = f'$K = {k_val}$'
        offset = (i - num_configs/2 + 0.5) * width
        ax.bar(x + offset, data_dict[k_val], width, label=lbl, 
               color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    if is_percentage:
        ax.set_ylim([0, 115])

# ==========================================
# 3. ΔΗΜΙΟΥΡΓΙΑ ΕΙΚΟΝΑΣ 1: 6-GRID ΜΕ PERFORMANCE METRICS
# ==========================================
fig, axs = plt.subplots(3, 2, figsize=(16, 16), dpi=300)

plot_metric(axs[0, 0], data_sr, '(a) Service Rate', 'Successful Tasks (%)', is_percentage=True)
plot_metric(axs[0, 1], data_cost, '(b) Average Execution Cost per Task', 'Cost per Task ($)')
plot_metric(axs[1, 0], data_idle, '(c) Average Idle Time per Task', 'Idle Time (ms)')
plot_metric(axs[1, 1], data_lat, '(d) Average End-to-End Latency per Task', 'Latency (ms)')
plot_metric(axs[2, 0], data_gpu, '(e) Peak GPU Utilization', 'Utilization (%)', is_percentage=True)
plot_metric(axs[2, 1], data_reward, '(f) Cumulative System Reward', 'Total Reward')

handles, labels = axs[2, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=12, 
           bbox_to_anchor=(0.5, 0.0), frameon=True, shadow=True, 
           title="Action Space Bounding ($Top-K$ Profiles)", title_fontsize=13)

plt.subplots_adjust(bottom=0.12, top=0.98, hspace=0.35, wspace=0.22)
fig.savefig('drl_pruning_performance.pdf', format='pdf', bbox_inches='tight')
fig.savefig('drl_pruning_performance.png', format='png', bbox_inches='tight')
plt.close(fig)

# ==========================================
# 4. ΔΗΜΙΟΥΡΓΙΑ ΕΙΚΟΝΑΣ 2: ΞΕΧΩΡΙΣΤΟ PLOT ΓΙΑ ΤΟ DECISION TIME
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

plot_metric(ax2, data_decision_time, 'DRL Agent Inference Time per Task', 'Inference Time (ms)')

# Προσθήκη υπομνήματος ειδικά για αυτό το γράφημα
handles2, labels2 = ax2.get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='lower center', ncol=7, fontsize=11, 
            bbox_to_anchor=(0.5, -0.05), frameon=True, shadow=True, 
            title="Action Space Bounding ($Top-K$ Profiles)", title_fontsize=12)

plt.subplots_adjust(bottom=0.2)
fig2.savefig('drl_pruning_decision_time.pdf', format='pdf', bbox_inches='tight')
fig2.savefig('drl_pruning_decision_time.png', format='png', bbox_inches='tight')
plt.close(fig2)

print("✅ Όλα τα γραφήματα Pruning δημιουργήθηκαν επιτυχώς (από K=1 έως K=20)!")
print(" -> Αποθηκεύτηκαν: 'drl_pruning_performance' (6-grid) & 'drl_pruning_decision_time' (μονό)")