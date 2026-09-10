import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ΟΡΙΣΜΟΣ ΣΕΝΑΡΙΩΝ ΚΑΙ ΡΥΘΜΙΣΕΩΝ ΜΠΑΡΑΣ
# ==========================================
scenarios = ['Abundance', 'Marginal', 'Mild\nContention', 'Heavy\nContention']
x = np.arange(len(scenarios))

configs = ['0.0/1.0', '0.2/0.8', '0.4/0.6', '0.5/0.5', '0.6/0.4', '0.8/0.2', '1.0/0.0']
num_configs = len(configs)
total_width = 0.8
width = total_width / num_configs

colors = ['#4A235A', '#2471A3', '#138D75', '#229954', '#D4AC0D', '#E67E22', '#A93226']

# ==========================================
# 2. ΕΙΣΑΓΩΓΗ ΠΡΑΓΜΑΤΙΚΩΝ ΔΕΔΟΜΕΝΩΝ (Από το terminal output)
# Κάθε λίστα αντιστοιχεί στα 4 σενάρια.
# ==========================================
data_sr = {
    '0.0/1.0': [100.0, 100.0, 86.9, 43.7],
    '0.2/0.8': [100.0, 100.0, 83.6, 43.7],
    '0.4/0.6': [100.0, 100.0, 86.9, 43.0],
    '0.5/0.5': [100.0, 100.0, 86.9, 40.0],
    '0.6/0.4': [100.0, 100.0, 86.9, 43.0],
    '0.8/0.2': [100.0, 100.0, 85.2, 40.7],
    '1.0/0.0': [100.0, 100.0, 83.6, 40.0]
}

data_cost = {
    '0.0/1.0': [0.01407, 0.06799, 0.04434, 0.03935],
    '0.2/0.8': [0.00225, 0.07324, 0.07290, 0.05253],
    '0.4/0.6': [0.00230, 0.07300, 0.05862, 0.04940],
    '0.5/0.5': [0.00203, 0.06898, 0.05843, 0.05248],
    '0.6/0.4': [0.00199, 0.05250, 0.06086, 0.05064],
    '0.8/0.2': [0.00189, 0.06921, 0.05841, 0.04823],
    '1.0/0.0': [0.00170, 0.04120, 0.05111, 0.06276]
}

data_idle = {
    '0.0/1.0': [106722, 102723, 164148,  88506],
    '0.2/0.8': [106572, 101855, 396983,  89665],
    '0.4/0.6': [106625, 174562, 398717, 117563],
    '0.5/0.5': [106996, 464931, 230874, 290498],
    '0.6/0.4': [107003, 619271, 493210, 356348],
    '0.8/0.2': [107782, 539107, 647831, 483199],
    '1.0/0.0': [494809, 1078936, 1433363, 1154124]
}

data_lat = {
    '0.0/1.0': [111073, 109775, 170579,  94798],
    '0.2/0.8': [108490, 107676, 405418,  98678],
    '0.4/0.6': [108533, 180627, 406763, 125612],
    '0.5/0.5': [108906, 470661, 238597, 298596],
    '0.6/0.4': [108892, 625119, 501409, 365553],
    '0.8/0.2': [109684, 545196, 655708, 489754],
    '1.0/0.0': [496702, 1084581, 1439819, 1162050]
}

data_gpu = {
    '0.0/1.0': [50.0, 95.4,  98.9, 100.0],
    '0.2/0.8': [50.0, 95.4, 100.0,  99.4],
    '0.4/0.6': [50.0, 95.4, 100.0,  99.4],
    '0.5/0.5': [50.0, 95.4, 100.0, 100.0],
    '0.6/0.4': [50.0, 95.4, 100.0,  98.9],
    '0.8/0.2': [50.0, 95.4,  98.9, 100.0],
    '1.0/0.0': [50.0, 95.4,  99.4, 100.0]
}

data_reward = {
    '0.0/1.0': [426.7, 635.1, 262.9, -3035.2],
    '0.2/0.8': [439.4, 643.0, 122.8, -3038.5],
    '0.4/0.6': [452.0, 652.5, 272.8, -3103.7],
    '0.5/0.5': [458.4, 652.1, 271.2, -3366.6],
    '0.6/0.4': [464.8, 659.4, 277.3, -3101.1],
    '0.8/0.2': [477.7, 671.3, 224.2, -3281.5],
    '1.0/0.0': [491.3, 692.3, 183.4, -3343.5]
}

# ==========================================
# 3. ΡΥΘΜΙΣΕΙΣ ΚΑΙ ΣΧΕΔΙΑΣΗ ΠΛΕΓΜΑΤΟΣ
# ==========================================
fig, axs = plt.subplots(3, 2, figsize=(16, 16), dpi=300)

def plot_metric(ax, data_dict, title, ylabel, is_percentage=False, is_gpu=False):
    for i, cfg in enumerate(configs):
        # Σπάμε το string '0.4/0.6' στα δύο επιμέρους βάρη
        w_c, w_i = cfg.split('/')
        
        # Χρήση LaTeX Math mode με \text{} για δείκτες COST και IDLE
        lbl = f'$w_{{\\text{{COST}}}}={w_c}$, $w_{{\\text{{IDLE}}}}={w_i}$'
        
        offset = (i - num_configs/2 + 0.5) * width
        ax.bar(x + offset, data_dict[cfg], width, label=lbl, 
               color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    if is_percentage:
        ax.set_ylim([0, 115])

# Σχεδίαση των 6 υπο-διαγραμμάτων
plot_metric(axs[0, 0], data_sr, '(a) Service Rate', 'Successful Tasks (%)', is_percentage=True)
plot_metric(axs[0, 1], data_cost, '(b) Average Execution Cost per Task', 'Cost per Task ($)')
plot_metric(axs[1, 0], data_idle, '(c) Average Idle Time per Task', 'Idle Time (ms)')
plot_metric(axs[1, 1], data_lat, '(d) Average End-to-End Latency per Task', 'Latency (ms)')
plot_metric(axs[2, 0], data_gpu, '(e) Peak GPU Utilization', 'Utilization (%)', is_percentage=True, is_gpu=True)
plot_metric(axs[2, 1], data_reward, '(f) Cumulative System Reward', 'Total Reward')

# ==========================================
# 4. ΕΝΙΑΙΟ ΥΠΟΜΝΗΜΑ (LEGEND) ΚΑΙ ΑΠΟΘΗΚΕΥΣΗ
# ==========================================
# Παίρνουμε τα handles από το γράφημα (e) που έχει και την κόκκινη γραμμή (αν υπήρχε) ή απλά τα bar handles
handles, labels = axs[2, 0].get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, 
           bbox_to_anchor=(0.5, 0.0), frameon=True, shadow=True, title="Reward Vector Configurations", title_fontsize=13)

plt.subplots_adjust(bottom=0.12, top=0.98, hspace=0.35, wspace=0.22)

plt.savefig('drl_7_configs_evaluation.pdf', format='pdf', bbox_inches='tight')
plt.savefig('drl_7_configs_evaluation.png', format='png', bbox_inches='tight')
print("✅ Τα γραφήματα ανανεώθηκαν επιτυχώς με τα πραγματικά δεδομένα και LaTeX Legend!")
