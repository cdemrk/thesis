import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ΟΡΙΣΜΟΣ ΣΕΝΑΡΙΩΝ ΚΑΙ ΜΟΝΤΕΛΩΝ
# ==========================================
scenarios = ['Abundance', 'Marginal', 'Mild\nContention', 'Heavy\nContention']
x = np.arange(len(scenarios))

# Τα 3 μοντέλα που συγκρίνουμε
models = ['Shallow', 'Proposed', 'Deep']
num_models = len(models)
total_width = 0.6  # Ελαφρώς μικρότερο πλάτος για 3 μπάρες ώστε να είναι κομψές
width = total_width / num_models

# Χρώματα: Ανοιχτό μπλε (Shallow), Κανονικό μπλε (Proposed), Σκούρο μπλε (Deep)
colors = ['#85C1E9', '#2E86C1', '#1B4F72']

# ==========================================
# 2. ΕΙΣΑΓΩΓΗ ΔΕΔΟΜΕΝΩΝ (ΑΠΟ ΤΟ EVAL_ABLATION.PY)
# /!\ ΠΡΟΣΟΧΗ: Αντικατάστησε αυτά τα νούμερα με τα δικά σου! /!\
# ==========================================

# (a) Service Rate (%)
data_sr = {
    'Shallow':  [100.0, 100.0, 80.0, 40.0],
    'Proposed': [100.0, 100.0, 82.0, 43.7],
    'Deep':     [100.0, 100.0, 81.5, 41.5]
}

# (b) Average Execution Cost per Task ($)
data_cost = {
    'Shallow':  [0.002, 0.065, 0.060, 0.050],
    'Proposed': [0.002, 0.064, 0.059, 0.049],
    'Deep':     [0.002, 0.064, 0.061, 0.051]
}

# (c) Average Idle Time per Task (ms)
data_idle = {
    'Shallow':  [106900, 250000, 460000, 140000],
    'Proposed': [106935, 245786, 448189, 135024],
    'Deep':     [106950, 248000, 455000, 138000]
}

# (d) DRL Agent Inference Time (ms)
data_decision_time = {
    'Shallow':  [1.10, 1.25, 0.90, 0.40],
    'Proposed': [1.51, 1.66, 1.28, 0.69],
    'Deep':     [2.90, 3.10, 2.80, 1.50]
}

# (e) Peak GPU Utilization (%)
data_gpu = {
    'Shallow':  [45.0, 85.0, 95.0, 100.0],
    'Proposed': [45.0, 85.0, 98.0, 100.0],
    'Deep':     [45.0, 85.0, 97.0, 100.0]
}

# (f) Cumulative System Reward
data_reward = {
    'Shallow':  [458.0, 640.0,  50.0, -3100.0],
    'Proposed': [458.4, 653.8,  79.9, -3034.6],
    'Deep':     [458.1, 645.5,  60.0, -3080.0]
}

# ==========================================
# 3. ΡΥΘΜΙΣΕΙΣ ΚΑΙ ΣΧΕΔΙΑΣΗ ΠΛΕΓΜΑΤΟΣ
# ==========================================
fig, axs = plt.subplots(3, 2, figsize=(16, 16), dpi=300)

def plot_metric(ax, data_dict, title, ylabel, is_percentage=False):
    for i, m_val in enumerate(models):
        offset = (i - num_models/2 + 0.5) * width
        
        # Οπτικό τρικ: Κάνουμε το "Proposed" να ξεχωρίζει με λίγο πιο χοντρό περίγραμμα
        edge_lw = 1.5 if m_val == 'Proposed' else 0.5
        
        ax.bar(x + offset, data_dict[m_val], width, label=m_val, 
               color=colors[i], edgecolor='black', linewidth=edge_lw)

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
plot_metric(axs[1, 1], data_decision_time, '(d) DRL Agent Inference Time', 'Inference Time (ms)')
plot_metric(axs[2, 0], data_gpu, '(e) Peak GPU Utilization', 'Utilization (%)', is_percentage=True)
plot_metric(axs[2, 1], data_reward, '(f) Cumulative System Reward', 'Total Reward')

# ==========================================
# 4. ΕΝΙΑΙΟ ΥΠΟΜΝΗΜΑ ΚΑΙ ΑΠΟΘΗΚΕΥΣΗ
# ==========================================
handles, labels = axs[2, 0].get_legend_handles_labels()

# Υπόμνημα προσαρμοσμένο για 3 στοιχεία
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=13, 
           bbox_to_anchor=(0.5, 0.0), frameon=True, shadow=True, 
           title="Neural Network Architecture", title_fontsize=14)

plt.subplots_adjust(bottom=0.12, top=0.98, hspace=0.35, wspace=0.22)

# Αποθήκευση
plt.savefig('ablation_architecture_evaluation.pdf', format='pdf', bbox_inches='tight')
plt.savefig('ablation_architecture_evaluation.png', format='png', bbox_inches='tight')
print("✅ Το γράφημα Ablation (Architecture) δημιουργήθηκε επιτυχώς!")