import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ΟΡΙΣΜΟΣ ΣΕΝΑΡΙΩΝ ΚΑΙ ΜΟΝΤΕΛΩΝ
# ==========================================
scenarios = ['Abundance', 'Marginal', 'Mild\nContention', 'Heavy\nContention']
x = np.arange(len(scenarios))

models = ['shallow', 'proposed', 'deep']
num_models = len(models)
total_width = 0.6  # Πιο στενές μπάρες επειδή είναι μόνο 3
width = total_width / num_models

# Χρώματα: Ανοιχτό μπλε (Shallow), Κανονικό μπλε (Proposed), Σκούρο μπλε (Deep)
colors = ['#85C1E9', '#2E86C1', '#1B4F72']
labels = ['Shallow', 'Proposed', 'Deep']

# ==========================================
# 2. ΕΙΣΑΓΩΓΗ ΔΕΔΟΜΕΝΩΝ ΑΠΟ TO TERMINAL OUTPUT
# ==========================================
data_sr = {
    'shallow':  [100.0, 100.0, 83.6, 40.0],
    'proposed': [100.0, 100.0, 83.6, 41.5],
    'deep':     [100.0, 100.0, 83.6, 43.7]
}

data_cost = {
    'shallow':  [0.00201, 0.06371, 0.05822, 0.05334],
    'proposed': [0.00201, 0.05743, 0.05912, 0.05576],
    'deep':     [0.00202, 0.05014, 0.05830, 0.04932]
}

data_idle = {
    'shallow':  [106927, 285959, 439497, 260151],
    'proposed': [106903, 348123, 372418, 340583],
    'deep':     [106928, 405153, 397875, 135083]
}

data_lat = {
    'shallow':  [108809, 291604, 446794, 268131],
    'proposed': [108792, 353615, 380667, 349573],
    'deep':     [108815, 410993, 405130, 144116]
}

data_gpu = {
    'shallow':  [50.0, 95.4, 99.4, 100.0],
    'proposed': [50.0, 95.4, 97.7, 99.4],
    'deep':     [50.0, 95.4, 99.4, 99.4]
}

data_reward = {
    'shallow':  [458.4, 653.4, 143.3, -3362.6],
    'proposed': [458.4, 655.0, 147.5, -3241.1],
    'deep':     [458.4, 654.0, 143.1, -3034.8]
}

# --- NΕΑ ΔΕΔΟΜΕΝΑ INFERENCE ΜΕΤΑ ΤΟ GPU WARM-UP ---
data_decision_time = {
    'shallow':  [1.499, 1.524, 1.265, 0.590],
    'proposed': [1.504, 1.540, 1.277, 0.604],
    'deep':     [1.552, 1.585, 1.313, 0.643]
}
# --------------------------------------------------

# Κοινή συνάρτηση σχεδίασης
def plot_metric(ax, data_dict, title, ylabel, is_percentage=False):
    for i, m_val in enumerate(models):
        offset = (i - num_models/2 + 0.5) * width
        
        # Οπτικό τρικ: Κάνουμε το "Proposed" να ξεχωρίζει με λίγο πιο χοντρό περίγραμμα
        edge_lw = 1.5 if m_val == 'proposed' else 0.5
        
        ax.bar(x + offset, data_dict[m_val], width, label=labels[i], 
               color=colors[i], edgecolor='black', linewidth=edge_lw)

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

handles, leg_labels = axs[2, 0].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc='lower center', ncol=3, fontsize=13, 
           bbox_to_anchor=(0.5, 0.0), frameon=True, shadow=True, 
           title="Neural Network Architecture", title_fontsize=14)

plt.subplots_adjust(bottom=0.12, top=0.98, hspace=0.35, wspace=0.22)
fig.savefig('drl_ablation_performance.pdf', format='pdf', bbox_inches='tight')
fig.savefig('drl_ablation_performance.png', format='png', bbox_inches='tight')
plt.close(fig)

# ==========================================
# 4. ΔΗΜΙΟΥΡΓΙΑ ΕΙΚΟΝΑΣ 2: ΞΕΧΩΡΙΣΤΟ PLOT ΓΙΑ ΤΟ INFERENCE TIME
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

plot_metric(ax2, data_decision_time, 'DRL Agent Inference Time per Task', 'Inference Time (ms)')

handles2, labels2 = ax2.get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='lower center', ncol=3, fontsize=12, 
            bbox_to_anchor=(0.5, -0.05), frameon=True, shadow=True, 
            title="Neural Network Architecture", title_fontsize=13)

plt.subplots_adjust(bottom=0.2)
fig2.savefig('drl_ablation_inference.pdf', format='pdf', bbox_inches='tight')
fig2.savefig('drl_ablation_inference.png', format='png', bbox_inches='tight')
plt.close(fig2)

print("✅ Τα γραφήματα Ablation δημιουργήθηκαν επιτυχώς με τα νέα (warm-up) δεδομένα!")