import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ΟΡΙΣΜΟΣ ΣΕΝΑΡΙΩΝ ΚΑΙ ΑΛΓΟΡΙΘΜΩΝ
# ==========================================
scenarios = ['Abundance\n(7 Tasks)', 'Marginal\n(18 Tasks)', 'Mild Contention\n(19 Tasks)', 'Heavy Contention\n(40 Tasks)']
x = np.arange(len(scenarios))

algorithms = ['Greedy Baseline', 'DRL Agent', 'MILP Optimal']
num_algos = len(algorithms)
total_width = 0.65  
width = total_width / num_algos

# Χρώματα: Γκρι για Greedy, Μπλε για DRL, Πράσινο για MILP
colors = ['#95A5A6', '#2980B9', '#27AE60']

# ==========================================
# 2. ΕΙΣΑΓΩΓΗ ΔΕΔΟΜΕΝΩΝ (Με τα ανανεωμένα νούμερα)
# ==========================================
data_cost = {
    'Greedy Baseline': [0.006569, 0.014036, 0.009163, 0.034650],
    'DRL Agent':       [0.006569, 0.012868, 0.011229, 0.0112],
    'MILP Optimal':    [0.006569, 0.008087, 0.007102, 0.006338]
}

data_idle = {
    'Greedy Baseline': [117059.3, 81936.4, 253094.6, 80644.2],
    'DRL Agent':       [171725.8, 116226.4, 261940.4, 198502.8],
    'MILP Optimal':    [117059.3, 81088.8, 80856.4, 23227.4]
}

data_latency = {
    'Greedy Baseline': [118635.6, 84283.1, 255271.7, 82351.8],
    'DRL Agent':       [173302.1, 118571.3, 264684.0, 200384.6],
    'MILP Optimal':    [118635.6, 83532.6, 83329.6, 26417.5]
}

data_gpu = {
    'Greedy Baseline': [52.1, 93.8, 91.7, 100.0],
    'DRL Agent':       [52.1, 93.8, 97.9, 100.0],
    'MILP Optimal':    [52.1, 93.8, 91.7, 97.9]
}

data_reward = {
    'Greedy Baseline': [88.30, 241.06, -9.34, -926.31],
    'DRL Agent':       [85.94, 234.52, 49.30, -854.12],
    'MILP Optimal':    [100.74, 260.28, 146.39, -252.79]
}

data_service = {
    'Greedy Baseline': [100.0, 100.0, 78.9, 42.5],
    'DRL Agent':       [100.0, 100.0, 84.2, 45.0],
    'MILP Optimal':    [100.0, 100.0, 89.5, 67.5]
}

data_time = {
    'Greedy Baseline': [7.31, 7.96, 8.92, 7.96],
    'DRL Agent':       [88.34, 94.72, 95.32, 94.77],
    'MILP Optimal':    [179.06, 345.53, 582.02, 1305.20]
}

# ==========================================
# 3. ΓΕΝΙΚΗ ΣΥΝΑΡΤΗΣΗ ΣΧΕΔΙΑΣΗΣ ΜΠΑΡΩΝ
# ==========================================
def plot_metric(ax, data_dict, title, ylabel, is_percentage=False, show_legend=False):
    for i, algo in enumerate(algorithms):
        offset = (i - num_algos/2 + 0.5) * width
        edge_lw = 1.5 if algo == 'DRL Agent' else 0.5
        
        ax.bar(x + offset, data_dict[algo], width, label=algo, 
               color=colors[i], edgecolor='black', linewidth=edge_lw)

    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    if is_percentage:
        ax.set_ylim(0, 110)
        
    if show_legend:
        ax.legend(loc='best', fontsize=10, frameon=True, shadow=False)

# ==========================================
# 4. ΓΡΑΦΗΜΑ 1: ΠΛΕΓΜΑ 2x2 (Cost, Idle, Latency, GPU)
# ==========================================
fig_grid, axs_grid = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

plot_metric(axs_grid[0, 0], data_cost, '(a) Average Execution Cost per Task', 'Cost ($)')
plot_metric(axs_grid[0, 1], data_idle, '(b) Average Idle Time per Task', 'Idle Time (ms)')
plot_metric(axs_grid[1, 0], data_latency, '(c) Average Latency per Task', 'Latency (ms)')
plot_metric(axs_grid[1, 1], data_gpu, '(d) Peak GPU Utilization', 'Percentage (%)', is_percentage=True)

# Υπόμνημα για το 2x2 πλέγμα
handles, labels = axs_grid[0, 0].get_legend_handles_labels()
fig_grid.legend(handles, labels, loc='lower center', ncol=3, fontsize=13, 
                bbox_to_anchor=(0.5, 0.01), frameon=True, shadow=True, 
                title="Orchestration Algorithm", title_fontsize=14)

plt.subplots_adjust(bottom=0.12, top=0.94, hspace=0.3, wspace=0.2)
fig_grid.savefig('resource_metrics_grid.pdf', format='pdf', bbox_inches='tight')
fig_grid.savefig('resource_metrics_grid.png', format='png', bbox_inches='tight')
plt.close(fig_grid)
print("✅ 1/4: Το 2x2 Grid (Cost, Idle, Latency, GPU) δημιουργήθηκε επιτυχώς!")

# ==========================================
# 5. ΓΡΑΦΗΜΑ 2: REWARD (ΞΕΧΩΡΙΣΤΟ)
# ==========================================
fig_rew, ax_rew = plt.subplots(figsize=(9, 6), dpi=300)
plot_metric(ax_rew, data_reward, 'Cumulative System Reward per Scenario', 'Total Reward', show_legend=True)
plt.tight_layout()
fig_rew.savefig('cumulative_reward.pdf', format='pdf', bbox_inches='tight')
fig_rew.savefig('cumulative_reward.png', format='png', bbox_inches='tight')
plt.close(fig_rew)
print("✅ 2/4: Το γράφημα Reward δημιουργήθηκε επιτυχώς!")

# ==========================================
# 6. ΓΡΑΦΗΜΑ 3: SERVICE RATE (ΞΕΧΩΡΙΣΤΟ)
# ==========================================
fig_srv, ax_srv = plt.subplots(figsize=(9, 6), dpi=300)
plot_metric(ax_srv, data_service, 'Service Rate (Successful Tasks)', 'Percentage (%)', is_percentage=True, show_legend=True)
plt.tight_layout()
fig_srv.savefig('service_rate.pdf', format='pdf', bbox_inches='tight')
fig_srv.savefig('service_rate.png', format='png', bbox_inches='tight')
plt.close(fig_srv)
print("✅ 3/4: Το γράφημα Service Rate δημιουργήθηκε επιτυχώς!")

# ==========================================
# 7. ΓΡΑΦΗΜΑ 4: ΧΡΟΝΟΣ ΑΠΟΦΑΣΗΣ (ΚΑΝΟΝΙΚΗ ΓΡΑΜΜΙΚΗ ΚΛΙΜΑΚΑ)
# ==========================================
fig_time, ax_time = plt.subplots(figsize=(9, 6), dpi=300)

for i, algo in enumerate(algorithms):
    offset = (i - num_algos/2 + 0.5) * width
    edge_lw = 1.5 if algo == 'DRL Agent' else 0.5
    ax_time.bar(x + offset, data_time[algo], width, label=algo, 
                color=colors[i], edgecolor='black', linewidth=edge_lw)

ax_time.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
ax_time.set_title('Algorithmic Decision Time Comparison', fontsize=13, fontweight='bold')
ax_time.set_xticks(x)
ax_time.set_xticklabels(scenarios, fontsize=10)
ax_time.grid(axis='y', linestyle='--', alpha=0.5, which="major")
ax_time.set_axisbelow(True)

# Εξασφαλίζουμε ότι ο άξονας Y ξεκινάει από το 0 για σωστή γραμμική απεικόνιση
ax_time.set_ylim(bottom=0)

ax_time.legend(loc='upper left', fontsize=10, frameon=True, shadow=False)

plt.tight_layout()
fig_time.savefig('decision_time.pdf', format='pdf', bbox_inches='tight')
fig_time.savefig('decision_time.png', format='png', bbox_inches='tight')
plt.close(fig_time)
print("✅ 4/4: Το γράφημα Decision Time (Γραμμική Κλίμακα) δημιουργήθηκε επιτυχώς!")

print("\n🎉 Όλα τα διαγράμματα έχουν παραχθεί και αποθηκευτεί!")