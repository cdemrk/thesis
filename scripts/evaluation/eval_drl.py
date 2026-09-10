import json
import torch
import numpy as np
import time
import copy
import random
import os

from environment import AIOrchestrationEnv
from dqn_agent import DQNAgent
from preprocessing import create_profile_table

# --- GLOBAL REWARD NORMALIZATION CONSTANTS ---
MAX_COST = 0.1500
MAX_IDLE_TIME = 250000.0
PENALTY_SCALE = 10.0
SUCCESS_BONUS = 15.0
REJECT_PENALTY = -50.0

# ==========================================
# DRL AGENT EVALUATION FUNCTION
# ==========================================
def run_drl_evaluation(workloads, initial_network, weights_path, w_cost, w_idle, top_k=10, epsilon_val=0.0):
    env = AIOrchestrationEnv(initial_network, w_cost=w_cost, w_idle=w_idle)
    state_dim = (len(initial_network) * 4) + 3
    action_dim = top_k  
    
    agent = DQNAgent(input_dim=state_dim, output_dim=action_dim)
    
    # Φόρτωση των βαρών αν υπάρχει το αρχείο, αλλιώς επιστρέφει μηδενικά
    if os.path.exists(weights_path):
        agent.policy_net.load_state_dict(torch.load(weights_path, map_location=agent.device))
    else:
        print(f"  [ΠΡΟΕΙΔΟΠΟΙΗΣΗ] Το αρχείο {weights_path} δεν βρέθηκε! Τα αποτελέσματα θα είναι τυχαία.")
        
    agent.policy_net.eval()  
    agent.epsilon = epsilon_val  
    
    total_cost, total_idle, total_latency, successful_tasks = 0, 0, 0, 0
    cumulative_reward = 0  
    
    start_time = time.time()
    for w_id, current_workload in workloads.items():
        valid_profiles = create_profile_table(w_id, current_workload, env.current_state)
        valid_profiles_list = list(valid_profiles.values())
        
        if len(valid_profiles_list) == 0:
            cumulative_reward += REJECT_PENALTY
            continue
        
        valid_profiles_list.sort(key=lambda p: PENALTY_SCALE * (
            w_cost * min(p['C_total'] / MAX_COST, 1.0) + 
            w_idle * min(p['T_idle'] / MAX_IDLE_TIME, 1.0)
        ))
        valid_profiles_list = valid_profiles_list[:top_k]
            
        env.current_workload = current_workload
        env.valid_profiles_for_current = valid_profiles_list
        state = env._get_state_vector()
        
        # 1-Step Lookahead
        best_action_idx = 0
        best_q_value = -float('inf')
        
        for idx in range(len(valid_profiles_list)):
            sim_env = copy.deepcopy(env)
            next_state_vec, sim_reward, _, _ = sim_env.step(idx)
            
            if sim_reward <= env.RESOURCE_EXHAUSTION_PENALTY:
                continue
                
            state_tensor = torch.FloatTensor(next_state_vec).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                future_q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
                max_future_q = np.max(future_q_values)
                
            lookahead_q = sim_reward + (agent.gamma * max_future_q)
            
            if lookahead_q > best_q_value:
                best_q_value = lookahead_q
                best_action_idx = idx

        if best_q_value == -float('inf'):
            best_action_idx = 0
            
        selected_profile = valid_profiles_list[best_action_idx]
        next_state, reward, done, info = env.step(best_action_idx)
        
        cumulative_reward += reward
        
        if "profile_used" in info and reward > env.RESOURCE_EXHAUSTION_PENALTY:
            total_cost += selected_profile['C_total']
            total_idle += selected_profile['T_idle']
            total_latency += selected_profile['T_total']
            successful_tasks += 1
            
    execution_time = (time.time() - start_time) * 1000 
    return total_cost, total_idle, total_latency, successful_tasks, execution_time, env.total_initial_gpus, env.peak_used_gpus, cumulative_reward


# ==========================================
# ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ: BATCH EVALUATION
# ==========================================
if __name__ == "__main__":
    
    # 1. Τα 4 JSON datasets
    DATASETS = [
        {"name": "Scenario 1: 50% Demand (Abundance)", "file": "dataset_1_abundance_big.json"},
        {"name": "Scenario 2: 95% Demand (Marginal)", "file": "dataset_2_marginal_big.json"},
        {"name": "Scenario 3: 120% Demand (Mild Contention)", "file": "dataset_3_mild_contention_big.json"},
        {"name": "Scenario 4: 250% Demand (Heavy Contention)", "file": "dataset_4_heavy_contention_big.json"}
    ]
    
    # 2. Οι 7 διαφορετικές ρυθμίσεις βαρών που εκπαίδευσες
    WEIGHT_CONFIGS = {
        "0.0/1.0": {"file": "weights_all_cases_big_0.0_1.0_new.pth", "w_cost": 0.0, "w_idle": 1.0},
        "0.2/0.8": {"file": "weights_all_cases_big_0.2_0.8_new.pth", "w_cost": 0.2, "w_idle": 0.8},
        "0.4/0.6": {"file": "weights_all_cases_big_0.4_0.6_new.pth", "w_cost": 0.4, "w_idle": 0.6},
        "0.5/0.5": {"file": "weights_all_cases_big_0.5_0.5_new.pth", "w_cost": 0.5, "w_idle": 0.5},
        "0.6/0.4": {"file": "weights_all_cases_big_0.6_0.4_new.pth", "w_cost": 0.6, "w_idle": 0.4},
        "0.8/0.2": {"file": "weights_all_cases_big_0.8_0.2_new.pth", "w_cost": 0.8, "w_idle": 0.2},
        "1.0/0.0": {"file": "weights_all_cases_big_1.0_0.0_new.pth", "w_cost": 1.0, "w_idle": 0.0}
    }

    print("="*140)
    print(" ΕΚΚΙΝΗΣΗ ΜΑΖΙΚΗΣ ΑΞΙΟΛΟΓΗΣΗΣ ΒΑΡΩΝ DRL (MULTI-OBJECTIVE IMPACT - 7 CONFIGURATIONS)")
    print("="*140)

    for ds in DATASETS:
        dataset_name = ds["name"]
        dataset_file = ds["file"]
        
        if not os.path.exists(dataset_file):
            print(f"\n[ΣΦΑΛΜΑ] Το αρχείο {dataset_file} δεν βρέθηκε. Προσπέραση...")
            continue
            
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        initial_network = dataset['network']
        workloads = dataset['workloads']
        total_workloads = len(workloads)
        
        print(f"\n\n{'='*140}")
        print(f" ΕΚΤΕΛΕΣΗ ΓΙΑ: {dataset_name} ({total_workloads} Tasks)")
        print(f"{'='*140}")
        
        results = {}
        
        for config_name, config_data in WEIGHT_CONFIGS.items():
            print(f"  -> Αξιολόγηση πολιτικής W_cost/W_idle: {config_name} ... ", end="", flush=True)
            
            w_file = config_data["file"]
            w_c = config_data["w_cost"]
            w_i = config_data["w_idle"]
            
            # Τρέχουμε τον agent
            d_cost, d_idle, d_lat, d_tasks, d_time, t_gpus, peak_gpus, d_reward = run_drl_evaluation(
                workloads=workloads, 
                initial_network=initial_network, 
                weights_path=w_file, 
                w_cost=w_c, 
                w_idle=w_i, 
                top_k=10, 
                epsilon_val=0.0
            )
            
            # Υπολογισμός μέσων όρων (Προστέθηκε ο υπολογισμός του avg_idle)
            avg_cost = d_cost / d_tasks if d_tasks > 0 else 0
            avg_idle = d_idle / d_tasks if d_tasks > 0 else 0
            avg_lat = d_lat / d_tasks if d_tasks > 0 else 0
            service_rate = (d_tasks / total_workloads) * 100
            gpu_util = (peak_gpus / t_gpus) * 100 if t_gpus > 0 else 0
            
            results[config_name] = {
                "Service Rate": f"{d_tasks}/{total_workloads} ({service_rate:.1f}%)",
                "Avg Cost ($)": f"{avg_cost:.5f}",
                "Avg Idle Time (ms)": f"{avg_idle:.0f}",
                "Avg Latency (ms)": f"{avg_lat:.0f}",
                "Peak GPU (%)": f"{peak_gpus} ({gpu_util:.1f}%)",
                "Total Reward": f"{d_reward:.1f}"
            }
            print("Ολοκληρώθηκε!")

        # --- ΕΚΤΥΠΩΣΗ ΔΥΝΑΜΙΚΟΥ ΣΥΓΚΡΙΤΙΚΟΥ ΠΙΝΑΚΑ ΓΙΑ ΤΑ 7 CONFIGURATIONS ---
        print("\n" + "-"*155)
        header = f"{'Performance Metric':<20}"
        for cfg in WEIGHT_CONFIGS.keys():
            header += f" | {cfg:^15}"
        print(header)
        print("-" * 155)
        
        # Προστέθηκε το "Avg Idle Time (ms)" στη λίστα των μετρικών προς εκτύπωση
        metrics_to_print = ["Service Rate", "Avg Cost ($)", "Avg Idle Time (ms)", "Avg Latency (ms)", "Peak GPU (%)", "Total Reward"]
        
        for metric in metrics_to_print:
            row_str = f"{metric:<20}"
            for cfg in WEIGHT_CONFIGS.keys():
                row_str += f" | {results[cfg][metric]:^15}"
            print(row_str)
        print("-" * 155)

    print("\n✅ Η μαζική αξιολόγηση ολοκληρώθηκε επιτυχώς!")