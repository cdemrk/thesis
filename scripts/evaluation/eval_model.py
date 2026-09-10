import json
import torch
import numpy as np
import time
import copy
import os

from environment import AIOrchestrationEnv
from dqn_agent import DQNAgent
from preprocessing import create_profile_table

# --- GLOBAL CONSTANTS ---
MAX_COST = 0.1500
MAX_IDLE_TIME = 250000.0
PENALTY_SCALE = 10.0
SUCCESS_BONUS = 15.0
REJECT_PENALTY = -50.0

def run_ablation_evaluation(workloads, initial_network, weights_path, w_cost, w_idle, top_k, net_type):
    env = AIOrchestrationEnv(initial_network, w_cost=w_cost, w_idle=w_idle)
    state_dim = (len(initial_network) * 4) + 3
    
    # ΣΗΜΑΝΤΙΚΟ: Δυναμικό δίκτυο (net_type) και σταθερή έξοδος (top_k=4)
    agent = DQNAgent(input_dim=state_dim, output_dim=top_k, net_type=net_type)
    
    if os.path.exists(weights_path):
        agent.policy_net.load_state_dict(torch.load(weights_path, map_location=agent.device))
    else:
        print(f"  [ΣΦΑΛΜΑ] Το αρχείο {weights_path} δεν βρέθηκε! Τα αποτελέσματα θα είναι τυχαία.")
        
    agent.policy_net.eval()  
    agent.epsilon = 0.0  
    
    total_cost, total_idle, total_latency, successful_tasks = 0, 0, 0, 0
    cumulative_reward = 0  
    total_decision_time_ms = 0.0 
    
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
        
        # === ΜΕΤΡΗΣΗ INFERENCE TIME ΤΟΥ ΜΟΝΤΕΛΟΥ ===
        start_decision = time.perf_counter()
        
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
            
        end_decision = time.perf_counter()
        total_decision_time_ms += (end_decision - start_decision) * 1000
        # ==========================================

        selected_profile = valid_profiles_list[best_action_idx]
        next_state, reward, done, info = env.step(best_action_idx)
        
        cumulative_reward += reward
        
        if "profile_used" in info and reward > env.RESOURCE_EXHAUSTION_PENALTY:
            total_cost += selected_profile['C_total']
            total_idle += selected_profile['T_idle']
            total_latency += selected_profile['T_total']
            successful_tasks += 1
            
    avg_decision_time = total_decision_time_ms / len(workloads) if len(workloads) > 0 else 0
    
    return total_cost, total_idle, total_latency, successful_tasks, avg_decision_time, env.total_initial_gpus, env.peak_used_gpus, cumulative_reward


if __name__ == "__main__":
    
    DATASETS = [
        {"name": "Scenario 1: Abundance", "file": "dataset_1_abundance_big.json"},
        {"name": "Scenario 2: Marginal", "file": "dataset_2_marginal_big.json"},
        {"name": "Scenario 3: Mild Contention", "file": "dataset_3_mild_contention_big.json"},
        {"name": "Scenario 4: Heavy Contention", "file": "dataset_4_heavy_contention_big.json"}
    ]
    
    # Σταθερές παράμετροι
    W_COST = 0.5
    W_IDLE = 0.5
    TOP_K = 4
    
    # Τα 3 μοντέλα που εκπαίδευσες στο train.py
    # Κλειδί: το όνομα του net_type | Τιμή: Το αρχείο .pth που έβγαλε το train.py
    MODELS_CONFIG = {
        "shallow":  "weights_ablation_shallow_k4.pth",
        "proposed": "weights_ablation_proposed_k4.pth",
        "deep":     "weights_ablation_deep_k4.pth"
    }

    print("="*140)
    print(" ΕΚΚΙΝΗΣΗ ΑΞΙΟΛΟΓΗΣΗΣ ΑΡΧΙΤΕΚΤΟΝΙΚΩΝ (ABLATION STUDY)")
    print(f" Σταθερά: W_cost={W_COST}, W_idle={W_IDLE}, K={TOP_K}")
    print("="*140)

    for ds in DATASETS:
        dataset_name = ds["name"]
        dataset_file = ds["file"]
        
        if not os.path.exists(dataset_file):
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
        
        for net_type, weight_file in MODELS_CONFIG.items():
            print(f"  -> Αξιολόγηση με μοντέλο {net_type.upper()} ... ", end="", flush=True)
            
            d_cost, d_idle, d_lat, d_tasks, avg_dec_time, t_gpus, peak_gpus, d_reward = run_ablation_evaluation(
                workloads=workloads, 
                initial_network=initial_network, 
                weights_path=weight_file, 
                w_cost=W_COST, 
                w_idle=W_IDLE, 
                top_k=TOP_K,
                net_type=net_type
            )
            
            avg_cost = d_cost / d_tasks if d_tasks > 0 else 0
            avg_idle = d_idle / d_tasks if d_tasks > 0 else 0
            avg_lat = d_lat / d_tasks if d_tasks > 0 else 0
            service_rate = (d_tasks / total_workloads) * 100
            gpu_util = (peak_gpus / t_gpus) * 100 if t_gpus > 0 else 0
            
            results[net_type] = {
                "Inference (ms)": f"{avg_dec_time:.3f} ms",
                "Service Rate": f"{service_rate:.1f}%",
                "Avg Cost ($)": f"{avg_cost:.5f}",
                "Avg Idle": f"{avg_idle:.0f} ms",
                "Avg Latency": f"{avg_lat:.0f} ms",
                "Peak GPU (%)": f"{peak_gpus} ({gpu_util:.1f}%)",
                "Total Reward": f"{d_reward:.1f}"
            }
            print("Ολοκληρώθηκε!")

        # --- ΕΚΤΥΠΩΣΗ ΣΥΓΚΡΙΤΙΚΟΥ ΠΙΝΑΚΑ ---
        print("\n" + "-"*115)
        header = f"{'Metric / Model':<25} | {'SHALLOW':^25} | {'PROPOSED':^25} | {'DEEP':^25}"
        print(header)
        print("-" * 115)
        
        metrics_to_print = ["Inference (ms)", "Service Rate", "Avg Cost ($)", "Avg Idle", "Avg Latency", "Peak GPU (%)", "Total Reward"]
        
        for metric in metrics_to_print:
            row_str = f"{metric:<25}"
            for m in ["shallow", "proposed", "deep"]:
                row_str += f" | {results[m][metric]:^25}"
            print(row_str)
        print("-" * 115)

    print("\n✅ Το πείραμα Ablation ολοκληρώθηκε επιτυχώς!")