import json
import torch
import numpy as np
import time
import pulp 
import copy
import random
import argparse

from environment import AIOrchestrationEnv
from dqn_agent import DQNAgent
from preprocessing import create_profile_table

# --- GLOBAL REWARD NORMALIZATION CONSTANTS ---
MAX_COST = 0.1500
MAX_IDLE_TIME = 1500000.0
PENALTY_SCALE = 10.0
SUCCESS_BONUS = 15.0
REJECT_PENALTY = -50.0

# ==========================================
# 1. GREEDY BASELINE
# ==========================================
def run_greedy_heuristic(workloads, initial_network, w_cost, w_idle):
    env = AIOrchestrationEnv(initial_network, w_cost=w_cost, w_idle=w_idle)
    total_cost, total_idle, total_latency, successful_tasks = 0, 0, 0, 0
    cumulative_reward = 0  
    task_choices = {} 
    
    start_time = time.time()
    for w_id, current_workload in workloads.items():
        valid_profiles = create_profile_table(w_id, current_workload, env.current_state)
        valid_profiles_list = list(valid_profiles.values())
        
        if len(valid_profiles_list) == 0:
            task_choices[w_id] = "ΑΠΟΡΡΙΦΘΗΚΕ (Δεν βρέθηκε προφίλ)"
            cumulative_reward += REJECT_PENALTY
            continue
            
        # Ταξινόμηση - Η Greedy επιλέγει πάντα το 0 (το ελάχιστο κανονικοποιημένο penalty)
        valid_profiles_list.sort(key=lambda p: PENALTY_SCALE * (
            w_cost * min(p['C_total'] / MAX_COST, 1.0) + 
            w_idle * min(p['T_idle'] / MAX_IDLE_TIME, 1.0)
        ))

        env.current_workload = current_workload
        env.valid_profiles_for_current = valid_profiles_list
        selected_profile = valid_profiles_list[0]
        
        # Εκτέλεση στο περιβάλλον
        next_state, reward, done, info = env.step(0)
        cumulative_reward += reward
        
        if "profile_used" in info:
            total_cost += selected_profile['C_total']
            total_idle += selected_profile['T_idle']
            total_latency += selected_profile['T_total']
            successful_tasks += 1
            strat = selected_profile['strategy']
            nodes = ", ".join(selected_profile['placement'])
            task_choices[w_id] = f"{strat} [{nodes}]"
        else:
            task_choices[w_id] = "ΑΠΟΡΡΙΦΘΗΚΕ (Γεμάτοι πόροι)"
            
    execution_time = (time.time() - start_time) * 1000 
    return total_cost, total_idle, total_latency, successful_tasks, execution_time, env.total_initial_gpus, env.peak_used_gpus, task_choices, cumulative_reward


# ==========================================
# 2. DRL AGENT (EPSILON-GREEDY)
# ==========================================
def run_drl_evaluation(workloads, initial_network, weights_path, w_cost, w_idle, top_k, epsilon_val):
    env = AIOrchestrationEnv(initial_network, w_cost=w_cost, w_idle=w_idle)
    state_dim = (len(initial_network) * 4) + 3
    action_dim = top_k  
    
    # Αυτόματη ανίχνευση του net_type από το όνομα του αρχείου
    detected_net_type = "deep" if "deep" in weights_path.lower() else "proposed"
    agent = DQNAgent(input_dim=state_dim, output_dim=top_k, net_type=detected_net_type)
    
    # Φόρτωση βαρών με έξυπνο Mapping για αποφυγή KeyErrors
    checkpoint = torch.load(weights_path, map_location=agent.device)
    agent_state_keys = agent.policy_net.state_dict().keys()
    
    if "fc1.weight" in checkpoint and "network.0.weight" in agent_state_keys:
        new_checkpoint = {}
        # Δυναμικό mapping ανάλογα με το αν το agent δίκτυο έχει 3 ή 4 επίπεδα (network.4 ή network.6 είναι το out)
        has_network_6 = "network.6.weight" in agent_state_keys
        
        mapping = {
            "fc1.weight": "network.0.weight", "fc1.bias": "network.0.bias",
            "fc2.weight": "network.2.weight", "fc2.bias": "network.2.bias",
        }
        
        if has_network_6:
            mapping.update({
                "fc3.weight": "network.4.weight", "fc3.bias": "network.4.bias",
                "out.weight": "network.6.weight", "out.bias": "network.6.bias"
            })
        else:
            mapping.update({
                "fc3.weight": "network.4.weight", "fc3.bias": "network.4.bias", # Fallback for mismatch
                "out.weight": "network.4.weight", "out.bias": "network.4.bias"
            })
            
        for k, v in checkpoint.items():
            mapped_key = mapping.get(k, k)
            new_checkpoint[mapped_key] = v
        checkpoint = new_checkpoint

    # Φόρτωση του checkpoint στο μοντέλο
    agent.policy_net.load_state_dict(checkpoint, strict=False)
    agent.policy_net.eval()  
    
    agent.epsilon = epsilon_val  
    
    total_cost, total_idle, total_latency, successful_tasks = 0, 0, 0, 0
    cumulative_reward = 0  
    task_choices = {} 
    
    start_time = time.time()
    for w_id, current_workload in workloads.items():
        valid_profiles = create_profile_table(w_id, current_workload, env.current_state)
        valid_profiles_list = list(valid_profiles.values())
        
        if len(valid_profiles_list) == 0:
            task_choices[w_id] = "ΑΠΟΡΡΙΦΘΗΚΕ (Δεν βρέθηκε προφίλ)"
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
        
        # Epsilon-Greedy & 1-Step Lookahead
        best_action_idx = 0
        
        if random.random() < agent.epsilon:
            best_action_idx = random.randint(0, len(valid_profiles_list) - 1)
        else:
            best_q_value = -float('inf')
            
            for idx in range(len(valid_profiles_list)):
                sim_env = copy.deepcopy(env)
                next_state_vec, sim_reward, _, info_sim = sim_env.step(idx)
                
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
                
        action_idx = best_action_idx
        selected_profile = valid_profiles_list[action_idx]
        next_state, reward, done, info = env.step(action_idx)
        
        cumulative_reward += reward
        
        if "profile_used" in info and reward > env.RESOURCE_EXHAUSTION_PENALTY:
            total_cost += selected_profile['C_total']
            total_idle += selected_profile['T_idle']
            total_latency += selected_profile['T_total']
            successful_tasks += 1
            strat = selected_profile['strategy']
            nodes = ", ".join(selected_profile['placement'])
            task_choices[w_id] = f"{strat} [{nodes}]"
        else:
            task_choices[w_id] = "ΑΠΟΡΡΙΦΘΗΚΕ (Γεμάτοι πόροι)"
            
    execution_time = (time.time() - start_time) * 1000 
    return total_cost, total_idle, total_latency, successful_tasks, execution_time, env.total_initial_gpus, env.peak_used_gpus, task_choices, cumulative_reward


# ==========================================
# 3. MILP OPTIMAL SOLVER (OFFLINE)
# ==========================================
def run_milp_solver(workloads, initial_network, w_cost, w_idle):
    total_cost, total_idle, total_latency, successful_tasks = 0, 0, 0, 0
    cumulative_reward = 0
    task_choices = {}
    
    start_time = time.time()
    all_profiles = {}
    for w_id, w_data in workloads.items():
        profs = create_profile_table(w_id, w_data, initial_network)
        all_profiles[w_id] = profs
        if len(profs) == 0:
            task_choices[w_id] = "ΑΠΟΡΡΙΦΘΗΚΕ (Αδύνατο εξαρχής)"

    prob = pulp.LpProblem("AI_Orchestration_MILP", pulp.LpMinimize)
    x = {}
    for w_id, profs in all_profiles.items():
        for p_id in profs.keys():
            x[(w_id, p_id)] = pulp.LpVariable(f"x_{w_id}_{p_id}", cat='Binary')

    # Objective Function με Κανονικοποίηση
    obj_terms = []
    for w_id, profs in all_profiles.items():
        for p_id, prof in profs.items():
            norm_cost = min(prof['C_total'] / MAX_COST, 1.0)
            norm_idle = min(prof['T_idle'] / MAX_IDLE_TIME, 1.0)
            penalty_score = PENALTY_SCALE * ((w_cost * norm_cost) + (w_idle * norm_idle))
            
            obj_terms.append(x[(w_id, p_id)] * (penalty_score - SUCCESS_BONUS))
    prob += pulp.lpSum(obj_terms)

    for w_id in all_profiles.keys():
        prob += pulp.lpSum(x[(w_id, p_id)] for p_id in all_profiles[w_id].keys()) <= 1

    for v in initial_network.keys():
        prob += pulp.lpSum(x[(w_id, p_id)] * all_profiles[w_id][p_id]['gpus'].get(v, 0) for w_id in all_profiles.keys() for p_id in all_profiles[w_id].keys()) <= initial_network[v]['N_gpu']
        prob += pulp.lpSum(x[(w_id, p_id)] * all_profiles[w_id][p_id]['uplink'].get(v, 0) for w_id in all_profiles.keys() for p_id in all_profiles[w_id].keys()) <= initial_network[v]['U_avail']
        prob += pulp.lpSum(x[(w_id, p_id)] * all_profiles[w_id][p_id]['downlink'].get(v, 0) for w_id in all_profiles.keys() for p_id in all_profiles[w_id].keys()) <= initial_network[v]['D_avail']

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    execution_time = (time.time() - start_time) * 1000
    
    peak_gpus = 0
    total_gpus = sum(res['N_gpu'] for res in initial_network.values())

    if pulp.LpStatus[prob.status] in ['Optimal', 'Feasible']:
        for w_id, profs in all_profiles.items():
            placed = False
            for p_id, prof in profs.items():
                if x[(w_id, p_id)].varValue is not None and x[(w_id, p_id)].varValue > 0.5:
                    placed = True
                    total_cost += prof['C_total']
                    total_idle += prof['T_idle']
                    total_latency += prof['T_total']
                    peak_gpus += sum(prof['gpus'].values())
                    successful_tasks += 1
                    
                    norm_cost = min(prof['C_total'] / MAX_COST, 1.0)
                    norm_idle = min(prof['T_idle'] / MAX_IDLE_TIME, 1.0)
                    r_metric = SUCCESS_BONUS - PENALTY_SCALE * ((w_cost * norm_cost) + (w_idle * norm_idle))
                    cumulative_reward += r_metric
                    
                    strat = prof['strategy']
                    nodes = ", ".join(prof['placement'])
                    task_choices[w_id] = f"{strat} [{nodes}]"
                    
            if not placed and w_id not in task_choices:
                task_choices[w_id] = "ΑΠΟΡΡΙΦΘΗΚΕ (Από MILP)"
                cumulative_reward += REJECT_PENALTY
    else:
        for w_id in workloads.keys():
            task_choices[w_id] = "ΣΦΑΛΜΑ MILP"

    return total_cost, total_idle, total_latency, successful_tasks, execution_time, total_gpus, peak_gpus, task_choices, cumulative_reward


# ==========================================
# ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ ΑΞΙΟΛΟΓΗΣΗΣ
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Orchestration Evaluation")
    parser.add_argument('--algo', type=str, choices=['greedy', 'drl', 'milp', 'all'], default='all',
                        help='Choose which algorithm to run isolated')
    parser.add_argument('--dataset', type=str, default='small_dataset_new.json', help='Path to workload JSON')
    parser.add_argument('--weights', type=str, default='weights_all_executed.pth', help='Path to DRL weights')
    
    parser.add_argument('--w_cost', type=float, default=0.5, help='Weight for Cost (0.0 to 1.0)')
    parser.add_argument('--w_idle', type=float, default=0.5, help='Weight for Latency/Idle (0.0 to 1.0)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of profiles to mask (Max 10)')
    parser.add_argument('--epsilon', type=float, default=0.0, help='Epsilon for DRL exploration (0.0 = deterministic)')
    
    args = parser.parse_args()

    with open(args.dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    initial_network = dataset['network']
    workloads = dataset['workloads']
    
    print(f"=== Έναρξη Αξιολόγησης ({args.algo.upper()}) στο {args.dataset} ===")
    print(f"Παράμετροι: W_Cost={args.w_cost}, W_Idle={args.w_idle}, Top-K={args.top_k}, Epsilon={args.epsilon}")
    print("-" * 125)

    # 1. Αρχικοποίηση όλων των μεταβλητών στο 0 ώστε να μην "κρασάρει" το print
    g_cost = g_idle = g_lat = g_time = r_g = 0.0
    d_cost = d_idle = d_lat = d_time = r_d = 0.0
    m_cost = m_idle = m_lat = m_time = r_m = 0.0
    g_tasks = d_tasks = m_tasks = 0
    g_peak_used = d_peak_used = m_peak_used = 0
    g_choices, d_choices, m_choices = {}, {}, {}
    total_gpus = sum(res['N_gpu'] for res in initial_network.values())

    # 2. Εκτέλεση
    if args.algo in ['greedy', 'all']:
        print("Εκτέλεση Greedy Baseline...")
        g_cost, g_idle, g_lat, g_tasks, g_time, _, g_peak_used, g_choices, r_g = run_greedy_heuristic(
            workloads, initial_network, args.w_cost, args.w_idle)

    if args.algo in ['drl', 'all']:
        print("Εκτέλεση DRL Agent...")
        d_cost, d_idle, d_lat, d_tasks, d_time, _, d_peak_used, d_choices, r_d = run_drl_evaluation(
            workloads, initial_network, args.weights, args.w_cost, args.w_idle, args.top_k, args.epsilon)
        
    if args.algo in ['milp', 'all']:
        print("Επίλυση MILP (Παρακαλώ περιμένετε)...")
        m_cost, m_idle, m_lat, m_tasks, m_time, _, m_peak_used, m_choices, r_m = run_milp_solver(
            workloads, initial_network, args.w_cost, args.w_idle)

    # 3. Εκτύπωση Πάντα του Συγκριτικού Πίνακα
    print("\n" + "="*48 + " ΕΠΙΛΟΓΕΣ ΠΡΟΦΙΛ ΑΝΑ ΕΡΓΑΣΙΑ " + "="*48)
    print(f"{'Εργασία':<8} | {'Greedy Baseline':<35} | {'DRL Agent':<35} | {'MILP (Global Optimal)':<35}")
    print("-" * 125)
    for w_id in workloads.keys():
        gc = g_choices.get(w_id, "N/A")
        dc = d_choices.get(w_id, "N/A")
        mc = m_choices.get(w_id, "N/A")
        print(f"{w_id:<8} | {gc:<35} | {dc:<35} | {mc:<35}")
    
    g_percent = round((g_peak_used / total_gpus) * 100, 1) if total_gpus > 0 else 0
    d_percent = round((d_peak_used / total_gpus) * 100, 1) if total_gpus > 0 else 0
    m_percent = round((m_peak_used / total_gpus) * 100, 1) if total_gpus > 0 else 0

    avg_g_cost = g_cost / g_tasks if g_tasks > 0 else 0
    avg_d_cost = d_cost / d_tasks if d_tasks > 0 else 0
    avg_m_cost = m_cost / m_tasks if m_tasks > 0 else 0
    
    avg_g_idle = g_idle / g_tasks if g_tasks > 0 else 0
    avg_d_idle = d_idle / d_tasks if d_tasks > 0 else 0
    avg_m_idle = m_idle / m_tasks if m_tasks > 0 else 0
    
    avg_g_lat = g_lat / g_tasks if g_tasks > 0 else 0
    avg_d_lat = d_lat / d_tasks if d_tasks > 0 else 0
    avg_m_lat = m_lat / m_tasks if m_tasks > 0 else 0

    print("\n" + "="*48 + " ΣΤΑΤΙΣΤΙΚΑ ΠΕΙΡΑΜΑΤΟΣ " + "="*53)
    print(f"{'Μετρική Απόδοσης':<35} | {'Greedy Baseline':<25} | {'DRL Agent':<25} | {'MILP (Global Optimal)':<30}")
    print("-" * 125)
    print(f"{'Εξυπηρετημένες Εργασίες (Tasks)':<35} | {g_tasks:<25} | {d_tasks:<25} | {m_tasks:<30}")
    print(f"{'Μέσο Κόστος ανά Task ($)':<35} | {avg_g_cost:<25.6f} | {avg_d_cost:<25.6f} | {avg_m_cost:<30.6f}")
    print(f"{'Μέσος Χρόνος Αδράνειας (ms)':<35} | {avg_g_idle:<25.1f} | {avg_d_idle:<25.1f} | {avg_m_idle:<30.1f}")
    print(f"{'Μέσο Latency ανά Task (ms)':<35} | {avg_g_lat:<25.1f} | {avg_d_lat:<25.1f} | {avg_m_lat:<30.1f}")
    print(f"{'Συνολική Χωρητικότητα (GPUs)':<35} | {total_gpus:<25} | {total_gpus:<25} | {total_gpus:<30}")
    print(f"{'Μέγιστη Χρήση (Peak GPUs)':<35} | {f'{g_peak_used} ({g_percent}%)':<25} | {f'{d_peak_used} ({d_percent}%)':<25} | {f'{m_peak_used} ({m_percent}%)':<30}")
    print("-" * 125)
    print(f"{'Συνολικό Reward Συστήματος':<35} | {r_g:<25.2f} | {r_d:<25.2f} | {r_m:<30.2f}")
    print("-" * 125)
    print(f"-> Χρόνος Απόφασης Greedy : {g_time:.2f} ms")
    print(f"-> Χρόνος Απόφασης DRL    : {d_time:.2f} ms")
    print(f"-> Χρόνος Απόφασης MILP   : {m_time:.2f} ms")
    print("="*125)
