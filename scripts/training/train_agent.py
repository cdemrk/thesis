import json
import torch
import numpy as np
from environment import AIOrchestrationEnv
from dqn_agent import DQNAgent
from preprocessing import create_profile_table

# --- GLOBAL REWARD NORMALIZATION CONSTANTS ---
MAX_COST = 0.1500
MAX_IDLE_TIME = 250000.0

def train_drl_orchestrator(dataset_path, episodes=500, w_cost=0.5, w_idle=0.5, save_name="model.pth"):
    print(f"=== Έναρξη Εκπαίδευσης DRL Agent με το {dataset_path} ===")
    print(f"Παράμετροι: W_Cost={w_cost}, W_Idle={w_idle} | Αποθήκευση σε: {save_name}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    initial_network = dataset['network']
    workloads = dataset['workloads']
    
    # 1. ΔΥΝΑΜΙΚΑ ΒΑΡΗ: Αρχικοποίηση περιβάλλοντος με τις παραμέτρους της συνάρτησης
    env = AIOrchestrationEnv(initial_network, w_cost=w_cost, w_idle=w_idle)

    state_dim = (len(initial_network) * 4) + 3 
    action_dim = 10
    
    agent = DQNAgent(input_dim=state_dim, output_dim=action_dim, lr=0.001, gamma=0.99)
    
    for episode in range(1, episodes + 1):
        env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_idle = 0
        
        workload_items = list(workloads.items())
        
        for step, (w_id, current_workload) in enumerate(workload_items):
            
            # Ορίζουμε state & is_last_step για το Rejection Penalty
            env.current_workload = current_workload
            state = env._get_state_vector()
            is_last_step = (step == len(workload_items) - 1)

            valid_profiles_dict = create_profile_table(w_id, current_workload, env.current_state)
            valid_profiles_list = list(valid_profiles_dict.values())
            
            # ΠΟΙΝΗ ΑΠΟΡΡΙΨΗΣ (REJECTION PENALTY)
            if len(valid_profiles_list) == 0:
                rejection_penalty = -50.0 
                agent.remember(state, 0, rejection_penalty, state, is_last_step)
                agent.train_step(batch_size=64)
                continue
            
            # SORTING με ομοιόμορφη κανονικοποίηση
            valid_profiles_list.sort(key=lambda p: (
                env.W_COST * min(p['C_total'] / MAX_COST, 1.0) + 
                env.W_IDLE * min(p['T_idle'] / MAX_IDLE_TIME, 1.0)
            ))
            valid_profiles_list = valid_profiles_list[:10]

            env.valid_profiles_for_current = valid_profiles_list
            
            # Επιλογή ενέργειας και βήμα στο περιβάλλον
            action_idx = agent.select_action(state, len(valid_profiles_list))
            next_state, reward, done, info = env.step(action_idx)

            agent.remember(state, action_idx, reward, next_state, is_last_step)
            agent.train_step(batch_size=64)

            episode_reward += reward
            if action_idx < len(valid_profiles_list) and reward > env.RESOURCE_EXHAUSTION_PENALTY:
                prof = valid_profiles_list[action_idx]
                episode_cost += prof['C_total']
                episode_idle += prof['T_idle']
        
        agent.update_epsilon() 
        if episode % 5 == 0:
            agent.update_target_network()
            
        print(f"Επεισόδιο {episode}/{episodes} | Epsilon: {agent.epsilon:.2f} | "
              f"Συνολικό Reward: {episode_reward:.1f} | Συνολικό Κόστος: {episode_cost:.4f}$ | "
              f"Idle Time: {episode_idle:.0f} ms")

    # 2. ΔΥΝΑΜΙΚΗ ΑΠΟΘΗΚΕΥΣΗ: Σώζουμε τα βάρη με το όνομα που δόθηκε
    torch.save(agent.policy_net.state_dict(), save_name)
    print(f"\n✅ Η εκπαίδευση ολοκληρώθηκε! Το μοντέλο αποθηκεύτηκε ως '{save_name}'.")
    return agent

if __name__ == "__main__":
    # Παράδειγμα κλήσης αν τρέξεις απευθείας το αρχείο
    trained_agent = train_drl_orchestrator(
        dataset_path="dataset_3_mild_contention_big.json", 
        episodes=1500,
        w_cost=0.5,
        w_idle=0.5,
        save_name="weights_all_cases_big_0.5_0.5.pth"
    )
