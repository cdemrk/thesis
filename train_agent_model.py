import json
import torch
import numpy as np
import argparse
import csv
from environment import AIOrchestrationEnv
from dqn_agent import DQNAgent
from preprocessing import create_profile_table

# --- GLOBAL REWARD NORMALIZATION CONSTANTS ---
MAX_COST = 0.1500
MAX_IDLE_TIME = 250000.0

# ---> ΔΙΟΡΘΩΣΗ: Βάλαμε το top_k=4 ως προεπιλογή
def train_drl_orchestrator(dataset_path, episodes=1500, w_cost=0.5, w_idle=0.5, top_k=4, net_type="proposed"):
    # Δυναμικό όνομα αποθήκευσης βάσει του δικτύου
    save_name = f"weights_ablation_{net_type}_k{top_k}.pth"
    csv_name = f"rewards_ablation_{net_type}.csv"
    
    print(f"=== Έναρξη Εκπαίδευσης DRL Agent με το {dataset_path} ===")
    print(f"Παράμετροι: Top-K={top_k} | W_Cost={w_cost}, W_Idle={w_idle} | Μοντέλο: {net_type.upper()}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    initial_network = dataset['network']
    workloads = dataset['workloads']
    
    env = AIOrchestrationEnv(initial_network, w_cost=w_cost, w_idle=w_idle)

    state_dim = (len(initial_network) * 4) + 3 
    
    # ---> ΔΙΟΡΘΩΣΗ: Το action_dim τώρα παίρνει το top_k (δηλαδή 4)
    action_dim = top_k 
    
    agent = DQNAgent(input_dim=state_dim, output_dim=action_dim, lr=0.001, gamma=0.99, net_type=net_type)
    
    episode_rewards = []
    
    for episode in range(1, episodes + 1):
        env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_idle = 0
        
        workload_items = list(workloads.items())
        
        for step, (w_id, current_workload) in enumerate(workload_items):
            
            env.current_workload = current_workload
            state = env._get_state_vector()
            is_last_step = (step == len(workload_items) - 1)

            valid_profiles_dict = create_profile_table(w_id, current_workload, env.current_state)
            valid_profiles_list = list(valid_profiles_dict.values())
            
            if len(valid_profiles_list) == 0:
                rejection_penalty = -50.0 
                agent.remember(state, 0, rejection_penalty, state, is_last_step)
                agent.train_step(batch_size=64)
                continue
            
            valid_profiles_list.sort(key=lambda p: (
                env.W_COST * min(p['C_total'] / MAX_COST, 1.0) + 
                env.W_IDLE * min(p['T_idle'] / MAX_IDLE_TIME, 1.0)
            ))
            
            # ---> ΔΙΟΡΘΩΣΗ: Κόβουμε τη λίστα στο top_k (δηλαδή 4) αντί για 10
            valid_profiles_list = valid_profiles_list[:top_k]

            env.valid_profiles_for_current = valid_profiles_list
            
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
            
        episode_rewards.append(episode_reward)
            
        print(f"Επεισόδιο {episode}/{episodes} | Epsilon: {agent.epsilon:.2f} | "
              f"Συνολικό Reward: {episode_reward:.1f} | Συνολικό Κόστος: {episode_cost:.4f}$ | "
              f"Idle Time: {episode_idle:.0f} ms")

    torch.save(agent.policy_net.state_dict(), save_name)
    print(f"\n✅ Η εκπαίδευση ολοκληρώθηκε! Το μοντέλο αποθηκεύτηκε ως '{save_name}'.")
    
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])
        for i, r in enumerate(episode_rewards):
            writer.writerow([i + 1, r])
    print(f"📊 Τα δεδομένα του γραφήματος αποθηκεύτηκαν στο '{csv_name}'.\n")
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL Agent with specific Network Architecture")
    parser.add_argument("--net_type", type=str, default="proposed", 
                        choices=["shallow", "proposed", "deep"], 
                        help="Επιλογή αρχιτεκτονικής νευρωνικού δικτύου")
    args = parser.parse_args()

    trained_agent = train_drl_orchestrator(
        dataset_path="dataset_3_mild_contention_big.json", 
        episodes=1500,
        w_cost=0.5,
        w_idle=0.5,
        top_k=4,            # <--- ΔΙΟΡΘΩΣΗ: Περνάμε το K=4 εδώ
        net_type=args.net_type
    )