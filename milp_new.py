import itertools
import math
import pulp

# ==========================================
# 1. ΠΑΡΑΜΕΤΡΟΙ ΚΑΙ ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ==========================================
TIERS = [1, 2, 4, 8]        
UNIT_SPEED_MBPS = 50        

# ΝΕΑ ΠΑΡΑΜΕΤΡΟΣ: Κόστος Amazon AWS ανά δευτερόλεπτο για 1 GPU
# (π.χ. 3.60$ την ώρα / 3600 δευτερόλεπτα = 0.001$ ανά sec)
AWS_GPU_COST_PER_SEC = 0.001 

# Βάρη Πολυκριτηριακής Βελτιστοποίησης (Objective Function)
W_COST = 0.6  # 60% Βαρύτητα στο Οικονομικό Κόστος
W_IDLE = 0.4  # 40% Βαρύτητα στον Χρόνο Αδράνειας των πόρων

def get_max_tier_for_data(data_size):
    if data_size < 2: return 1      
    elif data_size < 10: return 2   
    elif data_size < 50: return 4   
    else: return 8                

def calc_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

# ==========================================
# 2. PREPROCESSING: ΔΗΜΙΟΥΡΓΙΑ ΠΡΟΦΙΛ 
# ==========================================
def create_profile_table(workload_id, workload, network_state):
    valid_profiles = {}
    
    n_a = workload['n_a']           
    g_a = workload['g_a']           
    N_total = n_a * g_a             # Συνολικές GPUs
    Lambda_a = workload['Lambda_a'] 
    D_a = workload['D_a']           # Data size (Model/Gradients)
    A_a = workload.get('A_a', 0)    # Activations size (για TP & PP)
    L = workload.get('L', 1)        # Layers (για TP)
    m = workload.get('m', 1)        # Micro-batches (για PP)
    G_a = workload['G_a']           
    type_a = workload['type_a']     
    K_a = workload.get('K_a', 1)    
    
    # Επιλογή Στρατηγικών βάσει τύπου εργασίας
    if type_a == 1: # TRAINING
        if n_a == 1: allowed_strategies = ['Single']
        else: allowed_strategies = ['PS', 'AR', 'TP', 'PP']
    else:           # INFERENCE
        if n_a == 1: allowed_strategies = ['Single']
        else: allowed_strategies = ['TP', 'PP'] 
    
    max_tier_data = get_max_tier_for_data(max(D_a, A_a))
    allowed_tiers = [b for b in TIERS if b <= max_tier_data]
    physical_nodes = list(network_state.keys())
    
    # Επιβάλλεται η χρήση διαφορετικών κόμβων όταν n_a > 1
    placements = list(itertools.permutations(physical_nodes, n_a))
        
    profile_counter = 1
    
    for placement in placements:
        # --- ΦΙΛΤΡΟ 1: Διαθεσιμότητα GPUs και VRAM ---
        gpu_needs = {v: 0 for v in physical_nodes}
        unique_nodes = set(placement)
        for phys_n in placement:
            gpu_needs[phys_n] += g_a 
            
        valid_hw = True
        for v, req_gpus in gpu_needs.items():
            if req_gpus > network_state[v]['N_gpu']:
                valid_hw = False
                break
            
            vram_required = req_gpus * 4 
            if vram_required > network_state[v]['M_v']:
                valid_hw = False
                break
                
        if not valid_hw: continue
            
        # Υπολογισμός ταχυτήτων F_v για το συγκεκριμένο placement
        f_v_list = [network_state[v]['F_v'] for v in placement]
        sum_f_v = sum(g_a * f for f in f_v_list)
        min_f_v = min(f_v_list)
        
        # Max απόσταση μεταξύ των κόμβων (worst-case t_prop)
        max_inter_dist = 0
        if len(unique_nodes) > 1:
            max_inter_dist = max(calc_distance(network_state[u]['coords_v'], network_state[v]['coords_v']) 
                                 for u in unique_nodes for v in unique_nodes)
        t_prop_inter = max_inter_dist * 0.1 
        
        # ΕΞΕΤΑΣΗ ΟΛΩΝ ΤΩΝ ΕΠΙΤΡΕΠΤΩΝ ΣΤΡΑΤΗΓΙΚΩΝ
        for strategy in allowed_strategies:
            
            # --- ΥΠΟΛΟΓΙΣΜΟΣ T_comp ---
            if strategy == 'Single':
                t_comp = (G_a / sum_f_v) * 1000
            elif strategy == 'PS':
                ps_f_v = f_v_list[0]
                worker_power = sum_f_v - ps_f_v 
                t_comp = (G_a / worker_power) * 1000 if worker_power > 0 else float('inf')
            elif strategy in ['AR', 'TP']:
                t_comp = (G_a / (N_total * min_f_v)) * 1000
            elif strategy == 'PP':
                t_comp = (G_a / (N_total * min_f_v)) * (1 + (N_total - 1) / m) * 1000

            # --- ΥΠΟΛΟΓΙΣΜΟΣ ΔΙΚΤΥΟΥ ---
            for b in allowed_tiers:
                uplink_needs = {v: 0 for v in physical_nodes}
                downlink_needs = {v: 0 for v in physical_nodes}
                t_comm = 0
                
                if len(unique_nodes) > 1:
                    for v in unique_nodes:
                        uplink_needs[v] += b
                        downlink_needs[v] += b
                        
                    if strategy == 'PS':
                        t_trans = (2 * (N_total - 1) * D_a * 8) / (b * UNIT_SPEED_MBPS) * 1000
                        t_comm = K_a * (t_trans + t_prop_inter)
                    elif strategy == 'AR':
                        t_trans = (2 * (N_total - 1) * D_a * 8) / (N_total * b * UNIT_SPEED_MBPS) * 1000
                        t_comm = K_a * (t_trans + t_prop_inter)
                    elif strategy == 'TP':
                        t_trans = (2 * (N_total - 1) * A_a * 8) / (N_total * b * UNIT_SPEED_MBPS) * 1000
                        t_comm = K_a * L * (t_trans + t_prop_inter)
                    elif strategy == 'PP':
                        t_trans = (2 * A_a * 8) / (b * UNIT_SPEED_MBPS) * 1000
                        t_comm = K_a * m * (t_trans + t_prop_inter)
                
                # --- ΦΙΛΤΡΟ 2: Δίκτυο ---
                valid_network = True
                for v in physical_nodes:
                    if uplink_needs[v] > network_state[v]['U_avail'] or downlink_needs[v] > network_state[v]['D_avail']:
                        valid_network = False
                        break
                if not valid_network: continue 
                    
                # --- ΦΙΛΤΡΟ 3: SLA ---
                t_access = 5.0 
                t_total = t_comp + t_access + t_comm
                
                if t_total <= Lambda_a:
                    # ΝΕΟΙ ΥΠΟΛΟΓΙΣΜΟΙ ΒΑΣΕΙ ΤΗΣ ΝΕΑΣ ΘΕΩΡΙΑΣ:
                    
                    # 1. Υπολογισμός Χρόνου Αδράνειας (Idle Time)
                    # Ιδανικός χρόνος χωρίς δικτυακά overheads και bubbles
                    ideal_t_comp = (G_a / sum_f_v) * 1000 
                    t_idle = max(0, t_total - ideal_t_comp) # t_total αντί t_comp για να πιάνει και την αναμονή δικτύου
                    
                    # 2. Υπολογισμός Οικονομικού Κόστους Amazon AWS
                    g_cloud = gpu_needs.get('Cloud', 0)
                    cost_total = g_cloud * AWS_GPU_COST_PER_SEC * (t_comp / 1000.0)
                    
                    profile_id = f"P{profile_counter}_{strategy}" 
                    valid_profiles[(workload_id, profile_id)] = {
                        'strategy': strategy,
                        'placement': placement,
                        'b_req': b,
                        'T_total': t_total,
                        'T_comp': t_comp,
                        'T_comm': t_comm,
                        'T_idle': t_idle,       # <--- ΝΕΟ ΠΕΔΙΟ
                        'C_total': cost_total,   # <--- ΝΕΟ ΠΕΔΙΟ
                        'gpus': gpu_needs,
                        'uplink': uplink_needs,
                        'downlink': downlink_needs
                    }
                    profile_counter += 1
                
    return valid_profiles

# ==========================================
# 3. ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ ΚΑΙ MILP SOLVER
# ==========================================
if __name__ == "__main__":
    print("=== Εκκίνηση Ενορχηστρωτή AI (Multi-Objective: Cost & Idle Time) ===")
    print("-" * 75)
    
    network_snapshot = {
        'Edge1': {'N_gpu': 2, 'F_v': 50,  'M_v': 16, 'U_avail': 15, 'D_avail': 15, 'coords_v': (10, 10)},
        'Edge2': {'N_gpu': 2, 'F_v': 50,  'M_v': 16, 'U_avail': 15, 'D_avail': 15, 'coords_v': (20, 30)},
        'Cloud': {'N_gpu': 8, 'F_v': 150, 'M_v': 80, 'U_avail': 100, 'D_avail': 100, 'coords_v': (100, 100)}
    }

    workloads = {
        'W1': { 
            'type_a': 1, 'n_a': 3, 'g_a': 1, 
            'G_a': 150, 'D_a': 12, 'A_a': 4, 'L': 12, 'm': 4, 'K_a': 20, 
            'Lambda_a': 15000
        },
        'W2': {
            'type_a': 0, 'n_a': 2, 'g_a': 1, 
            'G_a': 30, 'D_a': 0, 'A_a': 2, 'L': 6, 'm': 2, 'K_a': 1, 
            'Lambda_a': 5000
        }
    }
    
    all_generated_profiles = {}
    for w_id, w_data in workloads.items():
        w_profiles = create_profile_table(w_id, w_data, network_snapshot)
        print(f"-> Εργασία {w_id}: Παρήχθησαν {len(w_profiles)} έγκυρα προφίλ.")
        all_generated_profiles.update(w_profiles)
        
    print("\n=== Επίλυση MILP ===")
    prob = pulp.LpProblem("AI_Orchestration_Math_Model", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", all_generated_profiles.keys(), cat='Binary')
    
    # ΝΕΑ ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΣΥΝΑΡΤΗΣΗ: Ελαχιστοποίηση σταθμισμένου αθροίσματος Κόστους και Αδράνειας
    # (Διαιρούμε το T_idle με 1000 για να μετατραπεί σε δευτερόλεπτα, ώστε να είναι στην ίδια κλίμακα με το Κόστος)
    prob += pulp.lpSum(
        x[a, p] * (W_COST * data['C_total'] + W_IDLE * (data['T_idle'] / 1000.0)) 
        for (a, p), data in all_generated_profiles.items()
    ), "Minimize_Cost_And_IdleTime"
    
    # ΠΕΡΙΟΡΙΣΜΟΙ (C1, C2, C3, C4)
    for a in workloads.keys():
        prob += pulp.lpSum(x[w, p] for (w, p) in all_generated_profiles.keys() if w == a) == 1
    for v in network_snapshot.keys():
        prob += pulp.lpSum(x[a, p] * all_generated_profiles[(a, p)]['gpus'].get(v, 0) for (a, p) in all_generated_profiles.keys()) <= network_snapshot[v]['N_gpu']
    for v in network_snapshot.keys():
        prob += pulp.lpSum(x[a, p] * all_generated_profiles[(a, p)]['uplink'].get(v, 0) for (a, p) in all_generated_profiles.keys()) <= network_snapshot[v]['U_avail']
    for v in network_snapshot.keys():
        prob += pulp.lpSum(x[a, p] * all_generated_profiles[(a, p)]['downlink'].get(v, 0) for (a, p) in all_generated_profiles.keys()) <= network_snapshot[v]['D_avail']
        
    prob.solve(pulp.PULP_CBC_CMD(msg=0)) 
    
    print("\n=== ΑΠΟΤΕΛΕΣΜΑΤΑ ===")
    if pulp.LpStatus[prob.status] == 'Optimal':
        total_infrastructure_cost = 0
        total_infrastructure_idle = 0
        
        for (a, p) in all_generated_profiles.keys():
            if x[a, p].varValue == 1.0:
                d = all_generated_profiles[(a, p)]
                print(f"★ Εργασία {a}")
                print(f"   Επιλεγμένο Προφίλ: {p} (Στρατηγική: {d['strategy']})")
                print(f"   Τοποθέτηση       : {d['placement']}")
                print(f"   Οικ. Κόστος AWS  : {d['C_total']:.4f} $")
                print(f"   Χρός. Αδράνειας  : {d['T_idle']:.1f} ms")
                print(f"   Χρόνος Απόκρισης : Total={d['T_total']:.1f} ms | Comp={d['T_comp']:.1f} ms | Comm={d['T_comm']:.1f} ms")
                
                total_infrastructure_cost += d['C_total']
                total_infrastructure_idle += d['T_idle']
                print("-" * 75)
                
        print(f"Συνολικό Οικονομικό Κόστος Υποδομής : {total_infrastructure_cost:.4f} $")
        print(f"Συνολικός Χρόνος Αδράνειας Υποδομής: {total_infrastructure_idle:.1f} ms")
    else:
        print("Infeasible!")