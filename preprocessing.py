import math
import itertools

# ==========================================
# ΣΤΑΘΕΡΕΣ ΚΑΙ ΠΑΡΑΜΕΤΡΟΙ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑΣ
# ==========================================
TIERS = [1, 2, 4, 8]        
UNIT_SPEED_MBPS = 50        

EDGE_GPU_COST_PER_SEC = 0.002   
CLOUD_GPU_COST_PER_SEC = 0.0003  

def get_max_tier_for_data(data_size):
    if data_size < 2: return 1      
    elif data_size < 10: return 2   
    elif data_size < 50: return 4   
    else: return 8                

def calc_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def create_profile_table(workload_id, workload, network_state):
    valid_profiles = {}
    
    n_a = workload['n_a']           
    g_a = workload['g_a']           
    N_total = n_a * g_a             
    Lambda_a = workload['Lambda_a'] 
    D_a = workload['D_a']           
    A_a = workload.get('A_a', 0)    
    L = workload.get('L', 1)        
    m = workload.get('m', 1)        
    G_a = workload['G_a']           
    type_a = workload['type_a']     
    K_a = workload.get('K_a', 1)    
    
    if type_a == 1: 
        if n_a == 1: allowed_strategies = ['Single']
        else: allowed_strategies = ['PS', 'AR', 'TP', 'PP']
    else:           
        if n_a == 1: allowed_strategies = ['Single']
        else: allowed_strategies = ['TP', 'PP'] 
    
    max_tier_data = get_max_tier_for_data(max(D_a, A_a))
    allowed_tiers = [b for b in TIERS if b <= max_tier_data]
    
    physical_nodes = list(network_state.keys())
    
    # Βρίσκουμε ΟΛΟΥΣ τους κόμβους (Edge & Cloud) που έχουν αρκετές GPUs ελεύθερες
    available_nodes = [v for v, res in network_state.items() if res['N_gpu'] >= g_a]
    
    # --- ΕΞΑΝΤΛΗΤΙΚΗ ΕΥΡΕΣΗ ΣΥΝΔΥΑΣΜΩΝ ---
    placements = []
    if len(available_nodes) >= n_a:
        placements = list(itertools.combinations(available_nodes, n_a))
                
    if not placements:
        return {}
        
    profile_counter = 1
    
    for placement in placements:
        # Ορίζουμε το σύνολο των μοναδικών φυσικών κόμβων για αυτό το placement
        unique_nodes = set(placement)
        
        # --- ΦΙΛΤΡΟ 1: Διαθεσιμότητα GPUs και Δυναμικής VRAM ανά Κόμβο ---
        gpu_needs = {v: 0 for v in physical_nodes}
        vram_needs = {v: 0.0 for v in physical_nodes} # Λεξικό για την καταγραφή μνήμης
        
        for phys_n in placement:
            gpu_needs[phys_n] += g_a 
            
        valid_hw = True
        for v, req_gpus in gpu_needs.items():
            if req_gpus > network_state[v]['N_gpu']:
                valid_hw = False
                break
            
            if req_gpus > 0:
                # ΔΥΝΑΜΙΚΟΣ ΥΠΟΛΟΓΙΣΜΟΣ VRAM βάσει των πραγματικών αναγκών της εργασίας
                vram_per_gpu = max(2.0, (D_a + A_a) / 100.0) # Emergency κατώτατο όριο 2GB
                vram_required = req_gpus * vram_per_gpu
                vram_needs[v] = vram_required # Αποθήκευση για το environment
                
                if vram_required > network_state[v]['M_v']:
                    valid_hw = False
                    break
                
        if not valid_hw: 
            continue
            
        # Υπολογισμός υπολογιστικής ισχύος
        f_v_list = [network_state[v]['F_v'] for v in placement]
        sum_f_v = sum(g_a * f for f in f_v_list)
        min_f_v = min(f_v_list)
        
        # Υπολογισμός δικτυακής απόστασης (Propagation Delay)
        max_inter_dist = 0
        if len(unique_nodes) > 1:
            max_inter_dist = max(calc_distance(network_state[u]['coords_v'], network_state[v]['coords_v']) 
                                 for u in unique_nodes for v in unique_nodes)
        t_prop_inter = max_inter_dist * 0.1 
        
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

            # --- ΥΠΟΛΟΓΙΣΜΟΣ T_comm ---
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
                
                # --- ΦΙΛΤΡΟ 2: Bandwidth ---
                valid_network = True
                for v in physical_nodes:
                    if uplink_needs[v] > network_state[v]['U_avail'] or downlink_needs[v] > network_state[v]['D_avail']:
                        valid_network = False
                        break
                if not valid_network: 
                    continue 
                    
                # --- ΦΙΛΤΡΟ 3: SLA ---
                t_access = 5.0 
                t_total = t_comp + t_access + t_comm
                
                if t_total <= Lambda_a:
                    ideal_t_comp = (G_a / sum_f_v) * 1000 
                    t_idle = max(0, t_total - ideal_t_comp) 
                    
                    cost_total = 0
                    for node, gpus_allocated in gpu_needs.items():
                        if 'Cloud' in node:
                            cost_total += gpus_allocated * CLOUD_GPU_COST_PER_SEC * (t_comp / 1000.0)
                        elif 'Edge' in node:
                            cost_total += gpus_allocated * EDGE_GPU_COST_PER_SEC * (t_comp / 1000.0)

                    profile_id = f"P{profile_counter}_{strategy}" 
                    valid_profiles[profile_id] = {
                        'strategy': strategy,
                        'placement': placement,
                        'b_req': b,
                        'T_total': t_total,
                        'T_comp': t_comp,
                        'T_comm': t_comm,
                        'T_idle': t_idle,       
                        'C_total': cost_total,   
                        'gpus': gpu_needs,
                        'vram_consumed': vram_needs, # <-- Προσθήκη για να αφαιρείται η μνήμη στο Environment!
                        'uplink': uplink_needs,
                        'downlink': downlink_needs
                    }
                    profile_counter += 1
                
    return valid_profiles