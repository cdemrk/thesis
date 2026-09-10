import json
import random

# Σταθεροποιούμε το seed για να έχουμε επαναλαβήσιμα αποτελέσματα
random.seed(42)

def generate_infrastructure():
    """
    Δημιουργεί μια οριακή, ετερογενή υποδομή (5 Edge + 3 Cloud).
    """
    network = {}
    
    # 1. Δημιουργία 5 Edge Κόμβων
    for i in range(1, 6):
        node_name = f"Edge_{i}"
        network[node_name] = {
            "N_gpu": random.choice([2, 4]),          
            "U_avail": random.randint(500, 1000),    
            "D_avail": random.randint(500, 1000),    
            "F_v": round(random.uniform(5.0, 15.0), 2), 
            "M_v": random.choice([12, 16]),          
            "coords_v": (round(random.uniform(0, 100), 1), round(random.uniform(0, 100), 1))
        }

    # 2. Δημιουργία 3 Cloud Κόμβων
    for i in range(1, 4):
        node_name = f"Cloud_{i}"
        network[node_name] = {
            "N_gpu": 12,                             
            "U_avail": 5000,                         
            "D_avail": 5000,
            "F_v": round(random.uniform(60.0, 100.0), 2), 
            "M_v": 80,                               
            "coords_v": (round(random.uniform(800, 1000), 1), round(random.uniform(800, 1000), 1))
        }
        
    return network


def generate_workloads_by_capacity(target_gpu_demand):
    """
    Παράγει workloads μέχρι η συνολική τους ζήτηση σε GPUs (n_a * g_a)
    να φτάσει (ή να ξεπεράσει ελάχιστα) το target_gpu_demand.
    """
    workloads = {}
    current_gpu_demand = 0
    i = 1
    
    while current_gpu_demand < target_gpu_demand:
        w_id = f"W_{i}"
        type_a = random.choice([0, 1])  # 0: Inference, 1: Training
        
        if type_a == 0:
            n_a = random.choice([1, 2])
            g_a = 1
            d_a = round(random.uniform(10.0, 50.0), 2)
            a_a = round(random.uniform(5.0, 20.0), 2)
            l = 1
            m = 1
            k_a = 1
            g_flops = round(random.uniform(50.0, 200.0), 2)
            lambda_a = round(random.uniform(10000.0, 30000.0), 2) 
        else:
            n_a = random.randint(2, 4) 
            g_a = random.choice([1, 2])
            d_a = round(random.uniform(100.0, 400.0), 2)
            a_a = round(random.uniform(5.0, 150.0), 2)
            l = random.randint(4, 10)
            m = random.randint(2, 6)
            k_a = random.randint(10, 25)  
            g_flops = round(random.uniform(500.0, 1500.0), 2)
            lambda_a = round(random.uniform(5000000.0, 15000000.0), 2) 

        # Υπολογισμός της ζήτησης αυτού του workload
        workload_demand = n_a * g_a
        
        # Αν η προσθήκη αυτού του workload μας πάει ΠΟΛΥ πάνω από το target, το κάνουμε skip
        if current_gpu_demand + workload_demand > target_gpu_demand + 2:
            continue

        workloads[w_id] = {
            "type_a": type_a,
            "n_a": n_a,
            "g_a": g_a,
            "D_a": d_a,
            "A_a": a_a,
            "L": l,
            "m": m,
            "K_a": k_a,
            "G_a": g_flops,
            "Lambda_a": lambda_a
        }
        
        current_gpu_demand += workload_demand
        i += 1
        
    return workloads, current_gpu_demand

if __name__ == "__main__":
    print("=== Δημιουργία Οριακής Υποδομής (5 Edge + 3 Cloud) ===")
    common_network = generate_infrastructure()
    total_gpus = sum(res['N_gpu'] for res in common_network.values())
    print(f"Δημιουργήθηκαν {len(common_network)} κόμβοι με συνολική χωρητικότητα: {total_gpus} GPUs.\n")

    # Ορίζουμε τα 4 σενάρια χωρητικότητας
    scenarios = [
        {"name": "dataset_1_abundance.json", "ratio": 0.50, "desc": "Χωράει Άνετα (50% Demand)"},
        {"name": "dataset_2_marginal.json", "ratio": 0.95, "desc": "Χωράει Οριακά (95% Demand)"},
        {"name": "dataset_3_mild_contention.json", "ratio": 1.20, "desc": "Δεν Χωράει Οριακά (120% Demand)"},
        {"name": "dataset_4_heavy_contention.json", "ratio": 2.50, "desc": "Δεν Χωράει Κατά Πολύ (250% Demand)"}
    ]

    for scenario in scenarios:
        target_demand = int(total_gpus * scenario["ratio"])
        print(f"Παραγωγή {scenario['name']} | Σενάριο: {scenario['desc']}")
        print(f"-> Στόχος ζήτησης: ~{target_demand} GPUs")
        
        workloads, actual_demand = generate_workloads_by_capacity(target_demand)
        
        print(f"-> Δημιουργήθηκαν {len(workloads)} εργασίες με πραγματική ζήτηση: {actual_demand} GPUs.\n")
        
        with open(scenario["name"], "w", encoding="utf-8") as f:
            json.dump({"network": common_network, "workloads": workloads}, f, indent=4)

    print("✅ Όλα τα αρχεία δημιουργήθηκαν!")
