import numpy as np
import copy  # Μεταφέρθηκε εδώ για βέλτιστη απόδοση

class AIOrchestrationEnv:
    def __init__(self, network_snapshot, w_cost=0.5, w_idle=0.5):
        # Αρχική κατάσταση της υποδομής (snapshot)
        self.initial_network = network_snapshot
        
        # Τα βάρη της Πολυκριτηριακής Ανταμοιβής
        self.W_COST = w_cost
        self.W_IDLE = w_idle
        
        # Σταθερές Κανονικοποίησης & Ανταμοιβής
        self.MAX_COST = 0.1500         # Ανώτατο θεωρητικό όριο κόστους ($)
        self.MAX_IDLE_TIME = 250000.0  # ΣΩΣΤΟ: Νέο ανώτατο όριο αδράνειας (ms)
        self.PENALTY_SCALE = 10.0      # Συντελεστής κλίμακας για το penalty score
        self.SUCCESS_BONUS = 15.0      # Σταθερό Bonus επιτυχούς κατανομής
        self.REJECT_PENALTY = -50.0    # Στρατηγική ποινή απόρριψης εργασίας
        
        # Βαριές ποινές (Penalties) για δομικές αποτυχίες κατά το step
        self.INVALID_ACTION_PENALTY = -100.0      # Ποινή αν διαλέξει προφίλ που δεν υπάρχει
        self.RESOURCE_EXHAUSTION_PENALTY = -200.0 # Ποινή αν γεμίσουν οι φυσικοί πόροι

        self.total_initial_gpus = sum(res['N_gpu'] for res in self.initial_network.values())
        self.peak_used_gpus = 0
        
        self.current_state = None
        self.current_workload = None
        self.valid_profiles_for_current = []
        
        self.reset()

    def reset(self):
        """
        Μηδενίζει το περιβάλλον για να ξεκινήσει ένα νέο επεισόδιο εκπαίδευσης.
        Επαναφέρει όλες τις GPUs, τη VRAM και το Bandwidth στο 100%.
        """
        self.current_state = copy.deepcopy(self.initial_network)
        self.peak_used_gpus = 0  # ΔΙΟΡΘΩΣΗ: Μηδενίζεται σε κάθε επεισόδιο/test για σωστά στατιστικά!
        return self._get_state_vector()

    def _get_state_vector(self):
        state_values = []
        # 4 telemetry μετρικές ανά κόμβο
        for node, res in self.current_state.items():
            state_values.extend([res['N_gpu'], res['M_v'], res['U_avail'], res['D_avail']])
            
        if self.current_workload:
            w = self.current_workload
            state_values.extend([w['G_a'], w['D_a'], w['Lambda_a']])
        else:
            state_values.extend([0, 0, 0])
            
        return np.array(state_values, dtype=np.float32)

    def calculate_reward(self, selected_profile):
        """
        Υπολογίζει το Normalized Multi-Objective Reward χρησιμοποιώντας Min-Max κανονικοποίηση
        """
        # 1. Min-Max Normalization στο διάστημα [0, 1] με δικλείδα ασφαλείας min(..., 1.0)
        norm_cost = min(selected_profile['C_total'] / self.MAX_COST, 1.0)
        norm_idle = min(selected_profile['T_idle'] / self.MAX_IDLE_TIME, 1.0)
        
        # 2. Υπολογισμός του συνδυασμένου penalty score (κυμαίνεται από 0.0 έως 10.0)
        penalty_score = self.PENALTY_SCALE * ((self.W_COST * norm_cost) + (self.W_IDLE * norm_idle))
        
        # 3. Το τελικό reward (Success Bonus μείον την κανονικοποιημένη ποινή)
        reward = self.SUCCESS_BONUS - penalty_score
        return reward

    def step(self, action_index):
        """
        Εκτελεί την ενέργεια του Agent, ενημερώνει τους πόρους της υποδομής
        και επιστρέφει τη νέα κατάσταση και την ανταμοιβή.
        """
        # 1. Έλεγχος αν ο Agent διάλεξε προφίλ εκτός ορίων της τρέχουσας εργασίας
        if action_index >= len(self.valid_profiles_for_current):
            return self._get_state_vector(), self.INVALID_ACTION_PENALTY, False, {}
            
        selected_profile = self.valid_profiles_for_current[action_index]
        
        # 2. Έλεγχος διαθεσιμότητας των φυσικών πόρων
        can_allocate = True
        for node, gpus_needed in selected_profile['gpus'].items():
            if self.current_state[node]['N_gpu'] < gpus_needed:
                can_allocate = False
        for node, uplink_needed in selected_profile['uplink'].items():
            if self.current_state[node]['U_avail'] < uplink_needed:
                can_allocate = False
                
        if not can_allocate:
            # Δυναμική Αποτυχία: Το προφίλ δεν χωράει. Ο Agent τιμωρείται, αλλά το επεισόδιο συνεχίζει
            return self._get_state_vector(), self.RESOURCE_EXHAUSTION_PENALTY, False, {}

        # 3. Επιτυχής Δέσμευση πόρων στην υποδομή
        for node, gpus_needed in selected_profile['gpus'].items():
            self.current_state[node]['N_gpu'] -= gpus_needed
            self.current_state[node]['M_v'] -= selected_profile['vram_consumed'][node]
        for node, u_needed in selected_profile['uplink'].items():
            self.current_state[node]['U_avail'] -= u_needed
            self.current_state[node]['D_avail'] -= selected_profile['downlink'][node]
        
        # Καταγραφή της μέγιστης χρήσης πόρων (Peak GPUs)
        current_available_gpus = sum(res['N_gpu'] for res in self.current_state.values())
        current_used = self.total_initial_gpus - current_available_gpus
        if current_used > self.peak_used_gpus:
            self.peak_used_gpus = current_used

        # 4. Υπολογισμός Reward (Success Bonus - Normalized Penalty)
        reward = self.calculate_reward(selected_profile)
        
        done = False 
        return self._get_state_vector(), reward, done, {"profile_used": selected_profile['strategy']}
