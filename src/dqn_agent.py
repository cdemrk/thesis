import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ==========================================
# 1. ΑΡΧΙΤΕΚΤΟΝΙΚΗ ΝΕΥΡΩΝΙΚΟΥ ΔΙΚΤΥΟΥ
# ==========================================
class MultiObjectiveDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiObjectiveDQN, self).__init__()
        
        # Απλό και αποδοτικό Feed-Forward δίκτυο για γρήγορο inference
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) # Έξοδος: Q-value για κάθε πιθανό Profile Index
        )
        
    def forward(self, x):
        return self.network(x)


# Υπο-παραμετροποιημένο
class ShallowDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ShallowDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


# Υπερ-παραμετροποιημένο
class DeepDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128) # Προσθήκη έξτρα επιπέδου
        self.out = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.out(x)


# ==========================================
# 2. Ο DRL AGENT (ΔΙΑΧΕΙΡΙΣΗ ΜΑΘΗΣΗΣ & REPLAY BUFFER)
# ==========================================
class DQNAgent:
    # ---> ΑΛΛΑΓΗ 1: Προσθήκη του net_type στα ορίσματα
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, net_type="proposed"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma # Συντελεστής έκπτωσης για το μελλοντικό lookahead
        
        # Καθορισμός Συσκευής: Αυτόματα βρίσκει τη GPU του εργαστηρίου (αν υπάρχει), αλλιώς CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Ο DRL Agent αρχικοποιήθηκε. Συσκευή εκτέλεσης: {self.device}")
        
        # ---> ΑΛΛΑΓΗ 2: Δυναμική επιλογή αρχιτεκτονικής δικτύου
        if net_type == "shallow":
            self.policy_net = ShallowDQN(input_dim, output_dim).to(self.device)
            self.target_net = ShallowDQN(input_dim, output_dim).to(self.device)
            print("🧠 Αρχιτεκτονική: Shallow Network (64x64)")
        elif net_type == "deep":
            self.policy_net = DeepDQN(input_dim, output_dim).to(self.device)
            self.target_net = DeepDQN(input_dim, output_dim).to(self.device)
            print("🧠 Αρχιτεκτονική: Deep Network (128x128x128)")
        else: # Default το "proposed"
            self.policy_net = MultiObjectiveDQN(input_dim, output_dim).to(self.device)
            self.target_net = MultiObjectiveDQN(input_dim, output_dim).to(self.device)
            print("🧠 Αρχιτεκτονική: Proposed Network (128x128)")
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay Buffer: Μνήμη που αποθηκεύει εμπειρίες για την εκπαίδευση
        self.memory = deque(maxlen=10000)
        
        # --- ΣΩΣΤΗ ΡΥΘΜΙΣΗ EPSILON ΓΙΑ PER-EPISODE DECAY ---
        self.epsilon = 1.0           # Ξεκινάει με 100% εξερεύνηση
        self.epsilon_min = 0.05      # Ελάχιστη εξερεύνηση 5%
        self.epsilon_decay = 0.995   # Μείωση κατά % σε ΚΑΘΕ ΕΠΕΙΣΟΔΙΟ

    def remember(self, state, action, reward, next_state, done):
        """Αποθηκεύει μια μετάβαση στη μνήμη του Agent."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, valid_profiles_count):
        """
        Επιλέγει ενέργεια με βάση την πολιτική ε-greedy.
        Διασφαλίζει ότι ο Agent διαλέγει ΜΟΝΟ από τα έγκυρα προφίλ.
        """
        if valid_profiles_count == 0:
            return 0 # Αν δεν υπάρχει κανένα προφίλ, επιστρέφει default 0
            
        # Exploration: Τυχαία επιλογή
        if random.random() < self.epsilon:
            return random.randint(0, valid_profiles_count - 1)
            
        # Exploitation: Επιλογή του καλύτερου προφίλ βάσει Νευρωνικού
        # Μετατροπή του state σε Tensor και αποστολή στη σωστή συσκευή (GPU/CPU)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
        # Φιλτράρισμα (Masking): Κρατάμε μόνο τα Q-values των έγκυρων προφίλ
        valid_q_values = q_values[:valid_profiles_count]
        return int(np.argmax(valid_q_values))

    def train_step(self, batch_size=64):
        """
        Η ΚΑΡΔΙΑ ΤΟΥ 1-STEP LOOKAHEAD:
        Παίρνει ένα δείγμα από τη μνήμη και εκπαιδεύει το δίκτυο στη GPU.
        """
        if len(self.memory) < batch_size:
            return
            
        # Τυχαία δειγματοληψία από το Replay Buffer
        mini_batch = random.sample(self.memory, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        # Μετατροπή λιστών σε Tensors και αποστολή στη GPU για γρήγορους υπολογισμούς
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 1. Τρέχουσες προβλέψεις του δικτύου: Q(s_t, a_t)
        current_q = self.policy_net(states).gather(1, actions)
        
        # 2. Το 1-step lookahead: Υπολογισμός της μέγιστης μελλοντικής αξίας Q(s_{t+1}, a)
        with torch.no_grad():
            next_max_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Bellman Equation: Target = r + gamma * max Q(s', a)
            target_q = rewards + (1 - dones) * self.gamma * next_max_q
            
        # 3. Υπολογισμός του σφάλματος (Loss) και Backpropagation
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Ανανεώνει το Target δίκτυο αντιγράφοντας τα βάρη του Policy δικτύου."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Καλείται ΑΠΟΚΛΕΙΣΤΙΚΑ από το train_agent.py στο ΤΕΛΟΣ κάθε επεισοδίου
        για να μειώσει την τυχαιότητα ομαλά (Per-Episode Decay).
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)