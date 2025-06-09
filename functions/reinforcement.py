import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from box import Box
from .utils import open_process

class trainer:
    def __init__(self,args_path):
        with open(args_path,'r') as f:
            args = Box.from_yaml(f.read())
        
        # Parámetros generales
        self.csv_file     = args.general.csv_file
        self.n_tracks     = args.general.n_tracks
        self.valence_goal = args.general.valence_goal
        self.episodes     = args.general.episodes
        self.max_steps    = args.general.max_steps
        self.gamma        = args.general.gamma
        self.alpha        = args.general.alpha
        self.eps_start    = args.general.eps_start
        self.eps_min      = args.general.eps_min
        self.eps_decay    = args.general.eps_decay
        self.seed         = args.general.seed

        # Pesos del reward
        self.w_val = args.weights.w_val
        self.w_eng = args.weights.w_eng
        self.w_dnc = args.weights.w_dnc
        self.w_pop = args.weights.w_pop
        self.w_sim = args.weights.w_sim

        # Inicialización reproducible
        np.random.seed(self.seed)

        # Dataset y matriz de similitud
        self.df, self.sim_mat = open_process(self.csv_file,self.seed,self.n_tracks)
        self.df["happy"] = self.df["valence_n"] >= self.valence_goal

    def choose_action(self,Q_row, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(Q_row.size)
        return int(np.argmax(Q_row))

    def reward(self, prev_idx: int, act_idx: int) -> float:
        """Adaptive reward combining valence, arousal proxies and similarity."""
        r = (self.w_val * self.df.at[act_idx, "valence_n"] +
            self.w_eng * self.df.at[act_idx, "energy_n"] +
            self.w_dnc * self.df.at[act_idx, "danceability_n"] +
            self.w_pop * self.df.at[act_idx, "popularity_n"])
        if prev_idx != -1:
            r += self.w_sim * self.sim_mat[prev_idx, act_idx]
        return r

    def reinforce(self,politic_fn):
        self.Q   = np.zeros((self.n_tracks, self.n_tracks), dtype=np.float32)
        self.NSA = np.ones_like(self.Q, dtype=np.int32)  # visit counts for dynamic Q
        steps_ep = []
        epsilon = self.eps_start

        for ep in trange(self.episodes, desc=f"{politic_fn.__name__.upper():>5}", ncols=70, leave=False):
            s1 = -1  # start with no previous song
            for step in range(1, self.max_steps + 1):
                a1       = self.choose_action(self.Q[s1], epsilon)
                r        = self.reward(s1, a1)
                s2       = a1 # ???
                a2       = self.choose_action(self.Q[s2],epsilon)

                politic_fn(self,s1,a1,r,s2,a2)

                # Goal reached?
                if self.df.at[a1, "happy"]:
                    steps_ep.append(step)
                    break

                s1 = s2
            else:
                steps_ep.append(self.max_steps)  # never reached within limit
            epsilon = max(self.eps_min, epsilon * self.eps_decay)

        return steps_ep
    
# -------- Politics ---------
def SARSA(trainer,S1,A1,R,S2,A2):
    trainer.Q[S1,A1] += trainer.alpha * ((R + trainer.gamma * trainer.Q[S2,A2]) - trainer.Q[S1,A1])

def Q_LEARNING(trainer,S1,A1,R,S2,A2=None):
    trainer.Q[S1,A1] += trainer.alpha * ((R + trainer.gamma * np.max(trainer.Q[S2])) - trainer.Q[S1,A1])

def DYNAQ(trainer,S1,A1,R,S2,A2=None):
    trainer.alpha = (1.0/trainer.NSA[S1,A1])
    trainer.Q[S1,A1] += trainer.alpha * ((R + trainer.gamma * np.max(trainer.Q[S2])) - trainer.Q[S1,A1])
    trainer.NSA[S1,A1] += 1

# -------- Graphic design ---------
def plot_steps(title,steps):
    plt.figure()
    plt.plot(steps)
    plt.title(f"Pasos para alcanzar canción feliz\n{title}")
    plt.xlabel("Episodio")
    plt.ylabel("Pasos (≤ 30)")
    plt.ylim(0, len(steps) + 1)
    plt.tight_layout()
    plt.show()