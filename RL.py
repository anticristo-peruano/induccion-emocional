import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from collections import defaultdict, Counter
import time
import os
import warnings
warnings.filterwarnings('ignore')
from functions.environment.camera import thread_episode

def filter_tracks(df):
    folder_path = os.path.join(os.curdir,'functions','environment','.songs')
    existing_ids = set(
        os.path.splitext(fname)[0]
        for fname in os.listdir(folder_path)
        if fname.endswith('.mp3')
    )
    return df[df['track_id'].isin(existing_ids)].copy()

class MusicEnvironment:
    def __init__(self, dataset_path=None):
        # Definir columnas de características primero
        self.feature_cols = ['valence', 'energy', 'danceability', 
                             'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        
        if dataset_path and self._load_dataset(dataset_path):
            print(f"Dataset cargado: {len(self.df)} canciones")
        else:
            self._create_dummy_data()
            print("Usando datos de prueba")
        
        self._setup_states()
        self._setup_actions()
        self._initialize_profile()
        self._prepare_features()
        
        self.current_state = random.randint(0, 15)
        self.target_state = 15  # Alta valence, alta energy
        self.noise_level = 0.15
        self.personal_bias = np.random.random(7) * 0.3
        
        # Historial detallado
        self.detailed_history = []
    
    def _load_dataset(self, path):
        try:
            self.df = filter_tracks(pd.read_csv(path))
            required_cols = ['track_id','track_name','artists','duration_ms',
                             'valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
            
            if not all(col in self.df.columns for col in required_cols):
                return False
            
            self.df = self.df.dropna(subset=['valence', 'energy']).reset_index(drop=True)
            return True
        except:
            return False
    
    def _setup_states(self):
        # Grid 4x4 para valence-arousal
        self.state_map = {}
        state_id = 0
        cent = np.linspace(start=-0.75,stop=0.75,num=4)
        self.cents = []
        
        for i in range(4):  # arousal levels
            for j in range(4):  # valence levels
                self.state_map[state_id] = {
                    'valence': float(cent[j]),
                    'arousal': float(cent[i]),
                    'name': f"E{i+1}V{j+1}"
                }
                self.cents.append([float(cent[j]),float(cent[i])])
                state_id += 1
        
        self.cents = np.array(self.cents)

        print("Grid de estados configurado:")
        for i in range(3, -1, -1):
            row = f"E{i+1}: "
            for j in range(4):
                state_id = i * 4 + j
                row += f"{state_id:2d} "
            print(row)
    
    def _setup_actions(self):
        features = ['danceability', 'energy', 'valence', 'acousticness', 
                   'instrumentalness', 'liveness', 'speechiness']
        
        self.actions = []
        for feature in features:
            self.actions.append(f"increase_{feature}")
            self.actions.append(f"decrease_{feature}")
        
        print(f"Acciones configuradas ({len(self.actions)}):")
        for i, action in enumerate(self.actions):
            print(f"  {i:2d}: {action}")
    
    def _initialize_profile(self):
        self.target_profile = {}
        
        print("\nPerfil inicial (promedios del dataset):")
        for feature in self.feature_cols:
            if feature in self.df.columns:
                self.target_profile[feature] = self.df[feature].mean()
            else:
                self.target_profile[feature] = 0.5
            print(f"  {feature:15}: {self.target_profile[feature]:.3f}")
    
    def _prepare_features(self):
        # Crear DataFrame con nombres de columnas para evitar warning
        feature_df = self.df[self.feature_cols].copy()
        
        self.scaler = StandardScaler()
        self.normalized_features = self.scaler.fit_transform(feature_df)
    
    def _coords_to_state(self, ep_vec):
        return int(np.argmin(np.linalg.norm(self.cents - ep_vec,ord=2,axis=1)))
    
    def apply_action(self, action_id):
        if action_id < 0 or action_id >= len(self.actions):
            return
        
        action = self.actions[action_id]
        old_value = None
        
        if action.startswith("increase_"):
            feature = action.replace("increase_", "")
            delta = 0.1
        elif action.startswith("decrease_"):
            feature = action.replace("decrease_", "")
            delta = -0.1
        else:
            return
        
        if feature in self.target_profile:
            old_value = self.target_profile[feature]
            new_value = self.target_profile[feature] + delta
            self.target_profile[feature] = np.clip(new_value, 0.0, 1.0)
            
        return {
            'feature': feature,
            'old_value': old_value,
            'new_value': self.target_profile[feature],
            'delta': delta
        }
    
    def select_song(self):
        # Crear DataFrame con nombres para evitar warning
        profile_df = pd.DataFrame([self.target_profile])
        profile_normalized = self.scaler.transform(profile_df)
        
        similarities = cosine_similarity(profile_normalized, self.normalized_features)
        best_idx = np.argmax(similarities[0])
        
        return self.df.iloc[best_idx].copy(), similarities[0][best_idx]
    
    def step(self, action_id):
        action_info = self.apply_action(action_id)
        song, similarity = self.select_song()
        songid, duration_ms, name, artists = song['track_id'], song['duration_ms'], song['track_name'], song['artists']
        print(f'Reproducing: "{name}" by {artists}.')

        response_vec, response_details = thread_episode(songid,duration_ms,10)
        new_state = self._coords_to_state(response_vec)
        
        # Recompensa binaria
        reward = 1.0 if new_state == self.target_state else 0.0
        
        prev_state = self.current_state
        self.current_state = new_state
        print(f'{prev_state} --> {new_state} (Valence: {round(response_vec[0],2)}, Arousal: {round(response_vec[1],2)}). Reward: {reward}.')
        
        # Registrar en historial detallado
        step_info = {
            'prev_state': prev_state,
            'prev_state_name': self.state_map[prev_state]['name'],
            'action_id': action_id,
            'action_name': self.actions[action_id],
            'action_info': action_info,
            'song_selected': {
                'name': song['track_name'],
                'artist': song['artists'],
                'valence': song['valence'],
                'energy': song['energy'],
                'similarity': similarity
            },
            'response_details': response_details,
            'response_valence': response_vec[0],
            'response_energy': response_vec[1],
            'new_state': new_state,
            'new_state_name': self.state_map[new_state]['name'],
            'reward': reward,
            'target_profile': self.target_profile.copy()
        }
        
        self.detailed_history.append(step_info)
        
        info = {
            'song': song,
            'similarity': similarity,
            'response_valence': response_vec[0],
            'response_energy': response_vec[1],
            'prev_state': prev_state,
            'step_info': step_info
        }
        
        return new_state, reward, info

class BaseAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, epsilon=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        self.rewards = []
        self.states = []
        self.episodes = 0
        self.detailed_log = []
    
    def choose_action(self, state, q_values):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(q_values[state])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_stats(self, reward, state, episode_info=None):
        self.rewards.append(reward)
        self.states.append(state)
        self.episodes += 1
        if episode_info:
            self.detailed_log.append(episode_info)

class QLearningAgent(BaseAgent):
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, epsilon=1.0):
        super().__init__(n_states, n_actions, lr, gamma, epsilon)
        self.q_table = np.zeros((n_states, n_actions))
        self.name = "Q-Learning"
    
    def choose_action(self, state):
        return super().choose_action(state, self.q_table)
    
    def update(self, state, action, reward, next_state, episode_info=None):
        best_next_q = np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next_q
        old_q = self.q_table[state][action]
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        
        if episode_info:
            episode_info['q_learning_details'] = {
                'old_q_value': old_q,
                'new_q_value': self.q_table[state][action],
                'target': target,
                'best_next_q': best_next_q
            }
        
        self.save_stats(reward, next_state, episode_info)

class SARSAAgent(BaseAgent):
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, epsilon=1.0):
        super().__init__(n_states, n_actions, lr, gamma, epsilon)
        self.q_table = np.zeros((n_states, n_actions))
        self.name = "SARSA"
    
    def choose_action(self, state):
        return super().choose_action(state, self.q_table)
    
    def update(self, state, action, reward, next_state, next_action, episode_info=None):
        if next_action is not None:
            next_q = self.q_table[next_state][next_action]
        else:
            next_q = np.max(self.q_table[next_state])
        
        target = reward + self.gamma * next_q
        old_q = self.q_table[state][action]
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        
        if episode_info:
            episode_info['sarsa_details'] = {
                'old_q_value': old_q,
                'new_q_value': self.q_table[state][action],
                'target': target,
                'next_q': next_q,
                'next_action': next_action
            }
        
        self.save_stats(reward, next_state, episode_info)

class DynaQAgent(BaseAgent):
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, epsilon=1.0, n_planning=5):
        super().__init__(n_states, n_actions, lr, gamma, epsilon)
        self.q_table = np.zeros((n_states, n_actions))
        self.n_planning = n_planning
        self.name = f"Dyna-Q"
        
        self.model = {}
        self.experiences = []
    
    def choose_action(self, state):
        return super().choose_action(state, self.q_table)
    
    def update(self, state, action, reward, next_state, episode_info=None):
        # Actualizacion directa
        best_next_q = np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next_q
        old_q = self.q_table[state][action]
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
        
        # Actualizar modelo
        self.model[(state, action)] = (next_state, reward)
        self.experiences.append((state, action, reward, next_state))
        
        # Planning
        planning_updates = []
        for _ in range(self.n_planning):
            if self.experiences:
                s, a, r, s_next = random.choice(self.experiences)
                best_q = np.max(self.q_table[s_next])
                target_sim = r + self.gamma * best_q
                old_q_sim = self.q_table[s][a]
                self.q_table[s][a] += self.lr * (target_sim - self.q_table[s][a])
                planning_updates.append({
                    'state': s, 'action': a, 'reward': r, 'next_state': s_next,
                    'old_q': old_q_sim, 'new_q': self.q_table[s][a]
                })
        
        if episode_info:
            episode_info['dynaq_details'] = {
                'direct_update': {
                    'old_q_value': old_q,
                    'new_q_value': self.q_table[state][action],
                    'target': target,
                    'best_next_q': best_next_q
                },
                'planning_updates': planning_updates,
                'model_size': len(self.model),
                'experiences_count': len(self.experiences)
            }
        
        self.save_stats(reward, next_state, episode_info)

class MusicRLComparison:
    def __init__(self, dataset_path=None):
        self.env = MusicEnvironment(dataset_path)
        
        self.agents = {
            'Q-Learning': QLearningAgent(16, len(self.env.actions)),
            'SARSA': SARSAAgent(16, len(self.env.actions)),
            'Dyna-Q': DynaQAgent(16, len(self.env.actions), n_planning=3)
        }
        
        self.results = {name: [] for name in self.agents.keys()}
        
        print(f"\nEntorno inicializado con {len(self.env.actions)} acciones")
        print(f"Estado objetivo: {self.env.target_state} ({self.env.state_map[self.env.target_state]['name']})")
    
    def run_episode(self, agent, agent_name):   # Correr episodio
        self.env.current_state = random.randint(0, 15)
        state = self.env.current_state
        
        action = agent.choose_action(state)
        next_state, reward, info = self.env.step(action)
        
        # Crear info detallada del episodio
        episode_info = {
            'agent_name': agent_name,
            'episode_number': agent.episodes + 1,
            'initial_state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'epsilon': agent.epsilon,
            'step_details': info['step_info']
        }
        
        if agent_name == 'SARSA':
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, episode_info)
        else:
            agent.update(state, action, reward, next_state, episode_info)
        
        agent.decay_epsilon()
        
        return {
            'initial_state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'song': info['song']['track_name'],
            'artist': info['song']['artists'],
            'epsilon': agent.epsilon,
            'info': info,
            'episode_info': episode_info
        }
    
    def run_sequential_comparison(self, n_episodes=100, verbose=True):
        print(f"\nEjecutando {n_episodes} episodios SECUENCIALES para cada algoritmo")
        print("=" * 80)
        
        for agent_name, agent in self.agents.items():
            print(f"\n--- EJECUTANDO {agent_name} ({n_episodes} episodios) ---")
            start_time = time.time()
            
            for episode in range(n_episodes):
                # Backup del estado para consistencia
                env_backup = self.env.current_state
                profile_backup = self.env.target_profile.copy()
                
                result = self.run_episode(agent, agent_name)
                self.results[agent_name].append(result)
                print(f"Action: {self.env.actions[result['action']][:15]:15}\n")
                
                if verbose and (episode < 5 or (episode + 1) % 20 == 0): # 
                    success = "SUCCESS" if result['reward'] == 1.0 else "FAIL"
                    print(f"  Ep {episode + 1:3d}: {success:7} "
                          f"{self.env.state_map[result['initial_state']]['name']} -> "
                          f"{self.env.state_map[result['next_state']]['name']} "
                          f"Action: {self.env.actions[result['action']][:15]:15} "
                          f"R:{result['reward']:.0f} ε:{result['epsilon']:.3f}")
                
            # Restaurar estado
                self.env.current_state = env_backup
                self.env.target_profile = profile_backup
            
            elapsed = time.time() - start_time
            success_rate = np.mean(agent.rewards) * 100
            print(f"  Completado en {elapsed:.2f}s - Tasa de éxito: {success_rate:.1f}%")
        
        self.show_results()
        self.plot_learning_curves()
    
    def show_results(self):
        print("\n" + "=" * 80)
        print("RESULTADOS FINALES DETALLADOS")
        print("=" * 80)
        
        results_summary = []
        
        for agent_name, agent in self.agents.items():
            rewards = agent.rewards
            states = agent.states
            
            if rewards:
                success_rate = np.mean(rewards) * 100
                total_successes = np.sum(rewards)
                
                # Analisis por ventanas
                first_quarter = rewards[:len(rewards)//4]
                last_quarter = rewards[len(rewards)//4*3:]
                
                first_rate = np.mean(first_quarter) * 100 if first_quarter else 0
                last_rate = np.mean(last_quarter) * 100 if last_quarter else 0
                improvement = last_rate - first_rate
                
                print(f"\n{agent_name}:")
                print(f"  Episodios totales: {len(rewards)}")
                print(f"  Tasa de éxito global: {success_rate:.1f}% ({total_successes:.0f}/{len(rewards)})")
                print(f"  Primer cuarto: {first_rate:.1f}%")
                print(f"  Último cuarto: {last_rate:.1f}%")
                print(f"  Mejora: {improvement:+.1f}%")
                print(f"  Epsilon final: {agent.epsilon:.4f}")
                
                # Estados más visitados
                state_counts = Counter(states)
                print(f"  Estados más visitados:")
                for state_id, count in state_counts.most_common(3):
                    state_name = self.env.state_map[state_id]['name']
                    percentage = count / len(states) * 100
                    print(f"    {state_id:2d} ({state_name}): {count:3d} veces ({percentage:.1f}%)")
                
                results_summary.append((agent_name, success_rate, last_rate))
        
        # Ranking
        results_summary.sort(key=lambda x: x[2], reverse=True)  # Por último cuarto
        print(f"\nRANKING (por último cuarto):")
        for i, (name, global_rate, last_rate) in enumerate(results_summary):
            rank = ["1ro", "2do", "3ro"][i] if i < 3 else f"{i+1}to"
            print(f"  {rank}: {name:10} - {last_rate:.1f}% (global: {global_rate:.1f}%)")
    
    def plot_learning_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Curvas de recompensa
        ax1 = axes[0, 0]
        for agent_name, agent in self.agents.items():
            rewards = agent.rewards
            episodes = range(1, len(rewards) + 1)
            ax1.plot(episodes, rewards, alpha=0.3, label=f'{agent_name} (raw)')
            
            # Suavizado con ventana móvil
            window_size = max(5, len(rewards) // 20)
            if len(rewards) >= window_size:
                smoothed = []
                for i in range(len(rewards)):
                    start = max(0, i - window_size + 1)
                    smoothed.append(np.mean(rewards[start:i+1]))
                ax1.plot(episodes, smoothed, linewidth=2, label=f'{agent_name} (suavizado)')
        
        ax1.set_title('Recompensas por Episodio')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Recompensa')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tasa de éxito acumulada
        ax2 = axes[0, 1]
        for agent_name, agent in self.agents.items():
            rewards = agent.rewards
            cumulative_success = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            ax2.plot(range(1, len(rewards) + 1), cumulative_success, linewidth=2, label=agent_name)
        
        ax2.set_title('Tasa de Éxito Acumulada')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Tasa de Éxito')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax3 = axes[1, 0]
        for agent_name, agent in self.agents.items():
            epsilons = [log['epsilon'] for log in agent.detailed_log]
            ax3.plot(range(1, len(epsilons) + 1), epsilons, linewidth=2, label=agent_name)
        
        ax3.set_title('Decaimiento de Epsilon')
        ax3.set_xlabel('Episodio')
        ax3.set_ylabel('Epsilon')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Distribución de estados visitados
        ax4 = axes[1, 1]
        state_data = []
        labels = []
        
        for agent_name, agent in self.agents.items():
            state_counts = Counter(agent.states)
            # Solo mostrar estados más visitados
            top_states = state_counts.most_common(5)
            counts = [count for _, count in top_states]
            state_data.append(counts)
            if not labels:  # Solo agregar labels una vez
                labels = [f"Estado {state}" for state, _ in top_states]
        
        x = np.arange(len(labels))
        width = 0.25
        
        for i, (agent_name, data) in enumerate(zip(self.agents.keys(), state_data)):
            ax4.bar(x + i * width, data, width, label=agent_name)
        
        ax4.set_title('Estados Más Visitados')
        ax4.set_xlabel('Estados')
        ax4.set_ylabel('Frecuencia')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_detailed_history(self, agent_name, n_episodes=5):
        """Muestra historial detallado de los últimos n episodios de un agente"""
        print(f"\nHISTORIAL DETALLADO - {agent_name} (últimos {n_episodes} episodios)")
        print("=" * 100)
        
        agent = self.agents[agent_name]
        recent_logs = agent.detailed_log[-n_episodes:]
        
        for log in recent_logs:
            step = log['step_details']
            print(f"\nEpisodio {log['episode_number']}:")
            print(f"  Estado inicial: {log['initial_state']} ({step['prev_state_name']})")
            print(f"  Acción: {log['action']} - {step['action_name']}")
            
            if step['action_info']:
                ai = step['action_info']
                print(f"    Modificó {ai['feature']}: {ai['old_value']:.3f} -> {ai['new_value']:.3f}")
            
            song = step['song_selected']
            print(f"  Canción seleccionada: {song['name']} - {song['artist']}")
            print(f"    Similitud: {song['similarity']:.3f}")
            print(f"    Valence/Energy original: {song['valence']:.3f}/{song['energy']:.3f}")
            
            resp = step['response_details']
            print(f"  Respuesta emocional simulada:")
            print(f"    Resonancia: {resp['resonance']:.3f}, Complejidad: {resp['complexity']:.3f}")
            print(f"    Factor personal: {resp['personal_factor']:.3f}")
            print(f"    Respuesta final: {resp['final_response']:.3f}")
            print(f"    Valence/Energy resultante: {step['response_valence']:.3f}/{step['response_energy']:.3f}")
            
            print(f"  Estado final: {log['next_state']} ({step['new_state_name']})")
            print(f"  Recompensa: {log['reward']} - {'¡ÉXITO!' if log['reward'] == 1 else 'Falló'}")
            print(f"  Epsilon: {log['epsilon']:.4f}")

def run_detailed_experiment(dataset_path=None, episodes=150):
    """Ejecuta experimento completo con historial detallado"""
    print("EXPERIMENTO DETALLADO DE RL MUSICAL")
    print("=" * 50)
    
    comparison = MusicRLComparison(dataset_path)
    comparison.run_sequential_comparison(n_episodes=episodes, verbose=False)
    
    # Mostrar historial detallado del mejor agente
    best_agent = max(comparison.agents.items(), 
                    key=lambda x: np.mean(x[1].rewards[-episodes//4:]))
    
    print(f"\nMejor agente: {best_agent[0]}")
    comparison.show_detailed_history(best_agent[0], n_episodes=3)
    
    return comparison

def run_quick_test():
    """Prueba rápida del sistema"""
    print("PRUEBA RÁPIDA DEL SISTEMA")
    print("=" * 30)
    
    comparison = MusicRLComparison()
    comparison.run_sequential_comparison(n_episodes=50, verbose=True)
    return comparison

# Ejemplo de uso
if __name__ == "__main__":
    # Para una prueba rápida
    #comparison = run_quick_test()
    
    # Para experimento completo
    comparison = run_detailed_experiment(dataset_path='dataset.csv',episodes=200)