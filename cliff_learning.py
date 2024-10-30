import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pygame
import pickle

class MultiHazardCliffEnv(gym.Env):
    """Custom Environment that follows gym interface with added setback probability and Pygame visualization."""
    metadata = {'render_modes': ['human']}

    def __init__(self, setback_probability=0.05, setback_penalty=-3):
        super(MultiHazardCliffEnv, self).__init__()
        
        # Define grid size
        self.rows = 4
        self.cols = 12
        
        # Action space: up (0), right (1), down (2), left (3)
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([self.rows-1, self.cols-1]),
            dtype=np.int32
        )
        
        # Setback mechanics
        self.setback_probability = setback_probability
        self.setback_penalty = setback_penalty
        
        # Define hazards and their penalties
        self.hazards = {
            'cliff': -100,
            'mud': -10,
            'wind': -5
        }
        
        # Define hazard locations
        self.hazard_map = {}
        # Cliff hazards
        for i in range(1, self.cols-1):
            self.hazard_map[(self.rows-1, i)] = 'cliff'
        # Fixed mud hazards
        self.hazard_map.update({
            (0, 2): 'mud', (0, 3): 'mud', (0, 4): 'mud', (0, 5): 'mud',
            (1, 3): 'mud', (1, 4): 'mud', (1, 5): 'mud', (1, 8): 'mud',
            (1, 9): 'mud', (2, 8): 'mud', (2, 9): 'mud'
        })
        # Fixed wind hazards
        self.hazard_map.update({
            (0, 0): 'wind', (1, 0): 'wind', (1, 7): 'wind', (2, 7): 'wind'
        })
        
        self.start_state = (self.rows-1, 0)
        self.goal_state = (self.rows-1, self.cols-1)
        self.current_state = None
        
        # Pygame setup
        self.cell_size = 60
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size
        self.screen = None
        self.clock = None
        self.font = None
        
        # Colors
        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'agent': (0, 0, 255),
            'cliff': (255, 0, 0),
            'mud': (139, 69, 19),
            'wind': (173, 216, 230),
            'start': (0, 255, 0),
            'goal': (255, 215, 0),
            'text': (0, 0, 0)
        }

    def step(self, action):
        assert self.action_space.contains(action)
        
        x, y = self.current_state
        prev_state = self.current_state
        
        # Movement logic
        if action == 0:    # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.cols-1, y+1)
        elif action == 2:  # down
            x = min(self.rows-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
            
        new_state = (x, y)
        
        # Reward logic
        reward = -1
        done = False
        info = {}
        
        # Apply setback probability
        if new_state != prev_state and random.random() < self.setback_probability:
            reward += self.setback_penalty
            info['setback'] = True
        
        if new_state in self.hazard_map:
            hazard_type = self.hazard_map[new_state]
            reward += self.hazards[hazard_type]
            if hazard_type == 'cliff':
                done = True
                new_state = self.start_state
                info['hazard'] = 'cliff'
            else:
                info['hazard'] = hazard_type
        
        if new_state == self.goal_state:
            reward += 100
            done = True
            info['success'] = True
            
        self.current_state = new_state
        return np.array(new_state), reward, done, False, info
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        if not pygame.get_init():
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Multi-Hazard Cliff Environment')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        return np.array(self.current_state), {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        for i in range(self.rows + 1):
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (0, i * self.cell_size), 
                           (self.width, i * self.cell_size))
        for j in range(self.cols + 1):
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (j * self.cell_size, 0), 
                           (j * self.cell_size, self.height))
        
        # Draw hazards
        for pos, hazard_type in self.hazard_map.items():
            x, y = pos
            rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors[hazard_type], rect)
            
            # Add hazard label
            label = self.font.render(hazard_type[0].upper(), True, self.colors['text'])
            label_rect = label.get_rect(center=(y * self.cell_size + self.cell_size//2,
                                              x * self.cell_size + self.cell_size//2))
            self.screen.blit(label, label_rect)
        
        # Draw start and goal
        start_rect = pygame.Rect(self.start_state[1] * self.cell_size, 
                               self.start_state[0] * self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors['start'], start_rect)
        
        goal_rect = pygame.Rect(self.goal_state[1] * self.cell_size,
                              self.goal_state[0] * self.cell_size,
                              self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors['goal'], goal_rect)
        
        # Draw agent
        agent_x = self.current_state[1] * self.cell_size + self.cell_size//2
        agent_y = self.current_state[0] * self.cell_size + self.cell_size//2
        pygame.draw.circle(self.screen, self.colors['agent'], 
                         (agent_x, agent_y), self.cell_size//3)
        
        pygame.display.flip()
        self.clock.tick(30)
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
    
    def close(self):
        if pygame.get_init():
            pygame.quit()

class BaseQLearning:
    """Base class for Q-Learning implementations."""
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = env.action_space.n
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def get_state_key(self, state):
        """Convert state array to tuple for dictionary key."""
        return tuple(state)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy with epsilon decay."""
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state):
        """Update Q-values based on experience."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        
        self.q_table[state_key][action] += self.lr * td_error
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def save_q_table(self, file_path):
        """Save Q-table to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self, file_path):
        """Load Q-table from a file."""
        with open(file_path, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), q_table_dict)

class QLearning(BaseQLearning):
    """Standard Q-Learning implementation"""
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor, epsilon)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
    def get_action(self, state):
        """Select action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state):
        """Update Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Calculate TD target and error
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        
        # Update Q-value
        self.q_table[state_key][action] += self.lr * td_error

# Extend Q-Learning with Double Q-Learning, Triple Q-Learning, Quadruple Q-Learning
class DoubleQLearning(BaseQLearning):
    """Double Q-Learning implementation"""
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor, epsilon)
        self.q_table_A = defaultdict(lambda: np.zeros(self.n_actions))
        self.q_table_B = defaultdict(lambda: np.zeros(self.n_actions))

    def get_action(self, state):
        """Select action using epsilon-greedy policy on combined Q-values"""
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table_A[state_key] + self.q_table_B[state_key])

    def update(self, state, action, reward, next_state):
        """Update Q-values for state-action pair in either Q-table A or B"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if random.random() < 0.5:
            best_action_B = np.argmax(self.q_table_B[next_state_key])
            td_target = reward + self.gamma * self.q_table_B[next_state_key][best_action_B]
            td_error = td_target - self.q_table_A[state_key][action]
            self.q_table_A[state_key][action] += self.lr * td_error
        else:
            best_action_A = np.argmax(self.q_table_A[next_state_key])
            td_target = reward + self.gamma * self.q_table_A[next_state_key][best_action_A]
            td_error = td_target - self.q_table_B[state_key][action]
            self.q_table_B[state_key][action] += self.lr * td_error

class TripleQLearning(DoubleQLearning):
    """Triple Q-Learning implementation"""
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor, epsilon)
        self.q_table_C = defaultdict(lambda: np.zeros(self.n_actions))

    def get_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table_A[state_key] + self.q_table_B[state_key] + self.q_table_C[state_key])

    def update(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        choice = random.choice(['A', 'B', 'C'])
        if choice == 'A':
            best_action_BC = np.argmax(self.q_table_B[next_state_key] + self.q_table_C[next_state_key])
            td_target = reward + self.gamma * (self.q_table_B[next_state_key][best_action_BC] + self.q_table_C[next_state_key][best_action_BC]) / 2
            self.q_table_A[state_key][action] += self.lr * (td_target - self.q_table_A[state_key][action])
        elif choice == 'B':
            best_action_AC = np.argmax(self.q_table_A[next_state_key] + self.q_table_C[next_state_key])
            td_target = reward + self.gamma * (self.q_table_A[next_state_key][best_action_AC] + self.q_table_C[next_state_key][best_action_AC]) / 2
            self.q_table_B[state_key][action] += self.lr * (td_target - self.q_table_B[state_key][action])
        else:
            best_action_AB = np.argmax(self.q_table_A[next_state_key] + self.q_table_B[next_state_key])
            td_target = reward + self.gamma * (self.q_table_A[next_state_key][best_action_AB] + self.q_table_B[next_state_key][best_action_AB]) / 2
            self.q_table_C[state_key][action] += self.lr * (td_target - self.q_table_C[state_key][action])

class QuadrupleQLearning(TripleQLearning):
    """Quadruple Q-Learning implementation"""
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor, epsilon)
        self.q_table_D = defaultdict(lambda: np.zeros(self.n_actions))

    def get_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table_A[state_key] + self.q_table_B[state_key] + self.q_table_C[state_key] + self.q_table_D[state_key])

    def update(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        choice = random.choice(['A', 'B', 'C', 'D'])
        if choice == 'A':
            best_action = np.argmax(self.q_table_B[next_state_key] + self.q_table_C[next_state_key] + self.q_table_D[next_state_key])
            td_target = reward + self.gamma * (self.q_table_B[next_state_key][best_action] + self.q_table_C[next_state_key][best_action] + self.q_table_D[next_state_key][best_action]) / 3
            self.q_table_A[state_key][action] += self.lr * (td_target - self.q_table_A[state_key][action])
        elif choice == 'B':
            best_action = np.argmax(self.q_table_A[next_state_key] + self.q_table_C[next_state_key] + self.q_table_D[next_state_key])
            td_target = reward + self.gamma * (self.q_table_A[next_state_key][best_action] + self.q_table_C[next_state_key][best_action] + self.q_table_D[next_state_key][best_action]) / 3
            self.q_table_B[state_key][action] += self.lr * (td_target - self.q_table_B[state_key][action])
        elif choice == 'C':
            best_action = np.argmax(self.q_table_A[next_state_key] + self.q_table_B[next_state_key] + self.q_table_D[next_state_key])
            td_target = reward + self.gamma * (self.q_table_A[next_state_key][best_action] + self.q_table_B[next_state_key][best_action] + self.q_table_D[next_state_key][best_action]) / 3
            self.q_table_C[state_key][action] += self.lr * (td_target - self.q_table_C[state_key][action])
        else:
            best_action = np.argmax(self.q_table_A[next_state_key] + self.q_table_B[next_state_key] + self.q_table_C[next_state_key])
            td_target = reward + self.gamma * (self.q_table_A[next_state_key][best_action] + self.q_table_B[next_state_key][best_action] + self.q_table_C[next_state_key][best_action]) / 3
            self.q_table_D[state_key][action] += self.lr * (td_target - self.q_table_D[state_key][action])

def train_and_evaluate(agent_class, env, episodes=500):
    agent = agent_class(env)
    rewards = []
    start_time = time.time()
    checkpoint_time = start_time  # Initialize time for tracking checkpoints

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            # Render the environment and add delay
            env.render()
            time.sleep(0.01)  # 0.01-second delay for visualization
            
        rewards.append(total_reward)
        
        # Every 100 episodes, print progress and time taken
        if (episode + 1) % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - checkpoint_time
            checkpoint_time = current_time  # Update checkpoint
            print(f"{agent_class.__name__}: Episode {episode + 1} completed in {elapsed_time:.2f} seconds")

    end_time = time.time()
    avg_reward = np.mean(rewards)
    training_time = end_time - start_time
    return rewards, avg_reward, training_time

def run_all():
    env = MultiHazardCliffEnv()
    models = [QLearning, DoubleQLearning, TripleQLearning, QuadrupleQLearning]
    colors = ['blue', 'green', 'red', 'purple']
    
    # Dictionary to hold statistics for each model
    stats = {}
    
    # Generate separate plots for each model
    for model, color in zip(models, colors):
        rewards, avg_reward, training_time = train_and_evaluate(model, env)
        
        # Plot individual rewards for each model
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, color=color)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title(f"Rewards for {model.__name__}")
        plt.savefig(f"{model.__name__}_rewards.png")
        plt.show()
        
        # Save statistics
        stats[model.__name__] = {'avg_reward': avg_reward, 'training_time': training_time}
    
    # Display statistics for all models
    print("\nModel Statistics:")
    for model_name, data in stats.items():
        print(f"{model_name}: Avg Reward = {data['avg_reward']:.2f}, Training Time = {data['training_time']:.2f} seconds")

if __name__ == "__main__":
    run_all()