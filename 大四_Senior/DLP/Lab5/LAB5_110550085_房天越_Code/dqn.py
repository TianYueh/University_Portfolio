# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time


gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x/255.0)  # Normalize the input to [0, 1] range


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_frames=1e6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta
        self.beta = beta
        self.beta_frames = beta_frames
        self.frame = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ########## 
        
        
        if error is not None:
            p = (abs(error) + 1e-5) ** self.alpha
        else:
            p = self.priorities.max() if len(self.buffer)>0 else 1.0


        

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity
        
        ########## END OF YOUR CODE (for Task 3) ########## 
        #return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1

        # --- indices 管理修正 ---
        n = len(self.buffer)
        probs = self.priorities[:n]
        P = probs / (probs.sum() + 1e-8)

        
        indices = np.random.choice(n, batch_size, p=P)
        transitions = [self.buffer[i] for i in indices]
        
        weights = (n * P[indices]) ** (-self.beta)
        weights /= weights.max()
        
        
        ########## END OF YOUR CODE (for Task 3) ########## 
        return transitions, indices, torch.tensor(weights, dtype=torch.float32)
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        assert max(indices) < len(self.buffer)

        for i, error in zip(indices, errors):
            p = (abs(error) + 1e-5) ** self.alpha
            self.priorities[i] = p
        ########## END OF YOUR CODE (for Task 3) ########## 
        #return
        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.env.reset(seed=args.seed)
        self.env.action_space.seed(args.seed)
        self.test_env.reset(seed=args.seed)
        self.test_env.action_space.seed(args.seed)
        self.args = args

        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        
        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        
        if hasattr(args, "resume_model") and args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location=self.device)
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f"Resumed from {args.resume_model}")
            self.epsilon = args.epsilon_start
            #self.memory.clear()

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_mid = args.epsilon_mid
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay_steps1 = args.epsilon_decay_steps1
        self.epsilon_decay_steps2 = args.epsilon_decay_steps2
        self.epsilon_min = args.epsilon_min
        
        self.replay_start_size = args.replay_start_size
        self.epsilon_start = args.epsilon_start

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta=args.per_beta)
        self.save_threshold = set([200000, 400000, 600000, 800000, 1000000])
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    def store(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return
        cumulative_reward = 0
        next_state = None
        done = False
        for idx, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            cumulative_reward += (self.gamma ** idx) * r
            next_state = ns
            done = done or d
        s0, a0, _, _, _ = self.n_step_buffer[0]
        self.memory.add((s0, a0, cumulative_reward, next_state, done))

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.args.seed + ep)

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0
            self.n_step_buffer.clear()

            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                total_reward += reward
                self.env_count += 1
                step_count += 1
                self.store((state, action, reward, next_state, done))
                #self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()
                
                # epsilon schedule in run
                if self.env_count <= self.epsilon_decay_steps1:
                    frac = self.env_count / self.epsilon_decay_steps1
                    self.epsilon = self.epsilon_start + frac * (self.epsilon_mid - self.epsilon_start)
                elif self.env_count <= self.epsilon_decay_steps1 + self.epsilon_decay_steps2:
                    frac = (self.env_count - self.epsilon_decay_steps1) / self.epsilon_decay_steps2
                    self.epsilon = self.epsilon_mid + frac * (self.epsilon_end - self.epsilon_mid)
                else:
                    self.epsilon = self.epsilon_end
                    
                self.epsilon = max(self.epsilon, self.epsilon_min)


                state = next_state
                
                

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                
                if self.env_count in self.save_threshold:
                    filename = f"LAB5_110550085_task3_pong{self.env_count}.pt"
                    torch.save(self.q_net.state_dict(), os.path.join(self.save_dir, filename))
                ########## YOUR CODE HERE  ##########
                # Add additional wandb logs for debugging if needed 
                if hasattr(self, 'latest_loss'): wandb.log({"Loss": self.latest_loss})
                ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            wandb.log({"Eval_Loss": getattr(self, 'latest_loss', None)})
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset(seed=self.args.seed)
        state = self.preprocessor.reset(obs)
        done = False
        prev_epsilon = self.epsilon
        self.epsilon = 0.0
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        self.epsilon = prev_epsilon
        return total_reward


    def train(self):

        if len(self.memory.buffer) < self.replay_start_size:
            return

       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer

        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = weights.to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
      
        with torch.no_grad():
            best_action = self.q_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_action).squeeze()

            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
            
        td_errors = target_q_values - q_values
        loss = (td_errors.pow(2) * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm = 1.0)
        self.optimizer.step()
        
        # Update the priorities in the replay buffer
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.train_count += 1
            
        self.latest_loss = loss.item()
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--wandb-run-name", type=str, default="pong-enhanced-run")
    parser.add_argument("--save-dir", type=str, default="./results_per")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=200000)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-mid", type=float, default=0.1)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps1", type=int, default=500000)
    parser.add_argument("--epsilon-decay-steps2", type=int, default=500000)
    parser.add_argument("--target-update-frequency", type=int, default=5000)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--resume-model", type=str, default=None)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    wandb.init(project="DLP-Lab5-DQN-Atari-Enhanced-DQN", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name = args.env_name, args=args)
    agent.run(args.num_episodes)