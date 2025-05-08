from grpc import Status
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import pybullet as p

class OT2_wrapper(gym.Env):
    def __init__(self, render=False, max_steps=1000, accuracy_threshold=0.01):
        super(OT2_wrapper, self).__init__()

        # Set render and max steps
        self.render = render 
        self.max_steps = max_steps
        self.accuracy_threshold = accuracy_threshold  # Threshold for accuracy
        self.goal_position = None

        # PyBullet simulation instance
        self.sim = Simulation(render=render, num_agents=1)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]), 
            high=np.array([1, 1, 1]), 
            shape=(3,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1]), 
            high=np.array([1, 1, 1, 1, 1, 1]), 
            shape=(6,), 
            dtype=np.float32
        )

        # Track steps
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Define working envelope bounds
        low_bound = [-0.187, -0.1705, 0.1695]
        high_bound = [0.253, 0.2195, 0.2908]

        # Generate random goal position
        self.goal_position = np.random.uniform(low=low_bound, high=high_bound, size=(3,))

        # Reset simulation and extract initial state
        status = self.sim.reset(num_agents=1)
        robot_id = list(status.keys())[0]

        # Observation includes pipette and goal positions
        observation = np.array(status[robot_id]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([observation, self.goal_position])

        self.steps = 0

        return observation, {}

    def step(self, action):
        action = np.append(action, 0)  # Add placeholder for drop action

        # Execute action and get updated observation
        observation_dict = self.sim.run([action])
        robot_id = list(observation_dict.keys())[0]
        pipette_position = np.array(observation_dict[robot_id]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position])

        # Compute reward and distance
        reward, distance = self.compute_reward(observation)

        # Check termination conditions
        terminated, termination_reason = self.check_termination(distance)
        truncated = self.steps >= self.max_steps

        # Log additional info
        info = {
            'Pipette coordinates': pipette_position.tolist(),
            'Goal coordinates': self.goal_position.tolist(),
            'Distance to goal': distance,
            'Reward': reward,
            'Termination reason': termination_reason if terminated else ""
        }

        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def compute_reward(self, observation):
        # Calculate Euclidean distance to the goal
        distance = np.linalg.norm(observation[:3] - observation[3:6])

        # Reward is negative distance (closer is better)
        reward = -distance

        # Optionally, add a bonus for achieving very high accuracy
        if distance <= self.accuracy_threshold:
            reward += 1.0  # Positive reward for high accuracy

        return reward, distance

    def check_termination(self, distance):
        if distance < self.accuracy_threshold:
            return True, "Goal reached with required accuracy"
        return False, None

    def close(self):
        self.sim.close()

if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    # Instantiate and check the environment
    wrapped_env = OT2_wrapper(render=False, accuracy_threshold=0.001)
    check_env(wrapped_env)
from stable_baselines3.common.callbacks import BaseCallback

class SaveClosestToZeroRewardCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super(SaveClosestToZeroRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.closest_to_zero = float('inf')

    def _on_step(self) -> bool:
        # Safely retrieve the mean reward
        mean_reward = self.locals.get("rollout/ep_rew_mean", None)

        if mean_reward is not None:
            # Calculate the absolute distance to zero
            distance_to_zero = abs(mean_reward)

            # Save the model if it's the closest to zero so far
            if distance_to_zero < self.closest_to_zero:
                self.closest_to_zero = distance_to_zero
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"New best model saved with mean reward closest to zero: {mean_reward}")

        return True

