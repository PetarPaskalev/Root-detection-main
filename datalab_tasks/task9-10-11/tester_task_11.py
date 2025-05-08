import time
import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper_team import OT2_wrapper

# Load the trained model
model_path = r"D:\Holland_Year_2\Block_B\2024-25b-fai2-adsai-PetarPaskalev232725\datalab_tasks\task9-10-11\models\3c7jli03\model.zip"  # Replace <your_model_id> with your specific ID
model = PPO.load(model_path)

# Initialize the environment
env = OT2_wrapper(render=True, max_steps=1000, accuracy_threshold=0.001)

# Test parameters
num_episodes = 10  # Number of test episodes
success_threshold = 0.001  # Success if distance to goal is within 1 cm

# Initialize metrics
success_count = 0
total_distance = 0
total_steps = 0

print("\nStarting test...")

# Run test episodes
# Run test episodes
for episode in range(num_episodes):
    print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
    obs, info = env.reset()  # Extract observation and info
    done = False
    episode_distance = 0
    steps = 0

    while not done:
        # Get the action from the model
        action, _ = model.predict(obs)  # Only pass the observation
        obs, reward, done, truncated, info = env.step(action)

        # Calculate distance to the goal
        distance_to_goal = np.linalg.norm(obs[:3] - obs[3:])
        episode_distance += distance_to_goal
        steps += 1

        # Real-time display
        print(f"Step {steps}: Distance to goal = {distance_to_goal:.4f} meters", end="\r")
        time.sleep(0.05)  # Add a small delay for real-time effect

        # Check for truncation or success
        if done or truncated:
            success = distance_to_goal <= success_threshold
            if success:
                success_count += 1

            # Print results for the episode
            print(f"\nEpisode {episode + 1} complete:")
            print(f"  Steps Taken: {steps}")
            print(f"  Final Distance to Goal: {distance_to_goal:.4f} meters")
            print(f"  Success: {'Yes' if success else 'No'}")

    total_distance += episode_distance / steps  # Average distance per episode
    total_steps += steps
