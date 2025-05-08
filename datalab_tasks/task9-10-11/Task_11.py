import wandb
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper_team import OT2_wrapper, SaveClosestToZeroRewardCallback  # Ensure this includes the callback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
import os
import argparse

# Disable GPU usage if needed


# Disable symlinks in WandB to avoid permission issues on Windows


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for PPO.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for PPO.")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per rollout.")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of PPO epochs.")
parser.add_argument("--accuracy_threshold", type=float, default=0.001, help="Accuracy threshold for task completion.")
args = parser.parse_args()

# Ensure the WANDB_API_KEY is set before initializing wandb
os.environ['WANDB_API_KEY'] = 'dde7d1a68c5d76900e0e5bc636a28254a9759c13'

# Create the environment with the accuracy threshold
env = OT2_wrapper(render=False, max_steps=1000, accuracy_threshold=args.accuracy_threshold)

# Initialize the WandB project
run = wandb.init(
    project="sb3_OT2",
    sync_tensorboard=True,
    config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "accuracy_threshold": args.accuracy_threshold,
    }
)

# Define the path for saving the best model
best_model_path = f"models/{run.id}/best_model_closest_to_zero.zip"

# Initialize the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",
)

# Initialize callbacks
save_closest_to_zero_callback = SaveClosestToZeroRewardCallback(save_path=best_model_path)
wandb_callback = WandbCallback(
    model_save_path=f"models/{run.id}",
    verbose=2
)

# Combine callbacks
callback = CallbackList([save_closest_to_zero_callback, wandb_callback])

# Train the model
model.learn(
    total_timesteps=2000000,  # Train for 2 million steps
    callback=callback,  # Custom callback handles reward-based saving
    progress_bar=True,
    reset_num_timesteps=False,
    tb_log_name=f"runs/{run.id}",
)

# Save the final model
final_model_path = f"models/{run.id}/final_model_4.zip"
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)  # Ensure the directory exists
model.save(final_model_path)
print(f"Final model saved at: {final_model_path}")

# Close the environment and finalize WandB run
env.close()
run.finish()
