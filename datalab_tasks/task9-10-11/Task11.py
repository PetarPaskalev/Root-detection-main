import wandb
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2_wrapper
from stable_baselines3 import PPO
import os
import argparse
import tensorflow

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage if needed  # Optional: Disable WANDB syncing for debugging

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--milestones", type=list, default=[round(i * 0.05, 2) for i in range(1, 20)])

args = parser.parse_args()

# Ensure the WANDB_API_KEY is set before initializing wandb
os.environ['WANDB_API_KEY'] = 'dde7d1a68c5d76900e0e5bc636a28254a9759c13'  # Replace with your actual WandB API key

# Create the environment
env = OT2_wrapper(args.milestones, max_steps=1000)

# Initialize wandb project
run = wandb.init(
    project="sb3_OT2",
    sync_tensorboard=True
)

# Initialize the PPO model with WandB tensorboard logging
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",
)

# Create WandB callback
wandb_callback = WandbCallback(
    model_save_freq=2000000,  # Save the model at the end of 2 million steps
    model_save_path=f"models/{run.id}",
    verbose=2
)

# Train the model for 5 million timesteps
model.learn(
    total_timesteps=5000000,  # Train for 5 million steps
    callback=wandb_callback,
    progress_bar=True,
    reset_num_timesteps=False,
    tb_log_name=f"runs/{run.id}"
)

# Save the final model
save_path = f"models/{run.id}/final_model_1.zip"
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
model.save(save_path)
print(f"Final model saved at: {save_path}")
