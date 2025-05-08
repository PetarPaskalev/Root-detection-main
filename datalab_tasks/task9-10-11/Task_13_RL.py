import os
import time
import numpy as np

# Set OpenMP environment variable to avoid errors with multiple runtimes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for compatibility
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ot2_gym_wrapper_team import OT2_wrapper  # environment

from CV_pipeline_RL import Pipeline  # your pipeline

"""
NOTEBOOK CODE REFERENCE - Showing the correct coordinate system handling

The notebook code demonstrates proper image coordinate handling, where:
- Row coordinates (image_coord_*_0) represent Y position
- Column coordinates (image_coord_*_1) represent X position

This matches our fixed CV_pipeline_RL.py transformation, where:
- Row (y in image) maps to mm_y and then robot Y
- Column (x in image) maps to mm_x and then robot X

With this proper mapping, minimal calibration offsets are needed.

Key notebook code:
```python
# Extract start and end points of the skeleton
for skeleton_id, group in branch_data.groupby("skeleton-id"):
    # Start point (minimum y-coordinate)
    start_branch = group.loc[group[["image-coord-src-0", "image-coord-dst-0"]].min(axis=1).idxmin()]
    start_y = start_branch[["image-coord-src-0", "image-coord-dst-0"]].min()
    start_x = start_branch[["image-coord-src-1", "image-coord-dst-1"]].iloc[...]

    # End point (maximum y-coordinate)
    end_branch = group.loc[group[["image-coord-src-0", "image-coord-dst-0"]].max(axis=1).idxmax()]
    end_y = end_branch[["image-coord-src-0", "image-coord-dst-0"]].max()
    end_x = end_branch[["image-coord-src-1", "image-coord-dst-1"]].iloc[...]
```

This approach is now consistent with our fixed coordinate transformation.
"""

def test_rl_many_drops_one_plate():
    model_path = r"D:\Holland_Year_2\Block_B\2024-25b-fai2-adsai-PetarPaskalev232725\datalab_tasks\task9-10-11\models\9z0olsdb\model.zip"
    print(f"[INFO] Loading PPO model from: {model_path}")
    model = PPO.load(model_path)

    # Keep success threshold at 0.01 as requested
    success_threshold = 0.001
    env = OT2_wrapper(render=True, max_steps=1500, accuracy_threshold=0.01)
    obs, info = env.reset()

    # Wait for the environment to initialize properly
    print("[DEBUG] Sleeping 2s to ensure PyBullet GUI is visible.")
    time.sleep(2)

    # ========================================================================
    # COORDINATE SYSTEM CORRECTION
    # ========================================================================
    # Reverting to the original 90° rotation transformation:
    # 1. CV pipeline now maps row (y in image) → mm_x, column (x in image) → mm_y
    # 2. This matches the original PID approach which worked well
    # 3. Applying calibration offsets similar to PID implementation
    # ========================================================================

    # Use pipeline with correct parameters
    pipeline_model_path = "PetarPaskalev_232725_unet_model_3_256px_82F1.h5"
    print(f"[DEBUG] Initializing pipeline with model: {pipeline_model_path}")
    
    # Use the original plate origin from PID implementation
    # We're reverting to PID-style coordinate mapping which worked
    pipeline = Pipeline(
        model_path=pipeline_model_path,
        patch_size=256,
        plate_size_mm=150,
        plate_origin_in_robot=(0.10775, 0.062 - 0.038, 0.057)
    )

    sim = env.sim
    image_path = sim.get_plate_image()
    print(f"[DEBUG] Plate image path from sim: {image_path}")
    if not os.path.exists(image_path):
        print("[ERROR] Plate image not found, aborting test.")
        env.close()
        return

    print("[DEBUG] Running pipeline.run_pipeline(...) now...")
    final_mask, skel_bool, endpoints_px, endpoints_3d = pipeline.run_pipeline(
        image_path, visualize=True
    )
    print("[DEBUG] Pipeline finished! Checking endpoints...")

    if not endpoints_3d:
        print("[WARNING] Pipeline found no endpoints. Exiting.")
        env.close()
        return

    # Display original endpoints
    print("\n[INFO] Root Tip Targets (Robot Coords) from pipeline (xyz):")
    for i, tip in enumerate(endpoints_3d):
        print(f"  Tip #{i}: {tip}")

    # ========================================================================
    # CALIBRATION FOR 90° ROTATION APPROACH
    # ========================================================================
    # Using PID-style approach with rotation-based coordinate system:
    # 1. Apply the hover height from the PID implementation
    # 2. Use small calibration offsets based on PID success
    
    # Fixed hover height for consistent droplet formation
    fixed_hover_z = 0.19  # Original hover height from PID implementation
    
    # Small calibration offsets for fine-tuning
    coord_fix_x = 0.004   # 4mm X shift 
    coord_fix_y = -0.003  # -3mm Y shift
    
    print("\n[INFO] Applying PID-style coordinate calibration:")
    print(f"  X-correction: {coord_fix_x}m ({coord_fix_x*1000}mm)")
    print(f"  Y-correction: {coord_fix_y}m ({coord_fix_y*1000}mm)")
    print(f"  Z-height: {fixed_hover_z}m (consistent with PID implementation)")
    
    # Apply correction to all tip coordinates
    corrected_endpoints = []
    print("\n[INFO] Original vs. Corrected coordinates:")
    for i, original in enumerate(endpoints_3d):
        # Apply minimal correction and fixed Z height
        corrected_x = original[0] + coord_fix_x
        corrected_y = original[1] + coord_fix_y
        
        corrected_endpoints.append((corrected_x, corrected_y, fixed_hover_z))
        print(f"  Tip #{i}: Original: ({original[0]:.4f}, {original[1]:.4f}, {original[2]:.4f}) → "
              f"Corrected: ({corrected_x:.4f}, {corrected_y:.4f}, {fixed_hover_z:.4f})")
    
    # Keep only 5 tips maximum for consistency
    endpoints_3d = corrected_endpoints[:5]

    # Storage for plotting - similar to PID implementation
    distance_logs = []
    
    # Process each tip with smoother, more PID-like approach
    for i, goal in enumerate(endpoints_3d):
        print(f"\n=== Root Tip #{i} => (X={goal[0]:.3f}, Y={goal[1]:.3f}, Z={goal[2]:.3f}) ===")
        
        # Set goal for current tip
        env.goal_position = np.array(goal, dtype=float)
        
        # Track distances for this tip
        step_distances = []
        
        # Try reaching current tip
        max_steps = 700
        step_count = 0
        reached = False
        
        # Main movement loop - with smooth approach
        while step_count < max_steps and not reached:
            # Get RL model prediction
            action, _ = model.predict(obs, deterministic=True)
            
            # Use moderate scaling for smooth, stable movement
            scaled_action = action * 0.5  # Moderate scaling like PID
            
            # Execute action in environment
            obs, reward, done, truncated, info = env.step(scaled_action)
            step_count += 1
            
            # Pause every 10 steps to let motion stabilize
            if step_count % 10 == 0:
                time.sleep(0.05)
            
            # Get current 2D distance (exactly like in PID)
            current_pos = obs[:3]  # pipette position
            goal_pos = obs[3:6]    # goal position
            
            # Calculate 2D distance (XY only) exactly like PID
            dist_2d = np.sqrt(
                (goal_pos[0] - current_pos[0])**2 +
                (goal_pos[1] - current_pos[1])**2
            )
            step_distances.append(dist_2d)
            
            # Print status in same format as PID
            if step_count % 5 == 0:
                print(f"[DEBUG] dist2D={dist_2d:.4f}, "
                      f"pos=({current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f})")
            
            # Check if we reached the current tip using the success threshold
            if dist_2d < success_threshold:
                reached = True
                print(f"[INFO] Reached root tip {i}! Position stable.")
                
                # Ensure position is completely stable before dropping
                print("[INFO] Stabilizing position before dropping...")
                for j in range(10):  # 10 zero-action steps to ensure stability
                    obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.0])
                    time.sleep(0.05)  # Small pause between steps
                
                # Final position check after stabilization
                current_pos = obs[:3]
                print(f"[INFO] Final stable position: ({current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f})")
                
                # Drop inoculum
                print("[INFO] Dropping inoculum...")
                # First, move slightly up to ensure clean drop (0.001m as in simulation)
                obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.001, 0.0])
                time.sleep(0.2)  # Longer pause to ensure stability
                
                # Drop the inoculum
                print("[DEBUG] Creating droplet...")
                drop_action = [0.0, 0.0, 0.0, 1.0]
                obs, reward, done, truncated, info = env.step(drop_action)
                
                # Wait longer to see droplet (60 steps = 0.6s)
                print("[DEBUG] Waiting for droplet to appear...")
                for _ in range(60):
                    obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.0, 0.0])
                    time.sleep(0.01)  # Small delay between steps
                
                # Move up slightly after dropping to prevent sticking (0.002m)
                obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.002, 0.0])
                time.sleep(0.2)  # Longer pause after moving up
                
                # Store distance log
                distance_logs.append(step_distances)
                break
                
        # Handle timeout
        if not reached:
            print(f"[WARNING] Failed to reach tip #{i} within {max_steps} steps.")
            if step_distances:
                distance_logs.append(step_distances)
    
    # All tips have been processed - keep the simulation window open for inspection
    print("\n[INFO] Completed all inoculations with coordinate correction!")
    print(f"Tips processed: {len(distance_logs)} / {len(endpoints_3d)}")
    
    # Move pipette up to better see all droplets on the plate
    print("[INFO] Moving pipette up to better view all droplets...")
    for _ in range(20):
        obs, reward, done, truncated, info = env.step([0.0, 0.0, 0.1, 0.0])  # Move up
    
    # Keep the simulation window open for careful inspection
    print("\n[IMPORTANT] Keeping simulation window open for 30 seconds for drop placement inspection.")
    print("             Please examine the PyBullet window to verify improved droplet positioning.")
    print("             Press Ctrl+C to exit early if needed.")
    
    try:
        # Count down timer
        for i in range(30, 0, -1):
            if i % 5 == 0:  # Show countdown every 5 seconds
                print(f"[INFO] Closing in {i} seconds... (Ctrl+C to exit now)")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted. Proceeding to close...")
    
    # Now close the environment
    print("[INFO] Closing environment...")
    env.close()
    
    # Wait for environment to fully close to avoid resource conflicts
    time.sleep(2)
    
    # Plot exact same format as PID implementation
    print("[INFO] Creating distance plot...")
    plt.figure(figsize=(10, 6))
    for idx, dists in enumerate(distance_logs):
        color = f'C{idx}'
        plt.plot(dists, 'o', markersize=2, color=color)
        plt.plot(len(dists)-1, dists[-1], 'o', markersize=10, color=color,
                label=f"Tip #{idx} ({len(dists)} steps)")
    
    plt.title("2D Distance to Each Root Tip Over Time")
    plt.xlabel("Steps Made")
    plt.ylabel("Distance (m)")
    plt.axhline(y=success_threshold, color='r', linestyle='--', label=f"Success ({success_threshold}m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    print("[INFO] Displaying plot. Close the plot window to exit.")
    plt.show(block=True)  # Block until plot is closed
    
    print("[DEBUG] All done!")

if __name__ == "__main__":
    test_rl_many_drops_one_plate()
