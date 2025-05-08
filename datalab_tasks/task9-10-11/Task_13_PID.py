import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sim_class import Simulation
from CV_pipeline import Pipeline
from PID_Controller import PIDController

def inoculate_with_pid():
    """
    1) Create the Simulation
    2) Use the CV pipeline to find up to 5 root-tip coords
    3) Override the Z coordinate with a fixed 'hover' height
    4) Use only X/Y distance in the PID distance check
    5) Drop inoculum
    """

    print("[DEBUG] Starting inoculate_with_pid()...")

    # A) Create Simulation, Reset
    sim = Simulation(num_agents=1, render=True)
    print("[DEBUG] Simulation created. Now calling sim.reset()...")
    sim.reset()

    # Let the PyBullet GUI appear
    print("[DEBUG] Sleeping 2s to ensure PyBullet GUI is visible.")
    time.sleep(2)

    # B) Pipeline
    model_path = "PetarPaskalev_232725_unet_model_3_256px_82F1.h5"
    print(f"[DEBUG] Initializing pipeline with model: {model_path}")
    pipeline = Pipeline(
        model_path=model_path,
        patch_size=256,
        plate_size_mm=150,
        plate_origin_in_robot=(0.10775, 0.062 - 0.038, 0.057)  # example
    )

    image_path = sim.get_plate_image()
    print(f"[DEBUG] Plate image path from sim: {image_path}")
    if not os.path.exists(image_path):
        print("[ERROR] Plate image not found on disk. Aborting.")
        sim.close()
        return

    print("[DEBUG] Running pipeline.run_pipeline(...) now...")
    final_mask, skeleton_bool, endpoints_px, endpoints_robot = pipeline.run_pipeline(
        image_path, visualize=True
    )
    print("[DEBUG] Pipeline finished! Checking endpoints...")

    if not endpoints_robot:
        print("[WARNING] No root tips found by pipeline. Exiting.")
        sim.close()
        return

    print("\n[INFO] Root Tip Targets (Robot Coords) from pipeline (xyz):")
    for i, tip in enumerate(endpoints_robot):
        print(f"  Tip #{i}: {tip}")

    # C) Setup PID
    low_bound = np.array([-0.187, -0.1705, 0.1695])
    high_bound = np.array([0.253, 0.2195, 0.2908])

    pid_x = PIDController(kp=5, ki=0.1, kd=0.01, dt=0.1, 
                          low_bound=low_bound[0], high_bound=high_bound[0])
    pid_y = PIDController(kp=5, ki=0.1, kd=0.01, dt=0.1, 
                          low_bound=low_bound[1], high_bound=high_bound[1])
    pid_z = PIDController(kp=5, ki=0.1, kd=0.01, dt=0.1, 
                          low_bound=low_bound[2], high_bound=high_bound[2])

    distance_logs = []

    # D) For each root tip, override Z & do 2D distance check
    states = sim.get_states()
    robot_key = list(states.keys())[0]
    print(f"[DEBUG] Found robot key: {robot_key}")

    # If you only want 5 tips max:
    # endpoints_robot = endpoints_robot[:5]

    # Decide on a 'hover' Z
    fixed_hover_z = 0.19

    for i, original_goal in enumerate(endpoints_robot):
        # We only want x & y from pipeline
        # Force a constant Z:
        goal = (original_goal[0], original_goal[1], fixed_hover_z)

        print(f"\n=== Root Tip #{i} => (X={goal[0]:.3f}, Y={goal[1]:.3f}, Z={goal[2]:.3f}) ===")

        # Reset integrals
        pid_x.prev_error = 0;  pid_x.integral = 0
        pid_y.prev_error = 0;  pid_y.integral = 0
        pid_z.prev_error = 0;  pid_z.integral = 0

        # We'll do only X/Y distance
        threshold_2d = 0.001  # e.g. 1 cm
        step_distances = []

        while True:
            current_states = sim.get_states()
            current_pos = np.array(current_states[robot_key]["pipette_position"], dtype=float)

            # x/y distance only
            dist_2d = np.sqrt(
                (goal[0] - current_pos[0])**2 +
                (goal[1] - current_pos[1])**2
            )
            step_distances.append(dist_2d)

            if dist_2d < threshold_2d:
                print(f"[INFO] Reached root tip {i}! Dropping inoculum.")
                sim.run([[0, 0, 0, 1]], num_steps=1)
                print("[DEBUG] Droplet called. Waiting 50 steps to see it.")
                sim.run([[0, 0, 0, 0]], num_steps=50)
                break

            # Compute x,y,z
            ctrl_x = pid_x.compute(goal[0], current_pos[0])
            ctrl_y = pid_y.compute(goal[1], current_pos[1])
            # For Z, we keep the pipette at fixed_hover_z
            ctrl_z = pid_z.compute(goal[2], current_pos[2])

            print(f"[DEBUG] dist2D={dist_2d:.4f}, ctrl=({ctrl_x:.2f},{ctrl_y:.2f},{ctrl_z:.2f}), "
                  f"pos=({current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f})")

            action = [[ctrl_x, ctrl_y, ctrl_z, 0]]
            sim.run(action, num_steps=1)

        distance_logs.append(step_distances)

    # E) Plot Distances
    plt.figure()
    for idx, dists in enumerate(distance_logs):
        color = f'C{idx}'  # Get matplotlib's default color cycle
        # Plot trajectory without label
        plt.plot(dists, 'o', markersize=2, color=color)
        # Show inoculation points in legend with matching colors
        plt.plot(len(dists)-1, dists[-1], 'o', markersize=10, color=color,
                label=f"Tip #{idx} ({len(dists)} steps)")
    
    plt.title("2D Distance to Each Root Tip Over Time")
    plt.xlabel("Steps Made")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

    sim.close()
    print("[DEBUG] All done!")

if __name__ == "__main__":
    inoculate_with_pid()
