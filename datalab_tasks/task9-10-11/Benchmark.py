import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sim_class import Simulation
from CV_pipeline import Pipeline
from PID_Controller import PIDController
from stable_baselines3 import PPO
from ot2_gym_wrapper_team import OT2_wrapper  # RL wrapper

def benchmark_controllers():
    """
    Benchmarks the PID and RL controllers on the same plate and targets.
    Collects time, steps, final distances, and success rates.
    """

    ###########################################################################
    # A) Shared Settings
    ###########################################################################
    num_runs = 5
    accuracy_threshold = 0.013  # 1.3 cm
    success_threshold = 0.013

    # Initialize shared CV pipeline
    pipeline_model_path = "PetarPaskalev_232725_unet_model_3_256px_82F1.h5"
    pipeline = Pipeline(
        model_path=pipeline_model_path,
        patch_size=256,
        plate_size_mm=150,
        plate_origin_in_robot=(0.10775, 0.088 - 0.060, 0.057)
    )

    ###########################################################################
    # B) Metrics Storage
    ###########################################################################
    metrics = {
        "PID": {"time": [], "steps": [], "distance": [], "success": []},
        "RL": {"time": [], "steps": [], "distance": [], "success": []},
    }

    ###########################################################################
    # C) Load RL Model
    ###########################################################################
    rl_model_path = r"D:\Holland_Year_2\Block_B\2024-25b-fai2-adsai-PetarPaskalev232725\datalab_tasks\task9-10-11\models\9z0olsdb\model.zip"
    print(f"[INFO] Loading PPO model from: {rl_model_path}")
    rl_model = PPO.load(rl_model_path)

    ###########################################################################
    # D) Test Each Controller
    ###########################################################################
    for controller in ["PID", "RL"]:
        print(f"\n=== Testing {controller} Controller ===")

        for run in range(num_runs):
            print(f"\n--- Run {run+1}/{num_runs} ---")

            # 1) Initialize Simulation and CV Pipeline
            sim = Simulation(num_agents=1, render=False)
            sim.reset()

            image_path = sim.get_plate_image()
            if not os.path.exists(image_path):
                print("[ERROR] Plate image not found, skipping run.")
                sim.close()
                continue

            _, _, _, endpoints_3d = pipeline.run_pipeline(image_path, visualize=False)
            endpoints_3d = endpoints_3d[:5]

            if not endpoints_3d:
                print("[WARNING] No endpoints found by pipeline. Skipping run.")
                sim.close()
                continue

            # Track metrics for this run
            total_time = 0
            total_steps = 0
            success_count = 0
            final_distances = []

            for tip_index, goal in enumerate(endpoints_3d):
                print(f"  - Moving to Tip #{tip_index+1}: {goal}")

                # 2) Initialize for this tip
                start_time = time.time()
                steps = 0
                success = False
                distance_log = []

                if controller == "PID":
                    # Set up PID controllers
                    pid_x = PIDController(kp=5, ki=0.1, kd=0.01, dt=0.1)
                    pid_y = PIDController(kp=5, ki=0.1, kd=0.01, dt=0.1)
                    pid_z = PIDController(kp=5, ki=0.1, kd=0.01, dt=0.1)

                    while True:
                        current_states = sim.get_states()
                        robot_pos = np.array(
                            current_states[list(current_states.keys())[0]]["pipette_position"]
                        )
                        dist_xy = np.linalg.norm(robot_pos[:2] - goal[:2])
                        distance_log.append(dist_xy)

                        if dist_xy < accuracy_threshold:
                            success = True
                            break

                        ctrl_x = pid_x.compute(goal[0], robot_pos[0])
                        ctrl_y = pid_y.compute(goal[1], robot_pos[1])
                        ctrl_z = pid_z.compute(goal[2], robot_pos[2])
                        action = [[ctrl_x, ctrl_y, ctrl_z, 0]]
                        sim.run(action, num_steps=1)
                        steps += 1

                elif controller == "RL":
                    env = OT2_wrapper(render=False, max_steps=1000, accuracy_threshold=accuracy_threshold)
                    env.goal_position = np.array(goal)
                    obs, info = env.reset()

                    while True:
                        action, _ = rl_model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, info = env.step(action)
                        steps += 1
                        dist_xy = np.linalg.norm(obs[:2] - obs[3:5])
                        distance_log.append(dist_xy)

                        if dist_xy < accuracy_threshold or done or truncated:
                            success = dist_xy < accuracy_threshold
                            break

                # 3) Record metrics
                elapsed_time = time.time() - start_time
                final_distances.append(distance_log[-1])
                total_time += elapsed_time
                total_steps += steps
                if success:
                    success_count += 1

            # Store this run's metrics
            metrics[controller]["time"].append(total_time)
            metrics[controller]["steps"].append(total_steps)
            metrics[controller]["distance"].append(np.mean(final_distances))
            metrics[controller]["success"].append(success_count / len(endpoints_3d))

    ###########################################################################
    # E) Results Summary
    ###########################################################################
    print("\n=== Results Summary ===")
    for controller in metrics.keys():
        print(f"\n{controller} Controller:")
        print(f"  Avg Time: {np.mean(metrics[controller]['time']):.2f} s")
        print(f"  Avg Steps: {np.mean(metrics[controller]['steps']):.1f}")
        print(f"  Avg Final Distance: {np.mean(metrics[controller]['distance']):.4f} m")
        print(f"  Success Rate: {np.mean(metrics[controller]['success'])*100:.1f}%")

    # Close the environment
    env.close()
    print("[INFO] Benchmarking complete. Environment closed.")


if __name__ == "__main__":
    benchmark_controllers()
