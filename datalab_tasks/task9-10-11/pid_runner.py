import numpy as np
from PID_Controller import PIDController
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sim_class import Simulation

def gen_goal(low, high):
    return np.random.uniform(low=low, high=high, size=(3,))

def run_pid_control():
    # Define working envelope bounds
    low_bound = np.array([-0.187, -0.1705, 0.1695])
    high_bound = np.array([0.253, 0.2195, 0.2908])

    # Initialize PID controllers for each axis
    pid_x = PIDController(kp=5, ki=0.1, kd=0.01, low_bound=low_bound[0], high_bound=high_bound[0])
    pid_y = PIDController(kp=5, ki=0.1, kd=0.01, low_bound=low_bound[1], high_bound=high_bound[1])
    pid_z = PIDController(kp=5, ki=0.1, kd=0.01, low_bound=low_bound[2], high_bound=high_bound[2])

    sim = Simulation(render=True, num_agents=1)
    goal = gen_goal(low_bound, high_bound)

    status = sim.run([[0,0,0,0]])
    robot_id = list(status.keys())[0]
    current_position = np.array(status[robot_id]['pipette_position']) 
    distance = np.linalg.norm(goal - current_position)
       

    # For visualization
    distances = [] # To store the distance to target
    trajectory = []  # To store the pipette's trajectory

    # Simulation loop
    while distance >= 0.001:
        current_position = np.array(status[robot_id]['pipette_position'])        

        trajectory.append(current_position)  # Save the current position for 3D visualization

        # Print controller errors
        error_x = goal[0] - current_position[0]
        error_y = goal[1] - current_position[1]
        error_z = goal[2] - current_position[2]
        #print(f"Controller Errors -> X: {error_x:.4f}, Y: {error_y:.4f}, Z: {error_z:.4f}")
        control_x = pid_x.compute(goal[0], current_position[0])
        control_y = pid_y.compute(goal[1], current_position[1])
        control_z = pid_z.compute(goal[2], current_position[2])
        print(f"goal[2]: {goal[2]}, current_position[2]: {current_position[2]}, control_z: {control_z}")

        # Clamp actions to bounds
        action = [[control_x, control_y, control_z, 0]]
        action = np.append(action, 0)
        status = sim.run([action])

        # Calculate distance and log
        distance = np.linalg.norm(goal - current_position)
        distances.append(distance)
        # steps.append(step)
        print(f"x_error: {error_x:.4f}, y_error: {error_y:.4f}, z_error: {error_z:.4f}, distance: {distance:.4f}, action: {action}, goal: {goal}, current_position: {current_position}")
        # print(f"Step {step + 1}: Distance = {distance:.4f}, X = {current_position[0]:.4f}, Y = {current_position[1]:.4f}, Z = {current_position[2]:.4f}")

    # Plot distance over steps
    plt.figure()
    plt.plot(range(len(distances)), distances, label="Distance to Target")
    plt.axhline(y=0.001, color='r', linestyle='--', label="Target Threshold (0.001 m)")  # Add target line
    plt.xlabel("Steps")
    plt.ylabel("Distance")
    plt.title("PID Controller Performance")
    plt.legend()
    plt.show()

    # 3D Scatter Plot of Trajectory
    trajectory = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=range(len(trajectory)), cmap='viridis', label="Controller Positions")
    ax.scatter(goal[0], goal[1], goal[2], color='r', label="Target Position", s=100)
    ax.set_xlim([low_bound[0], high_bound[0]])
    ax.set_ylim([low_bound[1], high_bound[1]])
    ax.set_zlim([low_bound[2], high_bound[2]])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Trajectory of Pipette Movement")
    ax.legend()
    plt.show()

    # Close environment
    sim.close()

if __name__ == "__main__":
    # Addressing libiomp5md.dll initialization issue
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Use this workaround to avoid OpenMP runtime error

    run_pid_control()
