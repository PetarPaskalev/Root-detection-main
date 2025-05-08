# Robot Simulation Environment

## Overview

This project sets up a custom simulation environment using **PyBullet** to model robotic movement and interactions. The environment simulates:
- Multiple robots (similar to Opentrons).
- A textured plate (specimen) as the robot's workspace.
- Droplet deposition from a robot pipette onto the plate.

An example script demonstrates how the robot's pipette moves to eight defined corners of its working envelope.

---

## Key Files

1. **`sim_class.py`**  
   - Contains the `Simulation` class with features including:
     - Connecting to PyBullet.
     - Creating robots and assigning positions.
     - Dropping spheres (simulated droplets) from the robot pipette.
     - Checking collisions and recording droplet positions.
     - Managing textures and specimen visuals.

2. **`demo_corners.py`**  
   - Demonstrates the robot moving its pipette to eight corners of its 3D working envelope using velocity commands.
   - Logs the pipette's final positions at each corner.

3. **`Task_11.py`**  
   - Script for running the reinforcement learning task for specific scenarios.

4. **`ot2_gym_wrapper_team.py`**  
   - Wrapper script for integrating the OT-2 robotic environment into the simulation pipeline.

5. **`Task13_PID.py`**  
   - Implements PID control logic for precise robotic movements.

6. **`Task13_RL.py`**  
   - Reinforcement learning script for robot control using PPO.

7. **`tester_task11.py`**  
   - Testing script for Task 11, validating RL-based performance.

8. **`PID_Controller.py`**  
   - PID controller logic module, imported into relevant scripts for control tasks.

9. **`CV_pipeline.py`**  
   - Computer vision pipeline for preprocessing and extracting visual features.

10. **`CV_pipeline_RL.py`**  
    - Specialized CV pipeline for reinforcement learning, integrating vision data for decision-making.

---

## Dependencies

Below are the required dependencies for running the simulation:

- **Python** (>=3.8)
- **pybullet** (>=3.2.5)
- **numpy** (>=1.26.4)
- **opencv-python** or **opencv-contrib-python**
- **Pillow**
- **matplotlib**
- **argparse** (standard library for argument parsing)
- **stable-baselines3** (>=1.6.0) for RL integration

Install the required packages via pip:

```bash
pip install pybullet==3.2.5 numpy==1.26.4 opencv-python Pillow matplotlib stable-baselines3
```

---

## Usage

Ensure you have the following folder structure:

```
/project-folder
├── textures/
│   ├── <texture-files>
│   └── _plates/
│       └── <plate-textures>
├── sim_class.py
├── demo_corners.py
├── Task_11.py
├── ot2_gym_wrapper_team.py
├── Task13_PID.py
├── Task13_RL.py
├── tester_task11.py
├── PID_Controller.py
├── CV_pipeline.py
├── CV_pipeline_RL.py
└── README.md
```

### Running the Demonstration Script

Run the following command to start the simulation:

```bash
python demo_corners.py
```

This opens a PyBullet GUI. The robot will move to eight corners in its workspace, and the pipette's final positions will be logged in the console.

---

## Hyperparameters for RL Model

You can adjust hyperparameters for the RL model using command-line arguments. Below are the supported parameters:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for PPO.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for PPO.")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per rollout.")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of PPO epochs.")
parser.add_argument("--accuracy_threshold", type=float, default=0.001, help="Accuracy threshold for task completion.")
args = parser.parse_args()
```

These can be passed while running the RL model script, for example:

```bash
python Task13_RL.py --learning_rate 0.0005 --batch_size 256
```

---

## Working Envelope of the Pipette

The pipette’s movement was tested by sending velocity commands to reach eight corners. After simulation, the final pipette positions were logged as follows:

| Corner | X (m)   | Y (m)   | Z (m)   |
|--------|---------|---------|---------|
| 1      | -0.1870 | -0.1709 | 0.1196  |
| 2      | 0.2530  | -0.1706 | 0.1198  |
| 3      | -0.1870 | 0.2199  | 0.1199  |
| 4      | 0.2530  | 0.2196  | 0.1201  |
| 5      | -0.1870 | -0.1705 | 0.2896  |
| 6      | 0.2530  | -0.1706 | 0.2895  |
| 7      | -0.1870 | 0.2195  | 0.2895  |
| 8      | 0.2530  | 0.2195  | 0.2895  |

Bounding Box of the Working Envelope:

- X Range: ~[-0.19, 0.25] meters
- Y Range: ~[-0.17, 0.22] meters
- Z Range: ~[0.12, 0.29] meters

This represents the pipette’s reachable 3D space based on the tested velocities and kinematics.

---

## Example Usage

Below is an example of using the RL model and PID controller:

```python
from CV_pipeline_RL import process_for_rl
from PID_Controller import PIDController

# Example usage
processed_data = process_for_rl(image_data)
pid = PIDController(kp=5.0, ki=0.1, kd=0.01)
control_signal = pid.compute(error)
```

---

## Contributing

To contribute to this project:

1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add my feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/my-feature
   ```
5. Open a pull request.
