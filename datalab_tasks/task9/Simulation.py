import time
from sim_class import Simulation

# Initialize the simulation for one robot
sim = Simulation(num_agents=1)

# Define the 8 corners of the working envelope
corners = [
    [-0.9, -0.9, -0.9, 0],  # Bottom-front-left
    [0.9, -0.9, -0.9, 0],   # Bottom-front-right
    [-0.9, 0.9, -0.9, 0],   # Bottom-back-left
    [0.9, 0.9, -0.9, 0],    # Bottom-back-right
    [-0.9, -0.9, 0.9, 0],   # Top-front-left
    [0.9, -0.9, 0.9, 0],    # Top-front-right
    [-0.9, 0.9, 0.9, 0],    # Top-back-left
    [0.9, 0.9, 0.9, 0]      # Top-back-right
]

# List to store the states after reaching each corner
corner_positions = []

pipette_positions = []

# Move to each corner
for i, corner in enumerate(corners):
    print(f"Moving to Corner {i+1}: {corner}")
    
    # Set velocities directly from corner values
    velocity_x, velocity_y, velocity_z, drop_command = corner

    # Create action for the current corner
    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    # Run the simulation for 500 steps to approach the target corner gradually
    for _ in range(130):
        sim.run(actions)
        time.sleep(1 / 240)  # Delay for real-time simulation

    # Get the pipette position directly from the simulation class
    pipette_position = sim.get_pipette_position(sim.robotIds[0])

    # Append only the relevant pipette position to the list
    pipette_positions.append(pipette_position)

    # Print the simplified pipette position for the current corner
    print(f"Corner {i+1} Pipette Position: {pipette_position}")

# Print all pipette positions
print("\nRecorded Pipette Positions:")
for i, position in enumerate(pipette_positions):
    print(f"Corner {i+1}: {position}")

# Close the simulation after completing
sim.close()

print("Simulation finished successfully.")

print("Simulation finished successfully.")
