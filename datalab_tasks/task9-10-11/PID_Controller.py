import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt=0.1, low_bound=None, high_bound=None):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.dt = dt  # Time step

        self.prev_error = 0
        self.integral = 0

        # Add working envelope bounds
        self.low_bound = low_bound
        self.high_bound = high_bound

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error

        # Clamp the output to stay within the working envelope bounds
        return output
