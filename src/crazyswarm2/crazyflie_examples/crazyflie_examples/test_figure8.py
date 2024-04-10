import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = np.linspace(0, 2*np.pi, 1000)  # Time parameter
a = 1  # Amplitude in x-direction
b = 0.5  # Amplitude in y-direction

# Parametric equations for figure 8 trajectory
x = a * np.sin(t)
y = b * np.sin(2*t)

# Plotting the figure 8 trajectory
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Figure 8 Trajectory')
plt.title('Figure 8 Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
