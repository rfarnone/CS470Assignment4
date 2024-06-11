import matplotlib.pyplot as plt

# Define the points
x = [0.5, 0.2, 0.1, 0.05, 0.02]
y = [0.9215686274509803, 0.7037037037037037, 0.6923076923076923, 0.3125, 0.12121212121212122]

# Plot the points
plt.plot(x, y, marker='o', linestyle='-')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Graph of Points')

# Show the graph
plt.grid(True)
plt.show()
