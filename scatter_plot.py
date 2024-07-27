import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load data from the Excel file
load_data = pd.read_excel("")

# Extract the 'X', 'Y', 'Z' columns
x = load_data['X']
y = load_data['Y']
z = load_data['Z']

xo = 684006.63141068
yo = 5792362.4662047

# Calculate the radius for each data point
radius = np.sqrt((y - yo) ** 2 + (x - xo) ** 2)

# Normalize the 'Radius' and 'Z' values to the range [0, 1]
scaler = MinMaxScaler()
radius_normalized = scaler.fit_transform(radius.values.reshape(-1, 1))
z_normalized = scaler.fit_transform(load_data['Z'].values.reshape(-1, 1))

# Add the 'Radius' and 'Z' columns to your DataFrame
load_data['Radius'] = radius_normalized
load_data['Z_normalized'] = z_normalized

# Define a set of points from the normalized data
points = load_data[['Radius', 'Z_normalized']].values

# Create an empty list to store the points at the end of the radius in each section
end_points = []

# Iterate over sections with a step of 0.005 in Z before normalization
for section_start in np.arange(0, len(z), 0.03):
    section_end = section_start + 0.03
    section_points = points[(points[:, 1] >= section_start) & (points[:, 1] <= section_end)]
    if len(section_points) > 0:
        end_point = section_points[np.argmax(section_points[:, 0])]
        end_points.append(end_point)

# Convert end_points to numpy array for easy plotting
end_points = np.array(end_points)

# Plot the points at the end of the radius
plt.scatter(*zip(*points))
plt.plot(end_points[:, 0], end_points[:, 1], color='red', marker='o', linestyle='-', linewidth=2)
plt.title('368_S-6')
plt.xlabel('Radius (Normalized)')
plt.ylabel('Z (Normalized)')
plt.show()