import math
import numpy as np
import pandas as pd

def between(angle, start_angle, end_angle):
    return (
        (start_angle <= end_angle and start_angle <= angle <= end_angle)
        or (start_angle > end_angle and not end_angle < angle < start_angle)
    )

def calculate_wedge_counts(lidar_data, center, radius, wedge_angle, rotation_angle):
    polar_coordinates = []
    for point in lidar_data:
        x, y = point[0], point[1]
        distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        angle = (math.atan2(x - center[0], y - center[1]) * (180/math.pi) + 360) % 360  # Calculate and normalize angle
        polar_coordinates.append((angle, distance))

    counts = {}

    for i in range(6):  # Assuming 6 wedges
        start_angle = ((i * 60 + rotation_angle) + 360) % 360  # Normalize start_angle
        end_angle = ((start_angle + wedge_angle) + 360) % 360  # Normalize end_angle

        wedge_points_count = sum(between(angle, start_angle, end_angle) for (angle, distance) in polar_coordinates if distance <= radius)
        
        counts[i] = wedge_points_count

    return counts

# Center coordinates
center = []

# Load LiDAR data
lidar_data = np.loadtxt("")

# Constants
radius = 10
wedge_angle = 60

# Create a DataFrame to store the counts
result = {}  # Use a dictionary to store the results

# Calculate wedge counts for 360-degree rotation
for rotation_angle in range(360):
    counts = calculate_wedge_counts(lidar_data, center, radius, wedge_angle, rotation_angle)
    result[rotation_angle] = counts

# Create a DataFrame from the result
result_df = pd.DataFrame(result).T  # Transpose the DataFrame

# Save the DataFrame to an Excel file
output_file_path = "wedge_points_count_optimized_61.xlsx"
result_df.to_excel(output_file_path)

print(f"Points count data saved to {output_file_path}")