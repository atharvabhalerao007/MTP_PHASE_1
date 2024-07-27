import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load LiDAR data from a text file (x, y, z coordinates)
lidar_data = np.loadtxt("")

print("LiDAR Data:")
print(lidar_data[:10])  # Printing 10 rows

# Extract X, Y, and Z coordinates from the LiDAR data
xyz_points = lidar_data[:, :3]  # Assuming Z coordinates are in the third column



# Specify the radius for the circle (you can change this to the outside point of the canopy)
circle_radius = 10

# Initialize a list to store the detected tree crowns (center and radius)
tree_crowns = []

# Find local minima in the z values (assumed crown boundaries)
z_values = xyz_points[:, 2]
minima = np.where(z_values == np.min(z_values))[0]
print(f"Total number of points: {len(z_values)}")

# Define the number of segments
num_segments = 6

# Calculate the number of points per segment
points_per_segment = len(minima) // num_segments


# Initialize a list to store the angle ranges for each segment
angle_ranges = []

# Initialize the start angle for the first segment
start_angle = 0

# Initialize variables to store the minimum and maximum angles for each segment
min_angles_s = [0] * num_segments  # Initialize with low values
max_angles_s = [60] * num_segments  # Initialize with high values

# Define the desired angle ranges for each segment in degrees
desired_angle_ranges = [(0,60),(60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]

# Iterate through segments and calculate angle ranges
for i in range(num_segments):
    if i < num_segments - 1:
        end_index = (i + 1) * points_per_segment
    else:
        end_index = len(minima)

    # Calculate the end angle based on the proportion of data points
    end_angle = minima[end_index - 1] / len(z_values) * 2 * np.pi

    # Update the minimum and maximum angles for the segment in degrees
    min_angles_s[i] = desired_angle_ranges[i][0]
    max_angles_s[i] = desired_angle_ranges[i][1]

    # Update the start angle for the next segment
    start_angle = end_angle
    

    # Update the minimum and maximum angles for the segment in degrees
    min_angles_s[i] = desired_angle_ranges[i][0]
    max_angles_s[i] = desired_angle_ranges[i][1]

    # Update the start angle for the next segment
    start_angle = end_angle

# Correct the angle ranges for S1
min_angles_s[0] = 0
max_angles_s[0] = desired_angle_ranges[0][1]

# Create an empty DataFrame to store segment data
segment_data = pd.DataFrame(columns=['X', 'Y', 'Z', 'Segment'])

# Specify the output directory where Excel files will be saved
output_directory = ""

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Initialize variables to track the optimal starting segment
optimal_starting_segment = 0
min_point_cloud_data = float('inf')

# Print the corrected angles for each segment
for i in range(num_segments):
    print(f"Min Angle for S{i + 1}: {min_angles_s[i]} degrees")
    print(f"Max Angle for S{i + 1}: {max_angles_s[i]} degrees")

# Define a function to fit a circle using RANSAC
def fit_circle_ransac(points):
    # Calculate the centroid of the points
    center = np.mean(points, axis=0)
    print(f"THE CENTER ARE: {center}")

    # Calculate the radius as the maximum distance from the center
    radius = np.max(np.linalg.norm(points - center, axis=1))

    return center, radius

def rotate_segments_fixed_data(data_points, segment_points, angle_degree):
    # Rotate the segments by a specified angle in degrees while keeping data points fixed

    angle_radians = np.radians(angle_degree)

    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 1]])

    # Initialize a list to store the rotated segments
    rotated_segments = []

    for segment in segment_points:
        # Apply the 3D rotation to each point in the segment
        rotated_segment = np.dot(segment, rotation_matrix)

        # Append the rotated segment to the list
        rotated_segments.append(rotated_segment)

    # Combine the fixed data points with the rotated segments
    rotated_data = np.vstack((data_points, *rotated_segments))

    return rotated_data





def divide_circle_into_segments(center, radius, num_segments, xy_points):
    # Initialize a list to store points in each segment
    segment_points = []

    # Initialize a list to store colors for each point
    segment_colors = []

    # Iterate through segments and calculate the points in each segment
    for i in range(num_segments):
        # Calculate the start and end angles for the current segment
        angle_start = min_angles_s[i]
        angle_end = max_angles_s[i]

        delta_x = xy_points[:, 0] - center[0]
        delta_y = xy_points[:, 1] - center[1]

        # Calculate the polar coordinates (R and A) of each point
        R = np.sqrt(delta_x ** 2 + delta_y ** 2)
        A = np.degrees(np.arctan2(delta_y, delta_x))

        # Normalize angles to be in the range [0, 360)
        A = (A + 360) % 360

        # Normalize start and end angles to be in the range [0, 360)
        angle_start = (angle_start + 360) % 360
        angle_end = (angle_end + 360) % 360

        # Check if the point is inside the circular sector
        segment_mask = (
            (R <= radius) &
            (
                (angle_start < angle_end) & np.logical_and(angle_start <= A, A <= angle_end) |
                (angle_start > angle_end) & (np.logical_or(A >= angle_start, A <= angle_end))
            )
        )

        # Append points to the segment_points list
        segment_points.append(xy_points[segment_mask])

        # Assign a unique color to the points in this segment
        color = plt.cm.viridis(i / num_segments)  # Adjust colormap as needed
        segment_colors.append(color)

    return segment_points, segment_colors

# Loop through each local minimum (assumed to be a crown boundary)
for min_index in minima:
    # Extract the points within a neighborhood around the crown boundary
    neighborhood_radius = 3.0  # Adjust as needed
    neighborhood_points = xyz_points[
        np.linalg.norm(xyz_points - xyz_points[min_index], axis=1) <= neighborhood_radius
    ]

    # Fit a circle to the neighborhood points using RANSAC
    center, radius = fit_circle_ransac(neighborhood_points)

    # Divide the circle into segments based on Cartesian coordinates
    num_segments = 6
    segment_points, segment_colors = divide_circle_into_segments(center, radius, num_segments, xyz_points)

     # Initialize variables to track the number of data points in S1 to S6
    count_s = [len(segment_points[i]) if len(segment_points) > i else 0 for i in range(num_segments)]

    # Print the counts for each segment
    for i in range(num_segments):
        print(f"Number of Points in S{i + 1}: {count_s[i]}")

    # Store the detected tree crown (center, radius, segment points, and segment colors)
    tree_crowns.append((center, radius, segment_points, segment_colors))

    # Initialize a DataFrame to store shifting data
    shifting_data = pd.DataFrame(columns=['Angle'] + [f'S{segment_index + 1}' for segment_index in range(num_segments)])

    for angle in range(360):
        # Rotate segments by the current angle while keeping data fixed
        rotated_data = rotate_segments_fixed_data(xyz_points, segment_points, angle)

        # Split rotated data into data_points and rotated_segments
        data_points = rotated_data[:len(xyz_points)]
        rotated_segments = [rotated_data[len(xyz_points):] for i in range(num_segments)]

        # Count the number of points within each segment and print the counts
        segment_point_counts = [len(segment) for segment in rotated_segments]

        # Create a row for this shifting instance
        row_data = [angle] + segment_point_counts

        # Append the row to the shifting_data DataFrame
        shifting_data.loc[len(shifting_data)] = row_data

        # Check if this configuration has less point cloud data than the minimum
        if min(segment_point_counts) < min_point_cloud_data:
            min_point_cloud_data = min(segment_point_counts)
            optimal_starting_segment = angle

    # Use the optimal starting segment configuration
    final_segments = [np.roll(segment, optimal_starting_segment, axis=0) for segment in segment_points]

    # Iterate through segments and exclude points outside the circle
    for segment_index, points in enumerate(final_segments):
        segment_label = f"S{segment_index + 1}"

        # Calculate the distances from the center
        distances = np.linalg.norm(points - center, axis=1)

        # Filter points outside the circle
        inside_circle = (distances <= radius) | (distances <= 1.5)  # Adjust the inner circle radius as needed
        x_segment = points[inside_circle, 0]
        y_segment = points[inside_circle, 1]
        z_segment = points[inside_circle, 2]  # Assuming Z coordinates are in the third column

        # Create a DataFrame for this segment
        segment_df = pd.DataFrame({'X': x_segment, 'Y': y_segment, 'Z': z_segment, 'Segment': [segment_label] * len(x_segment)})

        # Save the segment data to an Excel file
        file_name = f"segment_{min_index}_{segment_label}.xlsx"
        file_path = os.path.join(output_directory, file_name)
        segment_df.to_excel(file_path, index=False)

# Save the shifting data to an Excel file
shifting_data_file = os.path.join(output_directory, "shifting_data.xlsx")
shifting_data.to_excel(shifting_data_file, index=False)

# Visualize the LiDAR data and detected tree crowns with segment labels
plt.figure(figsize=(20, 20))

# Plot points in each segment with their assigned colors and labels for the legend
for i, (points, colors) in enumerate(zip(segment_points, segment_colors)):
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y, c=colors, s=1, label=f'Segment {i + 1}')

# Initialize variables to store if segments have data points
has_data_s = [False] * num_segments

# Iterate through tree crowns and segments
for i, (center, radius, segment_points, _) in enumerate(tree_crowns):
    for segment_index, points in enumerate(segment_points):
        if len(points) > 0:
            has_data_s[segment_index] = True

# Print the results
for i in range(num_segments):
    if has_data_s[i]:
        print(f"S{i + 1} has data points.")
    else:
        print(f"S{i + 1} does not have data points.")

# Print the desired angle ranges for each segment
for i, (start_angle, end_angle) in enumerate(desired_angle_ranges):
    print(f"Desired Angle Range for S{i + 1}: {start_angle} degrees to {end_angle} degrees")

# Iterate through tree crowns and segments
for i, (center, radius, segment_points, _) in enumerate(tree_crowns):
    circle = plt.Circle(center, radius, color='red', fill=False, lw=2)
    plt.gca().add_patch(circle)

    # Plot lines to indicate segment boundaries
    for angle in np.linspace(0, 2 * np.pi, len(segment_points) + 1):
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        plt.plot([center[0], x], [center[1], y], color='green', linestyle='--')

# Add labels for S1 to S6
angles_degrees = [30, 90, 150, 210, 270, 330]

for i, angle_degree in enumerate(angles_degrees):
    x_coordinate_of_label = center[0] + radius * 0.5 * np.cos(np.radians(angle_degree))
    y_coordinate_of_label = center[1] + radius * 0.5 * np.sin(np.radians(angle_degree))
    plt.annotate(f"S{i + 1}", (x_coordinate_of_label, y_coordinate_of_label), fontsize=12, color='blue')

# Print the angle at which the minimum metric occurs
print(f"Angle at which the minimum metric occurs: {optimal_starting_segment} degrees")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Detected Tree Crowns with 6 Segments (RANSAC Circle Fitting)")
plt.colorbar(label="Z Coordinate (Height)")
plt.axis('equal')
plt.show()

output_directory = ""
print("Segment data saved to Excel files in:", output_directory)
