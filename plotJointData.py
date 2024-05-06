#!/usr/bin/env python3
import re

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    joint_angles = []
    joint_velocities = []
    end_effector_coords = []

    # Regular expressions to match the patterns in the file
    joint_angles_pattern = r"Joint Angles: \((.*?)\)"
    joint_velocities_pattern = r"Joint Velocities: \((.*?)\)"
    end_effector_coords_pattern = r"End Effector Cords: \[\[\[(.*?)\]\], \[\[(.*?)\]\], \[\[(.*?)\]\]\]"

    # Find all matches for each pattern in the content
    joint_angles_matches = re.findall(joint_angles_pattern, content)
    joint_velocities_matches = re.findall(joint_velocities_pattern, content)
    end_effector_coords_matches = re.findall(end_effector_coords_pattern, content)

    # Extract data from matches
    for match in joint_angles_matches:
        joint_angles.append(list(map(float, match.split(','))))
    
    for match in joint_velocities_matches:
        joint_velocities.append(list(map(float, match.split(','))))

    for match in end_effector_coords_matches:
        # Extract coordinates from the match
        coords = list(map(float, match))
        # Append the coordinates to the end effector coords list
        end_effector_coords.append(coords)

    return joint_angles, joint_velocities, end_effector_coords

# Test the function
file_path = "joint_data.txt"
joint_angles, joint_velocities, end_effector_coords = read_data_from_file(file_path)

print("Joint Angles:", joint_angles[0][0])
print("Joint Velocities:", joint_velocities[0])
print("End Effector Coords:", end_effector_coords[0])
