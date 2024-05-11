#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    joint_angles = []
    joint_velocities = []
    end_effector_cords = []

    # Regular expressions to match the patterns in the file
    joint_angles_pattern = r"Joint Angles: \((.*?)\)"
    joint_velocities_pattern = r"Joint Velocities: \((.*?)\)"
    end_effector_cords_pattern = r"End Effector Cords: \[\[\[(.*?)\]\], \[\[(.*?)\]\], \[\[(.*?)\]\]\]"

    # Find all matches for each pattern in the content
    joint_angles_matches = re.findall(joint_angles_pattern, content)
    joint_velocities_matches = re.findall(joint_velocities_pattern, content)
    end_effector_cords_matches = re.findall(end_effector_cords_pattern, content)

    # Extract data from matches
    for match in joint_angles_matches:
        joint_angles.append(list(map(float, match.split(','))))
    
    for match in joint_velocities_matches:
        joint_velocities.append(list(map(float, match.split(','))))

    for match in end_effector_cords_matches:
        # Extract cordinates from the match
        cords = list(map(float, match))
        # Append the cordinates to the end effector cords list
        end_effector_cords.append(cords)

    return joint_angles, joint_velocities, end_effector_cords

# Test the function
file_path = "joint_data.txt"
joint_angles, joint_velocities, end_effector_cords = read_data_from_file(file_path)

for iJoint in range(7):
    thisAngleVector = []
    thisVelocityVector = []
    for iTimeStep in range(len(joint_angles)):
        thisAngleVector.append(joint_angles[iTimeStep][iJoint])
        thisVelocityVector.append(joint_velocities[iTimeStep][iJoint])

    fig = plt.figure()
    plt.suptitle("Joint " + str(iJoint + 1) + " data")  # Title for the whole figure
    plt.subplot(2, 1, 1)
    plt.grid(True)
    plt.plot(thisAngleVector * 180/np.pi)
    plt.ylabel('Joint Angle [deg]')
    plt.subplot(2, 1, 2)
    plt.grid(True)
    plt.plot(thisVelocityVector)
    plt.xlabel('Steps')
    plt.ylabel('Joint Velocity [rad/s]')

    figureName = "Moved_Obstacels\Joint" + str(iJoint) + "_figure.png" 
    plt.savefig(figureName)
    plt.close()  # Close the figure to release memory and avoid overlapping plots

cordNames = ["X", "Y", "Z"]
for iCordinate in range(len(cordNames)):
    thisCordName = cordNames[iCordinate]
    thisEndEffectorCord = []
    for iTimeStep in range(len(end_effector_cords)):
        thisEndEffectorCord.append(end_effector_cords[iTimeStep][iCordinate])

    fig = plt.figure()
    plt.suptitle("Joint " + thisCordName + " data")  
    plt.grid(True)
    plt.plot(thisEndEffectorCord)
    plt.ylabel('Joint Angle [rad]')

    figureName = "End Effector" + thisCordName + "_figure.png" 
    plt.savefig(figureName)
    plt.close()  # Close the figure to release memory and avoid overlapping plots

# print("Joint Angles:", joint_angles[0][0])
# print("Joint Velocities:", joint_velocities[0])
# print("End Effector cords:", end_effector_cords[0])
