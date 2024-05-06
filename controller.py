#!/usr/bin/env python3

"""
Start ROS node to publish angles for the position control of the xArm7.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t

# Arm parameters
# xArm7 kinematics class
from kinematics import xArm7_kinematics

# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

class xArm7_controller():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # Init xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()

        # joints' angular positions
        self.joint_angpos = [0, 0, 0, 0, 0, 0, 0]
        # joints' angular velocities
        self.joint_angvel = [0, 0, 0, 0, 0, 0, 0]
        # joints' states
        self.joint_states = JointState()
        # joints' transformation matrix wrt the robot's base frame
        self.A01 = self.kinematics.tf_A01(self.joint_angpos)
        self.A02 = self.kinematics.tf_A02(self.joint_angpos)
        self.A03 = self.kinematics.tf_A03(self.joint_angpos)
        self.A04 = self.kinematics.tf_A04(self.joint_angpos)
        self.A05 = self.kinematics.tf_A05(self.joint_angpos)
        self.A06 = self.kinematics.tf_A06(self.joint_angpos)
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)
        # gazebo model's states
        self.model_states = ModelStates()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.joint1_pos_pub = rospy.Publisher('/xarm/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_pos_pub = rospy.Publisher('/xarm/joint2_position_controller/command', Float64, queue_size=1)
        self.joint3_pos_pub = rospy.Publisher('/xarm/joint3_position_controller/command', Float64, queue_size=1)
        self.joint4_pos_pub = rospy.Publisher('/xarm/joint4_position_controller/command', Float64, queue_size=1)
        self.joint5_pos_pub = rospy.Publisher('/xarm/joint5_position_controller/command', Float64, queue_size=1)
        self.joint6_pos_pub = rospy.Publisher('/xarm/joint6_position_controller/command', Float64, queue_size=1)
        self.joint7_pos_pub = rospy.Publisher('/xarm/joint7_position_controller/command', Float64, queue_size=1)
        # Obstacles
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of joint 1 is stored in :: self.joint_states.position[0])

    def model_states_callback(self, msg):
        # ROS callback to get the gazebo's model_states

        self.model_states = msg
        # (e.g. #1 the position in y-axis of GREEN obstacle's center is stored in :: self.model_states.pose[1].position.y)
        # (e.g. #2 the position in y-axis of RED obstacle's center is stored in :: self.model_states.pose[2].position.y)

    def publish(self):

        # set configuration
        # total pitch: j2-j4+j6+pi (upwards: 0rad)

        j2 = 0.7 ; j4 = np.pi/2
        j6 = - (j2-j4)
        self.joint_angpos = [0, j2, 0, j4, 0, j6, 0]
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        self.joint4_pos_pub.publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint2_pos_pub.publish(self.joint_angpos[1])
        self.joint6_pos_pub.publish(self.joint_angpos[5])
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        # Get end effector coordinates:
        A07 = self.kinematics.tf_A07(self.joint_angpos)
        print("End effector coordinates are: ", A07[:3, 3])

        # Trajectory definition
        # Define trajectory period
        T = 4

        # Define time vector
        numOfSteps = 2001
        Time       = np.linspace(0, 4, 2001)

        # Define end effector position and velocity vectors
        px = np.ones(numOfSteps)*0.617
        py = 0.2*np.sin(pi/2*Time)
        pz = np.ones(numOfSteps)*0.199

        pDotX = np.zeros(numOfSteps)
        pDotY = 0.2*pi/2*np.cos(pi/2*Time)
        pDotZ = np.zeros(numOfSteps)

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()

        counter = 0
        while not rospy.is_shutdown():

            thisIterationIdx = np.mod(counter, numOfSteps)

            # Compute each transformation matrix wrt the base frame from joints' angular positions
            self.A01 = self.kinematics.tf_A01(self.joint_angpos)
            self.A02 = self.kinematics.tf_A02(self.joint_angpos)
            self.A03 = self.kinematics.tf_A03(self.joint_angpos)
            self.A04 = self.kinematics.tf_A04(self.joint_angpos)
            self.A05 = self.kinematics.tf_A05(self.joint_angpos)
            self.A06 = self.kinematics.tf_A06(self.joint_angpos)
            self.A07 = self.kinematics.tf_A07(self.joint_angpos)

            # Compute jacobian matrix
            J = self.kinematics.compute_jacobian(self.joint_angpos)
            # pseudoinverse jacobian
            pinvJ = pinv(J)

            # First routine, calculate joint speeds
            pDot = np.matrix([[pDotX[thisIterationIdx]], [pDotY[thisIterationIdx]], [pDotZ[thisIterationIdx]]])
            p    = np.matrix([[px[thisIterationIdx]], [py[thisIterationIdx]], [pz[thisIterationIdx]]])
            pReal= self.A07[:3,3]
            K = 100

            q1Dot = pinvJ @ (pDot + K*(p - pReal))
            print(q1Dot)
            input("Press Enter to continue...")
            # Second routine, avoid obstacles
            greenObstPosY = self.model_states.pose[1].position.y
            redObstPosY   = self.model_states.pose[2].position.y

            # Important to adjust centerpoint with every iteration to acount for movement of obstacles
            centerPoint = (greenObstPosY+redObstPosY)/2

            # We will try to minimize the y distance of joints 3 4 and 5 from centerPoint
            distanceVectors = np.matrix([[(self.A03[1,3] - centerPoint)**2], \
                                         [(self.A04[1,3] - centerPoint)**2], \
                                         [(self.A04[1,3] - centerPoint)**2]])

            # Calculate gradient decent of distanceVectors:
            # Get joint positions
            q1 = self.joint_angpos[0]
            q2 = self.joint_angpos[1]
            q3 = self.joint_angpos[2]
            q4 = self.joint_angpos[3]

            # Get link lengths
            l1 = self.kinematics.l1
            l2 = self.kinematics.l2
            l3 = self.kinematics.l3
            l4 = self.kinematics.l4

            # Get configuration angles
            theta1 = self.kinematics.theta1

            objFunc03 = np.zeros((7,1))
            objFunc03[0] = 2*(self.A03[1,3] - centerPoint)*l2*sin(q2)*cos(q1)
            objFunc03[1] = 2*(self.A03[1,3] - centerPoint)*l2*sin(q1)*cos(q2)

            objFunc04 = np.zeros((7,1))
            objFunc04[0] = 2*(self.A04[1,3] - centerPoint)*l2*sin(q2)*(cos(q1) + l3*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)))
            objFunc04[1] = 2*(self.A04[1,3] - centerPoint)*(l2*sin(q1)*cos(q2) - l3*sin(q1)*sin(q2)*cos(q3))
            objFunc04[2] = 2*(self.A04[1,3] - centerPoint)*(l3*(-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3)))

            objFunc05 = np.zeros((7,1))
            factor0 = l2*sin(q2)*cos(q1) + l3*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)) + \
                      l4*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*sin(theta1) - \
                      l4*((sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3))*sin(q4) + sin(q2)*cos(q1)*cos(q4))*cos(theta1)
            
            factor1 = l2*sin(q1)*cos(q2) - l3*sin(q1)*sin(q2)*cos(q3) - \
                      l4*(sin(q1)*sin(q2)*sin(q4)*cos(q3) + sin(q1)*cos(q2)*cos(q4))*cos(theta1) + \
                      l4*(-sin(q1)*sin(q2)*cos(q3)*cos(q4) + sin(q1)*sin(q4)*cos(q2))*sin(theta1)
            
            factor2 = l3*(-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3)) + \
                      l4*(-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*sin(theta1)*cos(q4) - \
                      l4*(sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3))*sin(q4)*cos(theta1)
            
            factor3 = -l4*((-sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1))*cos(q4) - sin(q1)*sin(q2)*sin(q4))*cos(theta1) + \
                       l4*(-(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + sin(q1)*sin(q2)*cos(q4))*sin(theta1)
            
            objFunc05[0] = 2*(self.A05[1,3] - centerPoint)*factor0
            objFunc05[1] = 2*(self.A05[1,3] - centerPoint)*factor1
            objFunc05[2] = 2*(self.A05[1,3] - centerPoint)*factor2
            objFunc05[3] = 2*(self.A05[1,3] - centerPoint)*factor3

            jointWeights = np.array([10,40,10])

            objFunction  = jointWeights[0]*objFunc03 + jointWeights[1]*objFunc04 + jointWeights[2]*objFunc05
            
            # Using theory from lecture slides we know that 
            Kc = 50
            q2Dot = Kc*(np.identity(7) - pinvJ @ J) @ objFunction

            self.joint_angvel = [q1Dot[i,0] + q2Dot[i,0] for i in range(7)]

            print()
            # for i in range(7):
            #     self.joint_angvel[i] = q1Dot[i,0] + q2Dot[i,0]

            # Convertion to angular position after integrating the angular speed in time
            # Calculate time interval
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9
            # Integration
            self.joint_angpos = np.add( self.joint_angpos, [index * dt for index in self.joint_angvel] )

            # Publish the new joint's angular positions
            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            self.joint3_pos_pub.publish(self.joint_angpos[2])
            self.joint4_pos_pub.publish(self.joint_angpos[3])
            self.joint5_pos_pub.publish(self.joint_angpos[4])
            self.joint6_pos_pub.publish(self.joint_angpos[5])
            self.joint7_pos_pub.publish(self.joint_angpos[6])

            self.pub_rate.sleep()

            thisIterationIdx = thisIterationIdx+1

    def turn_off(self):
        pass

def controller_py():
    # Starts a new node
    rospy.init_node('controller_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    controller = xArm7_controller(rate)
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass
