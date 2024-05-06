#!/usr/bin/env python3

"""
Compute state space kinematic matrices for xArm7 robot arm (5 links, 7 joints)
"""

import numpy as np

class xArm7_kinematics():
    def __init__(self):

        self.l1 = 0.267
        self.l2 = 0.293
        self.l3 = 0.0525
        self.l4 = 0.3512
        self.l5 = 0.1232

        self.theta1 = 0.2225 #(rad) (=12.75deg)
        self.theta2 = 0.6646 #(rad) (=38.08deg)

        pass

    def compute_jacobian(self, r_joints_array):
        # Store transformation matrices
        A01 = self.tf_A01(r_joints_array)
        A02 = self.tf_A02(r_joints_array)
        A03 = self.tf_A03(r_joints_array)
        A04 = self.tf_A04(r_joints_array)
        A05 = self.tf_A05(r_joints_array)
        A06 = self.tf_A06(r_joints_array)
        A07 = self.tf_A07(r_joints_array)

        b0 = np.matrix([[0],
                        [0],
                        [1]])
        b1 = A01[:3, 2]
        b2 = A02[:3, 2]
        b3 = A03[:3, 2]
        b4 = A04[:3, 2]
        b5 = A05[:3, 2]
        b6 = A06[:3, 2]


        r0 = A07[:3, 3]
        r1 = r0 - A01[:3, 3]
        r2 = r0 - A02[:3, 3]
        r3 = r0 - A03[:3, 3]
        r4 = r0 - A04[:3, 3]
        r5 = r0 - A05[:3, 3]
        r6 = r0 - A06[:3, 3]
    
        # Here we flatten the row vectors of (3,1) shape for np.cross to work
        J1 = np.squeeze(np.cross(b0.flatten(),r0.flatten()))
        J2 = np.squeeze(np.cross(b1.flatten(), r1.flatten()))
        J3 = np.squeeze(np.cross(b2.flatten(), r2.flatten()))
        J4 = np.squeeze(np.cross(b3.flatten(), r3.flatten()))
        J5 = np.squeeze(np.cross(b4.flatten(), r4.flatten()))
        J6 = np.squeeze(np.cross(b5.flatten(), r5.flatten()))
        J7 = np.squeeze(np.cross(b6.flatten(), r6.flatten()))

        J = np.matrix([ [ J1[0] , J2[0] , J3[0] , J4[0] , J5[0] , J6[0] , J7[0] ],\
                        [ J1[1] , J2[1] , J3[1] , J4[1] , J5[1] , J6[1] , J7[1] ],\
                        [ J1[2] , J2[2] , J3[2] , J4[2] , J5[2] , J6[2] , J7[2] ]])
        return J

    def tf_A01(self, r_joints_array):
        tf = np.matrix([[np.cos(r_joints_array[0]) , -np.sin(r_joints_array[0]) , 0 , 0      ],\
                        [np.sin(r_joints_array[0]) ,  np.cos(r_joints_array[0]) , 0 , 0      ],\
                        [0                         ,  0                         , 1 , self.l1],\
                        [0                         ,  0                         , 0 , 1      ]])
        return tf

    def tf_A02(self, r_joints_array):
        tf_A12 = np.matrix([[np.cos(r_joints_array[1]) , -np.sin(r_joints_array[1]) , 0 , 0],\
                            [0                         , 0                          , 1 , 0],\
                            [-np.sin(r_joints_array[1]), -np.cos(r_joints_array[1]) , 0 , 0],\
                            [0                         , 0                          , 0 , 1]])
        tf = np.dot( self.tf_A01(r_joints_array), tf_A12 )
        return tf

    def tf_A03(self, r_joints_array):
        tf_A23 = np.matrix([[np.cos(r_joints_array[2]) , -np.sin(r_joints_array[2]) , 0 , 0       ],\
                            [0                         , 0                          , -1, -self.l2],\
                            [np.sin(r_joints_array[2]) , np.cos(r_joints_array[2])  , 0 , 0       ],\
                            [0                         , 0                          , 0 , 1       ]])
        tf = np.dot( self.tf_A02(r_joints_array), tf_A23 )
        return tf

    def tf_A04(self, r_joints_array):
        tf_A34 = np.matrix([[np.cos(r_joints_array[3]) , -np.sin(r_joints_array[3]) , 0 , self.l3],\
                            [0                         , 0                          , -1, 0      ],\
                            [np.sin(r_joints_array[3]) , np.cos(r_joints_array[3])  , 0 , 0      ],\
                            [0                         , 0                          , 0 , 1      ]])
        tf = np.dot( self.tf_A03(r_joints_array), tf_A34 )
        return tf

    def tf_A05(self, r_joints_array):
        tf_A45 = np.matrix([[np.cos(r_joints_array[4]) , -np.sin(r_joints_array[4]) ,  0 ,  self.l4 * np.sin(self.theta1)],\
                            [0                         , 0                          , -1 , -self.l4 * np.cos(self.theta1)],\
                            [np.sin(r_joints_array[4]) , np.cos(r_joints_array[4])  ,  0 , 0                             ],\
                            [0                         , 0                          ,  0 , 1                             ]])
        tf = np.dot( self.tf_A04(r_joints_array), tf_A45 )
        return tf

    def tf_A06(self, r_joints_array):
        tf_A56 = np.matrix([[np.cos(r_joints_array[5]) , -np.sin(r_joints_array[5]) , 0  , 0],\
                            [0                         , 0                          , -1 , 0],\
                            [np.sin(r_joints_array[5]) , np.cos(r_joints_array[5])  , 0  , 0],\
                            [0                         , 0                          , 0  , 1]])
        tf = np.dot( self.tf_A05(r_joints_array), tf_A56 )
        return tf

    def tf_A07(self, r_joints_array):
        tf_A67 = np.matrix([[np.cos(r_joints_array[6]) , -np.sin(r_joints_array[6]) , 0 , self.l5 * np.sin(self.theta2)],\
                            [0                         , 0                          , 1 , self.l5 * np.cos(self.theta2)],\
                            [-np.sin(r_joints_array[6]), -np.cos(r_joints_array[6]) , 0 , 0                            ],\
                            [0                         , 0                          , 0 , 1                            ]])
        tf = np.dot( self.tf_A06(r_joints_array), tf_A67 )
        return tf
