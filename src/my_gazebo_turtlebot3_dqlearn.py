import rospy
import roslaunch
import time
import numpy as np
import math
from math import pi

import random

from gym import spaces
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
import tf

from gym.utils import seeding

class GazeboTurtlebot3DQLearnEnv():

    def __init__(self):
        # Initialize the node
        rospy.init_node('gazebo_dqlearning_node', anonymous=True)
        # Connect to gazebo
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        #self.laser_point_count = 180  # 360 laser point in one time
        self.min_crash_range = 0.2  # Make done = True if agent close to wall 

        self.oldDistance = 0
        self.targetX = 10
        self.targetY = 10
        self.success = True

    def calculate_observation(self, data, heading, distance):
        min_range = 0.2
        max_range = 6.0
        isCrash = False
        data = list(data.ranges)
        for i in range(len(data)):
            if (min_range > data[i] > 0):
                isCrash = True
            if np.isinf(data[i]):
                data[i] = max_range
            if np.isnan(data[i]):
                data[i] = 0
        return data + [heading, distance], isCrash

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except Exception:
            print ("/gazebo/unpause_physics service call failed")

        # 3 actions
        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 1
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = -1
            self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except Exception:
                print("/scan Error laser data is empty!")
                time.sleep(5)

        odomData = None
        while odomData is None:
            try:
                odomData = rospy.wait_for_message('/odom', Odometry, timeout=5)
            except Exception:
                print("/odom Error odom data is empty!")
                time.sleep(5)

        myX = odomData.pose.pose.position.x
        myY = odomData.pose.pose.position.y

        self.newDistance = math.sqrt((self.targetX - myX)**2 + (self.targetY - myY)**2)

        orientation = odomData.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.targetY - myY, self.targetX - myX)

        heading = goal_angle - yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        heading = round(heading, 2)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except Exception:
            print ("/gazebo/pause_physics service call failed")

        state, isCrash = self.calculate_observation(data, heading, self.newDistance)
        
        done = False
        if isCrash:
            done = True

        if self.newDistance < 0.2:  # Reached to target
            done = True

        if isCrash:
            reward = -200
        elif done:
            # Reached to target
            rospy.logerr("Reached to target   x = " + str(self.targetX) + "   y = "+str(self.targetY))
            self.success = True
            reward = 200
        else:
            # Negative reward for distance
            reward = (self.oldDistance - self.newDistance)*8

        self.oldDistance = self.newDistance

        return np.asarray(state), reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except Exception:
            print ("/gazebo/reset_simulation service call failed")

        # maze 2 limits
        #xLimits = [[-1.4, -0.211], [-0.823, 2.192]]
        #yLimits = [[-2.603, -1.185], [-0.763, 0.47]]

        # maze 5 limits to create random target point in map
        #xLimits = [[-2, 5], [4, 5]]
        #yLimits = [[-7, -8], [-8, 1]]
        # stage 4 limits to create random target point in map
        xLimits = [[0, 1], [0, 1], [1.5, 2], [-0.7, -1.2], [-2, -1], [1, 2]]
        yLimits = [[1, 2], [-2, -1], [-2, 0], [-2, -1], [0.8, 1.2], [0.8, 1.2]]

        randomInt = random.randint(0,5)

        if self.success:
            self.targetX = random.uniform(*xLimits[randomInt])
            self.targetY = random.uniform(*yLimits[randomInt])
            rospy.logerr("X = " + str(self.targetX) +" Y = "+ str(self.targetY))
            self.success = False

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except Exception:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        laserData = None
        while laserData is None:
            try:
                laserData = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except Exception:
                print("/scan Error laser data is empty!")
                time.sleep(5)

        odomData = None
        while odomData is None:
            try:
                odomData = rospy.wait_for_message('/odom', Odometry, timeout=5)
            except Exception:
                print("/odom Error odom data is empty!")
                time.sleep(5)

        myX = odomData.pose.pose.position.x
        myY = odomData.pose.pose.position.y

        self.newDistance = math.sqrt((self.targetX - myX)**2 + (self.targetY - myY)**2)

        orientation = odomData.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.targetY - myY, self.targetX - myX)

        heading = goal_angle - yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        heading = round(heading, 2)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except Exception:
            print ("/gazebo/pause_physics service call failed")

        state, isCrash = self.calculate_observation(laserData, heading, self.newDistance)
        
        return np.asarray(state)