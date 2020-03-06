import rospy
import roslaunch
import time
import numpy as np
import math

import random

from gym import spaces
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

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

        self.laser_point_count = 180  # 360 laser point in one time
        self.min_crash_range = 0.2  # Make done = True if agent close to wall 

        self.oldDistance = 0

    def calculate_observation(self, data):
        min_range = 0.2
        max_range = 6.0
        isCrash = False
        data = list(data.ranges)
        for i, item in enumerate(data):
            if (min_range > data[i] > 0):
                isCrash = True
            if np.isinf(data[i]):
                data[i] = max_range
        return data, isCrash

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except Exception:
            print ("/gazebo/unpause_physics service call failed")

        ang_vel = action

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.5
        vel_cmd.angular.z = ang_vel
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

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except Exception:
            print ("/gazebo/pause_physics service call failed")

        state, isCrash = self.calculate_observation(data)
        
        done = False
        if isCrash:
            done = True
        
        # maze 2 limits
        #xLimits = [[-1.4, -0.211], [-0.823, 2.192]]
        #yLimits = [[-2.603, -1.185], [-0.763, 0.47]]

        # maze 5 limits to create random target point in map
        xLimits = [[-2.787, 5.524], [-2.787, 5.524]]
        yLimits = [[-8.9, 1.676], [-8.9, 1.676]]

        targetX = random.uniform(*xLimits[random.randint(0,1)])
        targetY = random.uniform(*yLimits[random.randint(0,1)])

        myX = odomData.pose.pose.position.x
        myY = odomData.pose.pose.position.y

        self.newDistance = math.sqrt((targetX - myX)**2 + (targetY - myY)**2)

        if self.newDistance < 0.2:  # Reached to target
            done = True

        if self.oldDistance > self.newDistance:
            distanceAward = 2
        else:
            distanceAward = -2

        if isCrash:
            reward = -200
        elif done:
            # Reached to target
            rospy.logerr("Reached to target   x = " + str(targetX) + "   y = "+str(targetY))
            reward = 200
        else:
            # Negative reward for distance
            reward = self.oldDistance - self.newDistance

        self.oldDistance = self.newDistance

        return np.asarray(state).reshape(1,180,1), np.asarray([targetX - myX, targetY - myY]), reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except Exception:
            print ("/gazebo/reset_simulation service call failed")

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

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except Exception:
            print ("/gazebo/pause_physics service call failed")

        state, isCrash = self.calculate_observation(laserData)

        return np.asarray(state).reshape(1,180,1), np.asarray([0, 0])
