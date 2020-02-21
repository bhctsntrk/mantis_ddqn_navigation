import rospy
import roslaunch
import time
import numpy as np
import math
import random

from gym import spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

class GazeboTurtlebot3DQLearnEnv():

    def __init__(self):
        # Launch the simulation with the given launchfile name
        rospy.init_node('gazebo_dqlearning_node', anonymous=True)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.oldDistance = 0

    def calculate_observation(self,data):
        min_range = 0.2
        isCrash = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                isCrash = True
        return data.ranges,isCrash

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except Exception:
            print ("/gazebo/unpause_physics service call failed")

        max_ang_speed = 0.9
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.4
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

        if isCrash:
            done = True

        xLimits = [-1.4, 2.192]
        yLimits = [-2.603, 0.47]

        targetX = random.uniform(*xLimits)
        targetY = random.uniform(*yLimits)
        myX = odomData.pose.pose.position.x
        myY = odomData.pose.pose.position.y

        self.newDistance = math.sqrt((targetX - myX)**2 + (targetY - myY)**2)

        if self.newDistance < 0.2:
            done = True

        if self.oldDistance > self.newDistance:
            distanceAward = 2
        else:
            distanceAward = -2

        if isCrash:
            reward = -200
        elif done:
            # Reached to target
            rospy.logerr("uuuuuuuuuuuuuuuuuuuuuuuu")
            reward = 200
        else:
            # Negative reward for fooling around more negative reward for turn around
            reward = -abs(ang_vel) - 1

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

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except Exception:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except Exception:
                print("/scan Error laser data is empty!")
                time.sleep(5)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except Exception:
            print ("/gazebo/pause_physics service call failed")

        state, isCrash = self.calculate_observation(data)

        return np.asarray(state)
