import rospy
import roslaunch
import time
import numpy as np

from random import randint

from gym import spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboMantisDQLearnEnv():

    def __init__(self):
        # Initialize the node
        rospy.init_node('gazebo_dqlearning_node', anonymous=True)
        # Connect to gazebo
        self.vel_pub = rospy.Publisher('/diffdrive/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.laser_point_count = 360  # 360 laser point in one time
        self.min_crash_range = 0.2  # Make done = True if agent close to wall 
        self.action_space = 3  # F,L,R

    def calculate_observation(self,data):
        # Detect crash
        done = False
        for i, item in enumerate(data.ranges):
            if (self.min_crash_range > data.ranges[i] > 0):
                done = True
        return data.ranges,done

    def randomAction(self):
        # return an action from 0 to 2 randomly
        return randint(0,2)

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except Exception:
            print("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 1.0
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = -1.0
            self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/mantis/base_scan', LaserScan, timeout=5)
            except Exception:
                print("/mantis/base_scanError laser data is empty!")
                time.sleep(5)
            
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except Exception:
            print("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 4
        else:
            reward = -200

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
                data = rospy.wait_for_message('/mantis/base_scan', LaserScan, timeout=5)
            except Exception:
                print("/mantis/base_scan Error laser data is empty!")
                time.sleep(5)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except Exception:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.calculate_observation(data)

        return np.asarray(state)
