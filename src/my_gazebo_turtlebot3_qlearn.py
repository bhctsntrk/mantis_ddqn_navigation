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

    def discretize_observation(self,data):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = 9
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges,done

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
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except Exception:
                print("/scan Error laser data is empty!")
                time.sleep(5)
            
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except Exception:
            print("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data)

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

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

        state, isCrash = self.discretize_observation(data)

        return state
