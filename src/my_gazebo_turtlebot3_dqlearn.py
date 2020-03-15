import rospy
import roslaunch
import time
import numpy as np
import math

import random

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import tf
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty


class MantisGymEnv():

    def __init__(self):
        # Initialize the node
        rospy.init_node('mantis_gym_env', anonymous=True)
        # Connect to gazebo
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy(
            '/gazebo/reset_simulation', Empty)

        self.laserPointCount = 360  # 180 laser point in one time
        self.minCrashRange = 0.2  # Asume crash below this distance
        self.laserMinRange = 0.2
        self.laserMaxRange = 10.0
        self.stateSize = 362
        self.actionSize = 5

        self.targetDistance = 0  # Distance to target

        self.targetPointX = None
        self.targetPointY = None

        # Means robot reached target point. True at beginning to calc random point in reset func
        self.isTargetReached = True

    def pauseGazebo(self):
        # Pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except Exception:
            print("/gazebo/pause_physics service call failed")

    def unpauseGazebo(self):
        # Unpause the simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except Exception:
            print("/gazebo/unpause_physics service call failed")

    def resetGazebo(self):
        # Reset simualtion to initial phase
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except Exception:
            print("/gazebo/reset_simulation service call failed")

    def calcTargetPoint(self):
        # Calc target point
        # maze 2 limits
        #xLimits = [[-1.4, -0.211], [-0.823, 2.192]]
        #yLimits = [[-2.603, -1.185], [-0.763, 0.47]]

        # maze 5 limits to create random target point in map
        #xLimits = [[-2, 5], [4, 5]]
        #yLimits = [[-7, -8], [-8, 1]]

        # stage 4 limits to create random target point in map
        # xLimits = [[0, 1], [0, 1], [1.5, 2], [-0.7, -1.2], [-2, -1], [1, 2]]
        # yLimits = [[1, 2], [-2, -1], [-2, 0], [-2, -1], [0.8, 1.2], [0.8, 1.2]]

        # stage 1 limits to create random target point in map
        xLimits = [[-1.8, 1.8], [-1.8, 1.8], [0.8, 1.8], [-1.8, -0.8]]
        yLimits = [[0.8, 1.8], [-1.8, -0.8], [-1.8, 1.8], [-1.8, 1.8]]

        randomInt = random.randint(0, len(xLimits)-1)

        self.targetPointX = random.uniform(*xLimits[randomInt])
        self.targetPointY = random.uniform(*yLimits[randomInt])

        rospy.logerr("New Target Point (" + str(self.targetPointX) +
                     ", " + str(self.targetPointY) + ")")

    def getLaserData(self):
        # Return laser 2D array of robot scanner
        try:
            laserData = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            return laserData
        except Exception as e:
            rospy.logerr("Error to get laser data " + str(e))

    def getOdomData(self):
        # Return yaw, robotX, robotY of robot
        try:
            odomData = rospy.wait_for_message('/odom', Odometry, timeout=5)
            odomData = odomData.pose.pose
            quat = odomData.orientation
            quatTuple = (
                quat.x,
                quat.y,
                quat.z,
                quat.w,
            )
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                quatTuple)
            robotX = odomData.position.x
            robotY = odomData.position.y
            return yaw, robotX, robotY
        except Exception as e:
            rospy.logerr("Error to get odom data " + str(e))

    def calcHeadingAngle(self, targetPointX, targetPointY, yaw, robotX, robotY):
        # Return heading angle from robot to target
        targetAngle = math.atan2(targetPointY - robotY, targetPointX - robotX)

        heading = targetAngle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        return round(heading, 2)

    def calcDistance(self, x1, y1, x2, y2):
        # Return euler distance of two point in 2D space
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def calculateState(self, laserData, odomData):
        # Calculate and return state
        heading = self.calcHeadingAngle(
            self.targetPointX, self.targetPointY, *odomData)
        _, robotX, robotY = odomData
        distance = self.calcDistance(
            robotX, robotY, self.targetPointX, self.targetPointY)

        isCrash = False  # If robot hit to an obstacle
        laserData = list(laserData.ranges)
        for i in range(len(laserData)):
            if (self.laserMinRange > laserData[i] > 0):
                isCrash = True
            if np.isinf(laserData[i]):
                laserData[i] = self.laserMaxRange
            if np.isnan(laserData[i]):
                laserData[i] = 0
        return laserData + [heading, distance], isCrash

    def step(self, action):
        self.unpauseGazebo()

        # Move
        maxAngularVel = 1.5
        angVel = ((self.actionSize - 1)/2 - action) * maxAngularVel / 2

        velCmd = Twist()
        velCmd.linear.x = 0.15
        velCmd.angular.z = angVel
        self.velPub.publish(velCmd)

        # Observe
        laserData = self.getLaserData()
        odomData = self.getOdomData()

        self.pauseGazebo()

        state, isCrash = self.calculateState(laserData, odomData)

        done = False
        if isCrash:
            done = True

        distanceToTarget = state[-1]

        if distanceToTarget < 0.2:  # Reached to target
            done = True

        if isCrash:
            reward = -200
        elif done:
            # Reached to target
            rospy.logerr("Reached to target!")
            reward = 200
            self.isTargetReached = True
        else:
            yawReward = []
            currentDistance = state[-1]
            heading = state[-2]

            for i in range(5):
                angle = -math.pi / 4 + heading + \
                    (math.pi / 8 * i) + math.pi / 2
                tr = 1 - 4 * \
                    math.fabs(0.5 - math.modf(0.25 + 0.5 * angle %
                                              (2 * math.pi) / math.pi)[0])
                yawReward.append(tr)

            distanceRate = 2 ** (currentDistance / self.targetDistance)
            reward = ((round(yawReward[action] * 5, 2)) * distanceRate)

        return np.asarray(state), reward, done

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        self.resetGazebo()

        if self.isTargetReached:
            self.calcTargetPoint()
            self.isTargetReached = False

        # Unpause simulation to make observation
        self.unpauseGazebo()
        laserData = self.getLaserData()
        odomData = self.getOdomData()
        self.pauseGazebo()

        state, isCrash = self.calculateState(laserData, odomData)
        self.targetDistance = state[-1]
        self.stateSize = len(state)

        return np.asarray(state)
