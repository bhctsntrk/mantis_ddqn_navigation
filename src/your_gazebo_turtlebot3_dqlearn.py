import rospy
import roslaunch
import time
import numpy as np
import math

import random

from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

from nav_msgs.msg import Odometry
import tf
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

class AgentPosController():
    # This class teleport agent when env.rest called
    # Because we dont want the agent spawn same center place everytime
    def __init__(self):
        self.agent_model_name = "turtlebot3_waffle"

    def teleportRandom(self):
        # Teleport agent return new x and y point
        model_state_msg = ModelState()

        model_state_msg.model_name = self.agent_model_name
        
        # maze 1
        xy_list = [
            [-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
            [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
            [5.5,-1.5], [5.5,-0.5], [5.5,0.5], [5.5,1.5]
        ]
        '''
        # maze 2
        xy_list = [
            [-1.5,-1.5], [-0.5,-1.5], [-1.5,-0.5], 
            [-0.5,1.5], [1.5,0.5], 
            [2.5,2.5], [2.5,3.5], [1.5,3.5], 
        ]

        # maze 3
        xy_list = [
            [0,0], 
            [0.5,0.5], [1.5,0.5], [0.5,1.5], [1.5,1.5],
            [-0.5,-0.5], [-1.5,-0.5], [-1.5,-1.5],
            [0.5,-0.5], [0.5,-1.5], [1.5,-1.5],
            [-0.5,0.5], [-1.5,0.5], [-0.5,1.5], [-1.5,1.5],
        ]
        '''
        
        pose = Pose()
        pose.position.x, pose.position.y = random.choice(xy_list)

        model_state_msg.pose = pose
        model_state_msg.twist = Twist()

        model_state_msg.reference_frame = "world"

        isTeleportSuccess = False
        for i in range(5):
            if not isTeleportSuccess:
                try:
                    rospy.wait_for_service('/gazebo/set_model_state')
                    telep_model_prox = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                    telep_model_prox(model_state_msg)
                    isTeleportSuccess = True
                    break
                except Exception as e:
                    rospy.logfatal("Error when teleporting agent " + str(e))
            else:
                rospy.logwarn("Trying to teleporting agent..." + str(i))
                time.sleep(2)
        
        if not isTeleportSuccess:
            rospy.logfatal("Error when teleporting agent")
            return "Err", "Err"
    
        return pose.position.x, pose.position.y

class GoalController():
    def __init__(self):
        self.model_path = "../models/gazebo/goal_sign/model.sdf"
        f = open(self.model_path, 'r')
        self.model = f.read()

        self.goal_position = Pose()
        self.goal_position.position.x = None  # Initial positions
        self.goal_position.position.y = None
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        self.model_name = 'goal_sign'
        self.check_model = False

    def respawnModel(self):
        # Spawn goal model
        isSpawnSuccess = False
        for i in range(5):
            if not self.check_model:
                try:
                    rospy.wait_for_service('gazebo/spawn_sdf_model')
                    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                    spawn_model_prox(self.model_name, self.model, 'robotos_name_space', self.goal_position, "world")
                    isSpawnSuccess = True
                    self.check_model = True
                    break
                except Exception as e:
                    rospy.logfatal("Error when spawning the goal sign " + str(e))
            else:
                rospy.logwarn("Trying to spawn goal sign ..." + str(i))
                time.sleep(2)
        
        if not isSpawnSuccess:
            rospy.logfatal("Error when spawning the goal sign")
        

    def deleteModel(self):
        # Delete goal model
        while True:
            if self.check_model:
                try:
                    rospy.wait_for_service('gazebo/delete_model')
                    del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                    del_model_prox(self.model_name)
                    self.check_model = False
                    break
                except Exception as e:
                    rospy.logfatal("Error when deleting the goal sign " + str(e))
            else:
                break

    def calcTargetPoint(self):
        self.deleteModel()


        # maze 1
        goal_xy_list = [
            [-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
            [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
            [5.5,-1.5], [5.5,-0.5], [5.5,0.5], [5.5,1.5]
        ]
        """
        # maze 2
        goal_xy_list = [
            [-1.5,-1.5], [-0.5,-1.5], [-1.5,-0.5], 
            [-0.5,1.5], [1.5,0.5], 
            [2.5,2.5], [2.5,3.5], [1.5,3.5], 
        ]
        
        # maze 3
        goal_xy_list = [
            [0,0], 
            [0.5,0.5], [1.5,0.5], [0.5,1.5], [1.5,1.5],
            [-0.5,-0.5], [-1.5,-0.5], [-1.5,-1.5],
            [0.5,-0.5], [0.5,-1.5], [1.5,-1.5],
            [-0.5,0.5], [-1.5,0.5], [-0.5,1.5], [-1.5,1.5],
        ]
        
        # maze3-maddness
        goal_xy_list = [
            [0,0], [-0.5,1.5], [0.5,1.5]
        ]
        """
        while True:
            self.goal_position.position.x, self.goal_position.position.y = random.choice(goal_xy_list)

            if self.last_goal_x != self.goal_position.position.x:
                if self.last_goal_y != self.goal_position.position.y:
                    break

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        rospy.logwarn("New goal position : " + str(self.goal_position.position.x) + " , " + str(self.goal_position.position.y))

        return self.goal_position.position.x, self.goal_position.position.y

    def getTargetPoint(self):
        return self.goal_position.position.x, self.goal_position.position.y


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

        self.laserPointCount = 24  # 24 laser point in one time
        self.minCrashRange = 0.2  # Asume crash below this distance
        self.laserMinRange = 0.2
        self.laserMaxRange = 10.0
        self.stateSize = self.laserPointCount + 4  # Laser(arr), heading, distance, obstacleMinRange, obstacleAngle
        self.actionSize = 5

        self.targetDistance = 0  # Distance to target

        self.targetPointX = 0
        self.targetPointY = 0

        # Means robot reached target point. True at beginning to calc random point in reset func
        self.isTargetReached = True
        self.goalCont = GoalController()
        self.agentController = AgentPosController()


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

    def getLaserData(self):
        # Return laser 2D array of robot scanner
        try:
            laserData = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            return laserData
        except Exception as e:
            rospy.logfatal("Error to get laser data " + str(e))

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
            rospy.logfatal("Error to get odom data " + str(e))

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
            if (self.minCrashRange > laserData[i] > 0):
                isCrash = True
            if np.isinf(laserData[i]):
                laserData[i] = self.laserMaxRange
            if np.isnan(laserData[i]):
                laserData[i] = 0

        obstacleMinRange = round(min(laserData), 2)
        obstacleAngle = np.argmin(laserData)

        return laserData + [heading, distance, obstacleMinRange, obstacleAngle], isCrash

    def step(self, action):
        self.unpauseGazebo()

        # Move

        maxAngularVel = 1.5
        angVel = ((self.actionSize - 1)/2 - action) * maxAngularVel / 2

        velCmd = Twist()
        velCmd.linear.x = 0.15
        velCmd.angular.z = angVel
        self.velPub.publish(velCmd)

        """
        if action == 0: #BRAKE LEFT
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = 1.6
            self.velPub.publish(velCmd)
        elif action == 1: #LEFT
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = 0.8
            self.velPub.publish(velCmd)
        elif action == 2: #FORWARD
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = 0.0
            self.velPub.publish(velCmd)
        elif action == 3: #RIGHT
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = -0.8
            self.velPub.publish(velCmd)
        elif action == 4: #BRAKE RIGHT
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = -1.6
            self.velPub.publish(velCmd)       
        """
        # Observe
        laserData = self.getLaserData()
        odomData = self.getOdomData()

        self.pauseGazebo()

        state, isCrash = self.calculateState(laserData, odomData)

        done = False
        if isCrash:
            done = True

        distanceToTarget = state[-3]

        if distanceToTarget < 0.2:  # Reached to target
            self.isTargetReached = True

        if isCrash:
            reward = -150 # Default val -150
        elif self.isTargetReached:
            # Reached to target
            rospy.logwarn("Reached to target!")
            reward = 200
            self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint()
            self.isTargetReached = False
        else:
            yawReward = []
            currentDistance = state[-3]
            heading = state[-4]

            for i in range(self.actionSize):
                angle = -math.pi / 4 + heading + (math.pi / 8 * i) + math.pi / 2
                tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
                yawReward.append(tr)

            try:
                distanceRate = 2 ** (currentDistance / self.targetDistance)
            except Exception:
                print("Overflow err CurrentDistance = ", currentDistance, " TargetDistance = ", self.targetDistance)
                distanceRate = 2 ** (currentDistance // self.targetDistance)
                
            reward = ((round(yawReward[action] * 5, 2)) * distanceRate)

        return np.asarray(state), reward, done

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        self.resetGazebo()

        while True:
            # Teleport turtlebot to a random point
            agentX, agentY = self.agentController.teleportRandom()
            if self.calcDistance(self.targetPointX, self.targetPointY, agentX, agentY) > self.minCrashRange:
                break
            else:
                rospy.logerr("Re teleporting the turtlebot!")
                time.sleep(2)

        if self.isTargetReached:
            while True:
                self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint()
                if self.calcDistance(self.targetPointX, self.targetPointY, agentX, agentY) > self.minCrashRange:
                    self.isTargetReached = False
                    break
                else:
                    rospy.logerr("Re calculating target point!")
                    time.sleep(2)

        # Unpause simulation to make observation
        self.unpauseGazebo()
        laserData = self.getLaserData()
        odomData = self.getOdomData()
        self.pauseGazebo()

        state, isCrash = self.calculateState(laserData, odomData)
        self.targetDistance = state[-3]
        self.stateSize = len(state)

        return np.asarray(state)