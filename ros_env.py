# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:12:40 2017

@author: thy
"""
import rospy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.msg import LinkStates
from tf.transformations import euler_from_quaternion

import math
import random
import subprocess
import numpy as np
from scipy.spatial.distance import euclidean 

SOURCE_PATH = "/home/thy/ros_workspaces/adabot_ws/devel_debug/setup.bash"

class RosEnv(object):
    
    def link_states_callback(self, data):
        #RETURN X1, X2, AND euler z
        self.pos_x = data.pose[data.name.index('adabot::front_left_wheel')].position.x
        self.pos_y = data.pose[data.name.index('adabot::front_left_wheel')].position.y
        orient = data.pose[data.name.index('adabot::front_left_wheel')].orientation
        quart  = (orient.x , orient.y, orient.z, orient.w)
        self.euler = euler_from_quaternion(quart)
        y_path , x_path = [self.goal[1] - self.pos_y, self.goal[0] - self.pos_x]
        self.yaw_path =  math.atan2(y_path, x_path)
        #if record_state:
#            print '\nposition.x: {0}\nsim_time: {1}\n'.format(
#                data.pose[data.name.index('adabot::base_link')].position.x,
#                current_second)
#            sys.stdout.flush()
#    
#            evals_completed += 1
#            record_state = False
#    
#            if evals_completed == NUM_EVALS:
#                # Clean up and shutdown this node (delay so that messages finish)
#                time.sleep(0.25)
#                rospy.signal_shutdown('Simulation should end.')
#            else:
#                # Wait for the simulation to start
#                rospy.wait_for_service('/gazebo/reset_world')
#                try:
#                    rospy.ServiceProxy('/gazebo/reset_world', Empty)()
#                except rospy.ServiceException, e:
#                    print "Service call failed: %s" %e
#                    exit(1)
                # AJC TODO: service call to reset pose (needed for positioning)
                # but not for velocity


    def clock_callback(self, data):
        """ Subscriber to /clock
    
        This callback is responsible for getting the current simulation
        time in seconds.
        """
        global current_second
        current_second = data.clock.secs

    

    def __init__(self,name, PUB_RATE, goal, source = False):
        if source:
           #This is for if gazebo not up and running)
           source_call = 'source ' + SOURCE_PATH
           world_call = 'roslaunch adabot_gazebo adabot.world.launch world:=rocky3'
           subprocess.call([source_call, world_call], shell = True)
        self.state = 0
        self.pos_x = 0
        self.pos_y = 0
        self.goal = goal
        self.euler = [0, 0, 0]
        self.timestep = 0
        self.threshold_dist = 1.0
        self.threshold_time = round(self.Distance() * 200)
        
        rospy.init_node('adabot_env', anonymous=True)
        #self.reset() #Get it started
        rospy.wait_for_service('/gazebo/unpause_physics')
        rospy.Subscriber('clock', Clock, self.clock_callback)
        rospy.Subscriber('gazebo/link_states', LinkStates, self.link_states_callback)
                
        while True:
            # print('spinning')
            try:
                self.wheel_radius = rospy.get_param('wh_radius')
                self.wegs_per_wheel = rospy.get_param('wg_per_wheel')
                self.axle_radius = rospy.get_param('ax_radius')
                break
            except KeyError as e:
                pass
    
        # Create topic lists
        self.l_wheel_topics = [
            '/adabot/front_left_wheel_joint_velocity_controller/command',
            '/adabot/rear_left_wheel_joint_velocity_controller/command',
        ]
        self.r_wheel_topics = [
            '/adabot/front_right_wheel_joint_velocity_controller/command',
            '/adabot/rear_right_wheel_joint_velocity_controller/command',
        ]
    
        self.l_weg_topics = []
        self.r_weg_topics = []
        for w in range(1, self.wegs_per_wheel + 1):
            self.l_weg_topics.extend([
                '/adabot/front_left_weg_' + str(w) + '_joint_position_controller/command',
                '/adabot/rear_left_weg_' + str(w) + '_joint_position_controller/command',
                ])
            self.r_weg_topics.extend([
                '/adabot/front_right_weg_' + str(w) + '_joint_position_controller/command',
                '/adabot/rear_right_weg_' + str(w) + '_joint_position_controller/command',
                ])
        # Create publishers for all wheel and weg motors 
        self.l_weg_publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.l_weg_topics]
        self.r_weg_publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.r_weg_topics]
        self.l_wheel_publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.l_wheel_topics]
        self.r_wheel_publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.r_wheel_topics]
        self.pub_rate_hz = 10 # Don't know what this for
        
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            rospy.ServiceProxy('/gazebo/unpause_physics', Empty)()
        except rospy.ServiceException, e:
            print "Service call failed: %s" %e
            exit(1)    
        self.rate = rospy.Rate(PUB_RATE)
        #Intializing intial twist?
        
        self.l_wh = 0
        self.r_wh = 0
        self.l_weg = 0
        self.r_weg = 0
        
        self.wheel_space  = 0.05
        self.weg_space = 0.01
        
        self.updateLWheel()
        self.updateRWheel()
        self.updateLWeg()
        self.updateRWeg()
        self.distance = self.Distance()
        
    def get_stateVec_raw(self):
        return [self.pos_x, self.pos_y, self.euler]
    def get_stateVec(self):
        return self.yaw_path, self.euler[-1], self.yaw_path - float(self.euler[-1]),self.l_wh, self.r_wh
    def reset(self):
        self.l_wh = 0
        self.r_wh = 0
        self.l_weg = 0
        self.r_weg = 0
        self.timestep = 0
        
        print("Pause")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            rospy.ServiceProxy('/gazebo/pause_physics', Empty)()
        except rospy.ServiceException, e:
            print "Service call failed: %s" %e
            exit(1)
        rospy.wait_for_service('/gazebo/reset_simulation')
        print("Reset")
        try:
            rospy.ServiceProxy('/gazebo/reset_simulation', Empty)()
        except rospy.ServiceException, e:
            print "Service call failed: %s" %e
            exit(1)
        print("Unpause")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            rospy.ServiceProxy('/gazebo/unpause_physics', Empty)()
        except rospy.ServiceException, e:
            print "Service call failed: %s" %e
            exit(1)
        return self.get_stateVec()
    def action(self,a):
        self.timestep = self.timestep + 1
        print(self.timestep)
        
        #We have 4 components to control left right wheel and wegs
        #This means action space of 8 or 9(if action is none)
        done = False
        a = a + 1
        A = a / 2
        direction = 1 if a % 2 == 1 else -1
        if A == 0:
            #Keep the same setting
            pass
        
        elif A == 1:
            self.l_wh = self.l_wh + direction * self.wheel_space
            self.l_wh = np.clip(self.l_wh, -1 ,1)
            self.updateLWheel()
        elif A == 2 :
            self.r_wh = self.r_wh + direction * self.wheel_space
            self.r_wh = np.clip(self.r_wh, -1 ,1)
            self.updateRWheel()
        elif A == 3:
            self.l_wh = abs(self.r_wh)
            self.r_wh = abs(self.l_wh)
            target = np.max([self.l_wh, self.r_wh])
            self.l_wh = target
            self.r_wh = target
            self.updateLWheel()
            self.updateRWheel
            
        new_distance = self.Distance()
        if new_distance < self.threshold_dist or self.timestep > self.threshold_time :
            done = True
        print(new_distance)
        solve = False if new_distance > self.threshold_dist else True
        diff = new_distance - self.distance
        if diff < 0:
            self.distance = new_distance
            
        return -diff , self.get_stateVec(),  done, solve
            
    
    def updateLWheel(self):
        self.l_rate = self.l_wh / self.wheel_radius
        for p in self.l_wheel_publishers:
                p.publish(Float64(self.l_rate))
        return self.l_rate
    
    def updateRWheel(self):
        self.r_rate = self.r_wh / self.wheel_radius
        for p in self.r_wheel_publishers:
                p.publish(Float64(self.r_rate))
        return self.r_rate
        
    def updateLWeg(self):
        self.l_extension = self.l_weg / (self.wheel_radius - self.axle_radius)
        for p in self.l_weg_publishers:
                p.publish(Float64(self.l_extension))
        return self.l_extension
    
    def updateRWeg(self):
        self.r_extension = self.r_weg / (self.wheel_radius - self.axle_radius)
        for p in self.r_weg_publishers:
                p.publish(Float64(self.r_extension))
        return self.r_extension
        
    def sampleAction(self):
        return random.choice(list(range(10)))
        
    def StateSpace(self):
        return range(6)
    
    def Distance(self):
        return euclidean([self.pos_x, self.pos_y], self.goal)
    
    def AdjustThreshold(self):
        self.threshold_time = round(self.Distance() * 200)
        self.distance = self.Distance()
