#!/usr/bin/env python

from __future__ import division
import rospy
import random
import math
import numpy as np
from std_msgs.msg import Bool,String,Float32
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import LaserScan
from read_config import read_config
from helper_functions import get_pose
from helper_functions import move_function
from map_utils import Map
from scipy import spatial 
from sklearn.neighbors import KDTree             

def GaussianPDF(value, mean, sd):
	return (np.exp(-1.0 * ((value - mean) ** 2.0) / (2.0 * (sd ** 2.0))))
class Robot():     
    def __init__(self):
	self.config = read_config()
        rospy.init_node("robot")
	self.rate = rospy.Rate(1)
	self.rate.sleep()
	self.result_publisher = rospy.Publisher(
		"/result_update",
		Bool,
		queue_size = 10
	)
	self.complete_publisher = rospy.Publisher(
		"/sim_complete",
		Bool,
		queue_size = 10
	)
	self.poseArray_publisher = rospy.Publisher(
		"/particlecloud",
		PoseArray,
		queue_size = 10
	)
	self.likelihood_publisher = rospy.Publisher(
		"/likelihood_field",
		OccupancyGrid,
		queue_size = 10,
		latch = True
	)
	self.map_service = rospy.Subscriber(            
		"/map",
		OccupancyGrid,
		self.handle_map_message
	)
	self.laser_service = rospy.Subscriber(            
		"/base_scan",
		LaserScan,
		self.handle_laser_message
	)
	self.initialized = False
	self.mess = Bool()
	self.mess.data = True
	self.originX = 0
	self.originY = 0
	self.width = 0
	self.height = 0
	self.resolution = 0
	self.numPart = self.config['num_particles']
	self.laser_sigma_hit = self.config['laser_sigma_hit']
	self.laser_z_hit = self.config['laser_z_hit']
	self.laser_z_rand = self.config['laser_z_rand']
	self.move_list = self.config['move_list']
	self.first_move_sigma_x = self.config['first_move_sigma_x']
	self.first_move_sigma_y = self.config['first_move_sigma_y']
	self.first_move_sigma_angle = self.config['first_move_sigma_angle']
	self.resample_sigma_x = self.config['resample_sigma_x']
	self.resample_sigma_y = self.config['resample_sigma_y']
	self.resample_sigma_angle = self.config['resample_sigma_angle']
	self.mapLH = None
	self.mapOri = None
	self.particleX = []
	self.particleY = []
	self.particleTheta = []
	self.particleWeight = []
	self.pose_array = None
	self.initializeParticle()
	self.move()
	rospy.spin()

    def get_nearest_obstacle(self, x, y, k=1):
	# Check if KDTree is initialized
	if not hasattr(self, 'obstacle_tree'):
		self._initialize_obstacle_KDTree()

	# Query for the given location
	dist, ind = self.obstacle_tree.query((x, y), k)
		
	# Transform index to actual locations
	obstacles = self.obstacle_tree.get_arrays()[0]
	obst = [obstacles[i] for i in ind]
		
	return dist, obst

   


    def handle_map_message(self, message):
	if(self.initialized == False):
		self.resolution = message.info.resolution
		self.origin_x = message.info.origin.position.x
        	self.origin_y = message.info.origin.position.y
		self.width = message.info.width
		self.height = message.info.height
		self.pose_array = PoseArray()
		self.pose_array.header.stamp = rospy.Time.now()
		self.pose_array.header.frame_id = 'map'
		self.pose_array.poses = []
		self.mapOri = Map(message)
		self.mapLH = Map(message)
		self.initialized = True
    def initializeParticle(self):
	self.rate.sleep()
	for i in range(self.numPart):
		
		x = random.uniform(0,self.width)
		y = random.uniform(0,self.height)
				
		while(self.mapOri.get_cell(x,y) != 0):                            
			x = random.uniform(0,self.width)
			y = random.uniform(0,self.height)
			
		self.particleX.append(x)
		self.particleY.append(y)
		self.particleTheta.append(random.uniform(0,2*np.pi)) 
		self.particleWeight.append(1/self.numPart)
		self.pose_array.poses.append(get_pose(self.particleX[i],self.particleY[i],self.particleTheta[i]))
	self.poseArray_publisher.publish(self.pose_array)
	obstalces = filter(lambda x: self.mapOri.get_cell(x[0], x[1]) != 0.0, [(x, y) for x in range(self.width) for y in range(self.height)])
	self.obstacle_tree = KDTree(obstalces, leaf_size=20)
	self.mapLH.grid = np.array([[GaussianPDF(self.get_nearest_obstacle(x,y)[0][0][0], 0, self.laser_sigma_hit)
							for x in range(self.width)]
						for y in range(self.height)])
	self.likelihood_publisher.publish(self.mapLH.to_message())

    def move(self):
	first_move = True
	for moveStep in self.move_list:
		angle = moveStep[0]
		dist = moveStep[1]
		num = moveStep[2]
		print(angle)
		print(dist)
		print(num)
		if(first_move == True):
			move_function(angle,0)
			wTot = 0
			wSoFar = 0
			resampleProb = []
			resampleX = []
			resampleY = []
			resampleAngle = []
			resampleWeight = []
			for i in range(self.numPart):
				self.particleTheta[i] = self.particleTheta[i]+(angle / 180.0 * self.angle_max)+random.gauss(0,self.first_move_sigma_angle)
				self.particleX[i] = (self.particleX[i]+random.gauss(0,self.first_move_sigma_x))
				self.particleY[i] = (self.particleY[i]+random.gauss(0,self.first_move_sigma_y))
				particle_poss = self.mapLH.get_cell(self.particleX[i], self.particleY[i])
				if math.isnan(particle_poss) or particle_poss == 1.0:
					self.particleWeight[i] = 0.0  
				wTot += self.particleWeight[i]
			for i in range(self.numPart):
				self.particleWeight[i] = self.particleWeight[i]/wTot
				wSoFar += self.particleWeight[i]
				resampleProb.append(wSoFar)
			self.pose_array.header.stamp = rospy.Time.now()
			self.pose_array.poses = []
			for i in range(self.numPart):
				re = random.random()
				index = -1
				for j in range(self.numPart):
					if(re<=resampleProb[j]):
						index = j
						break
				resampleX.append(self.particleX[index]+random.gauss(0,self.resample_sigma_x))
				resampleY.append(self.particleY[index]+random.gauss(0,self.resample_sigma_y))
				resampleAngle.append(self.particleTheta[index]+random.gauss(0,self.resample_sigma_angle))
				resampleWeight.append(self.particleWeight[index])
				self.pose_array.poses.append(get_pose(resampleX[i],resampleY[i],resampleAngle[i]))
			self.particleX = resampleX
			self.particleY = resampleY
			self.particleTheta = resampleAngle
			self.particleWeight = resampleWeight
			self.poseArray_publisher.publish(self.pose_array)
						
			
			for manyTimes in range(num): 
				move_function(0,dist)
				wTot = 0
				wSoFar = 0
				resampleProb = []
				resampleX = []
				resampleY = []
				resampleAngle = []
				resampleWeight = []
				for i in range(self.numPart):
					self.particleX[i] = self.particleX[i]+random.gauss(0,self.first_move_sigma_x)
					self.particleY[i] = self.particleY[i]+random.gauss(0,self.first_move_sigma_y)
					self.particleTheta[i] = self.particleTheta[i]+random.gauss(0,self.first_move_sigma_angle)
					particle_poss = self.mapLH.get_cell(self.particleX[i], self.particleY[i])
					if math.isnan(particle_poss) or particle_poss == 1.0:
						self.particleWeight[i] = 0.0  
					else:
						self.particleX[i] = self.particleX[i]+math.cos(self.particleTheta[i])*dist
						self.particleY[i] = self.particleY[i]+math.sin(self.particleTheta[i])*dist
						index = 0
						pTot = 1
						for z in self.ranges:
							laserAngle = self.angle_min + self.angle_incr*index
							index+=1
							if(z>=self.range_min and z<=self.range_max):
								xz = self.particleX[i]+math.cos(self.particleTheta[i]+laserAngle)*z
								yz = self.particleY[i]+math.sin(self.particleTheta[i]+laserAngle)*z
								prob = self.mapLH.get_cell(xz,yz)
								if(math.isnan(prob)):
									prob = self.laser_z_rand
								else:
									prob = self.laser_z_hit * prob + self.laser_z_rand
								pTot *= prob
						self.particleWeight[i] = self.particleWeight[i]*(pTot+0.001) 
					wTot += self.particleWeight[i]
				for i in range(self.numPart):
					self.particleWeight[i] = self.particleWeight[i]/wTot
					wSoFar += self.particleWeight[i]
					resampleProb.append(wSoFar)
				self.pose_array.header.stamp = rospy.Time.now()
				self.pose_array.poses = []
				for i in range(self.numPart):
					re = random.random()
					index = -1
					for j in range(self.numPart):
						if(re<=resampleProb[j]):
							index = j
							break
					resampleX.append(self.particleX[index]+random.gauss(0,self.resample_sigma_x))
					resampleY.append(self.particleY[index]+random.gauss(0,self.resample_sigma_y))
					resampleAngle.append(self.particleTheta[index]+random.gauss(0,self.resample_sigma_angle))
					resampleWeight.append(self.particleWeight[index])
					self.pose_array.poses.append(get_pose(resampleX[i],resampleY[i],resampleAngle[i]))
				self.particleX = resampleX
				self.particleY = resampleY
				self.particleTheta = resampleAngle
				self.particleWeight = resampleWeight
				self.poseArray_publisher.publish(self.pose_array)	 	

			first_move = False
		else:
			move_function(angle,0)
			wTot = 0
			wSoFar = 0
			resampleProb = []
			resampleX = []
			resampleY = []
			resampleAngle = []
			resampleWeight = []
			for i in range(self.numPart):
				self.particleTheta[i] = self.particleTheta[i]+(angle / 180.0 * self.angle_max)
				particle_poss = self.mapLH.get_cell(self.particleX[i], self.particleY[i])
				if math.isnan(particle_poss) or particle_poss == 1.0:
					self.particleWeight[i] = 0.0  
				wTot += self.particleWeight[i]
			for i in range(self.numPart):
				self.particleWeight[i] = self.particleWeight[i]/wTot
				wSoFar += self.particleWeight[i]
				resampleProb.append(wSoFar)
			self.pose_array.header.stamp = rospy.Time.now()
			self.pose_array.poses = []
			for i in range(self.numPart):
				re = random.random()
				index = -1
				for j in range(self.numPart):
					if(re<=resampleProb[j]):
						index = j
						break
				resampleX.append(self.particleX[index]+random.gauss(0,self.resample_sigma_x))
				resampleY.append(self.particleY[index]+random.gauss(0,self.resample_sigma_y))
				resampleAngle.append(self.particleTheta[index]+random.gauss(0,self.resample_sigma_angle))
				resampleWeight.append(self.particleWeight[index])
				self.pose_array.poses.append(get_pose(resampleX[i],resampleY[i],resampleAngle[i]))
			self.particleX = resampleX
			self.particleY = resampleY
			self.particleTheta = resampleAngle
			self.particleWeight = resampleWeight
			self.poseArray_publisher.publish(self.pose_array)
						
			
			for manyTimes in range(num): 
				move_function(0,dist)
				wTot = 0
				wSoFar = 0
				resampleProb = []
				resampleX = []
				resampleY = []
				resampleAngle = []
				resampleWeight = []
				for i in range(self.numPart):
					particle_poss = self.mapLH.get_cell(self.particleX[i], self.particleY[i])
					if math.isnan(particle_poss) or particle_poss == 1.0:
						self.particleWeight[i] = 0.0  
					else:
						self.particleX[i] = (self.particleX[i]+math.cos(self.particleTheta[i])*dist)
						self.particleY[i] = (self.particleY[i]+math.sin(self.particleTheta[i])*dist) 
						index = 0
						pTot = 1
						for z in self.ranges:
							laserAngle = self.angle_min + self.angle_incr*index
							index+=1
							if(z>=self.range_min and z<=self.range_max):
								xz = self.particleX[i]+math.cos(self.particleTheta[i]+laserAngle)*z
								yz = self.particleY[i]+math.sin(self.particleTheta[i]+laserAngle)*z
								prob = self.mapLH.get_cell(xz,yz)
								if(math.isnan(prob)):
									prob = self.laser_z_rand
								else:
									prob = self.laser_z_hit * prob + self.laser_z_rand
								pTot *= prob
						self.particleWeight[i] = self.particleWeight[i]*(-pTot+0.001)
					wTot += self.particleWeight[i]
				for i in range(self.numPart):
					self.particleWeight[i] = self.particleWeight[i]/wTot
					wSoFar += self.particleWeight[i]
					resampleProb.append(wSoFar)
				self.pose_array.header.stamp = rospy.Time.now()
				self.pose_array.poses = []
				for i in range(self.numPart):
					re = random.random()
					index = -1
					for j in range(self.numPart):
						if(re<=resampleProb[j]):
							index = j
							break
					resampleX.append(self.particleX[index]+random.gauss(0,self.resample_sigma_x))
					resampleY.append(self.particleY[index]+random.gauss(0,self.resample_sigma_y))
					resampleAngle.append(self.particleTheta[index]+random.gauss(0,self.resample_sigma_angle))
					resampleWeight.append(self.particleWeight[index])
					self.pose_array.poses.append(get_pose(resampleX[i],resampleY[i],resampleAngle[i]))
				self.particleX = resampleX
				self.particleY = resampleY
				self.particleTheta = resampleAngle
				self.particleWeight = resampleWeight
				self.poseArray_publisher.publish(self.pose_array)			
		self.result_publisher.publish(self.mess)
	self.complete_publisher.publish(self.mess)

	
    def handle_laser_message(self,message):
	self.angle_min = message.angle_min
	self.angle_max = message.angle_max
	self.angle_incr = message.angle_increment
	self.range_min = message.range_min
	self.range_max = message.range_max
	self.ranges = message.ranges	
	
	
			

if __name__ == '__main__':
    robot = Robot()
