#!/usr/bin/env python

import rospy
import numpy as np
from cse_190_assi_3.msg import AStarPath,PolicyList
from std_msgs.msg import Bool
from read_config import read_config
from astar import *
from mdp import *

class Robot():
    def __init__(self):
	self.config = read_config()
	rospy.init_node("robot")
	self.map_y = (read_config()["map_size"][0])
	self.map_x = (read_config()["map_size"][1])
	self.move_list = (read_config()["move_list"])
	self.start = (read_config()["start"])
	self.goal = (read_config()["goal"])
	self.walls = (read_config()["walls"])
	self.pits = (read_config()["pits"])
	self.move_list = (read_config()["move_list"])
	self.max_iterations = (read_config()["max_iterations"])
	self.threshold_difference = (read_config()["threshold_difference"])
	self.reward_step = (read_config()["reward_for_each_step"])
	self.reward_wall = (read_config()["reward_for_hitting_wall"])
	self.reward_goal = (read_config()["reward_for_reaching_goal"])
	self.reward_pit = (read_config()["reward_for_falling_in_pit"])
	
	self.discount = (read_config()["discount_factor"])
	self.prob_forward = (read_config()["prob_move_forward"])
	self.prob_back = (read_config()["prob_move_backward"])
	self.prob_left = (read_config()["prob_move_left"])
	self.prob_right = (read_config()["prob_move_right"])
	self.mapV =[[0 for x in range(self.map_x)] for y in range(self.map_y)]
	self.mapVpre =[[0 for x in range(self.map_x)] for y in range(self.map_y)]
	self.mapV[self.goal[0]][self.goal[1]] = self.reward_goal
	self.mapVpre[self.goal[0]][self.goal[1]] = self.reward_goal
	for pit in self.pits:
	    self.mapV[pit[0]][pit[1]] = self.reward_pit
	    self.mapVpre[pit[0]][pit[1]] = self.reward_pit
	self.rate = rospy.Rate(1)
	self.astar_publisher = rospy.Publisher(
		"/results/path_list",
		AStarPath,
		queue_size = 10
	)
	self.mdp_publisher = rospy.Publisher(
		"/results/policy_list",
		PolicyList,
		queue_size = 10
	)
	self.sim_complete = rospy.Publisher(
		"/map_node/sim_complete",
		Bool,
		queue_size = 10
	)
	self.rate.sleep()
	self.path = getAstar()
	self.mess = AStarPath()
	self.mess1 = PolicyList()
	for step in self.path:
		self.mess.data = step
		self.astar_publisher.publish(self.mess)
		self.rate.sleep()
	self.count = 0
	while(self.count < self.max_iterations):
		for y in range(self.map_y):
			for x in range(self.map_x):
				self.mapVpre[y][x] = self.mapV[y][x]
		self.mapV,self.policy = getMdp(self.mapV)
		self.mess1.data = []
		for y in range(self.map_y):
			for x in range(self.map_x):
				self.mess1.data.append(self.policy[y][x])
		self.mdp_publisher.publish(self.mess1)
		self.check = 0
		for y in range(self.map_y):
			for x in range(self.map_x):
				self.check += abs(self.mapV[y][x]-self.mapVpre[y][x])
			
		if(self.check <= self.threshold_difference):
			break
		
		self.count +=1
	print(self.mapV)
	self.sim_complete.publish(Bool(True))
	rospy.signal_shutdown("complete")
	rospy.spin()

if __name__ == '__main__':
    robot = Robot()
