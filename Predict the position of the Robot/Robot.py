#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Bool,String,Float32
from cse_190_assi_1.srv import requestTexture, moveService
from cse_190_assi_1.msg import temperatureMessage, RobotProbabilities
from read_config import read_config


class Robot():     
    def __init__(self):
	self.config = read_config()
        rospy.init_node("robot")
	self.temperature_publisher = rospy.Publisher(
		"/temp_sensor/activation",
		Bool,
		queue_size = 10
	)
	self.temp_publisher = rospy.Publisher(
		"/results/temperature_data",
		Float32,
		queue_size = 10
	)
	self.text_publisher = rospy.Publisher(
		"/results/texture_data",
		String,
		queue_size = 10
	)
	self.prob_publisher = rospy.Publisher(
		"/results/probabilities",
		RobotProbabilities,
		queue_size = 10
	)
	self.simulation_complete_pub = rospy.Publisher(
		"/map_node/sim_complete",
		Bool,
		queue_size = 10
	)
	self.temperature_service = rospy.Subscriber(
		"/temp_sensor/data",
		temperatureMessage,
		self.handle_temperature_message
	)
	self.texture_requester = rospy.ServiceProxy(
		"requestTexture",
		requestTexture
	)
	self.move_requester = rospy.ServiceProxy(
		"moveService",
		moveService
	)
	self.moveStep = 0
	self.num_rows = len(self.config['pipe_map'])
	self.num_cols = len(self.config['pipe_map'][0])
	self.position = [1.0/(float)(self.num_rows*self.num_cols)]*(self.num_rows*self.num_cols)
	rospy.sleep(1)
	self.temperature_publisher.publish(Bool(True))
	rospy.spin()

    def handle_temperature_message(self, message):
	temp = message.temperature
	self.temp_publisher.publish(temp)
	self.cal_gas(temp)
	tex_response = self.texture_requester()
	tex = tex_response.data
	self.cal_bay(tex)
	self.text_publisher.publish(tex)
	if(self.moveStep == len(self.config['move_list'])):
		self.prob_publisher.publish(self.position)
		self.simulation_complete_pub.publish(Bool(True))
		rospy.signal_shutdown("complete")
	self.move_requester(self.config['move_list'][self.moveStep])
	self.cal_move(self.config['move_list'][self.moveStep])
	self.prob_publisher.publish(self.position)
	self.moveStep += 1

    def cal_gas(self,temp):
	K = 0.0
	sigma = self.config['temp_noise_std_dev']
	for i in range(self.num_rows):
		for j in range(self.num_cols):
			if(self.config['pipe_map'][i][j] == 'H'):
				K+= (1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(temp-40.0)**2 / (2 * sigma**2) )) *self.position[i*self.num_cols+j]
			if(self.config['pipe_map'][i][j] == 'C'):
				K+= (1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(temp-20.0)**2 / (2 * sigma**2) )) *self.position[i*self.num_cols+j]
			if(self.config['pipe_map'][i][j] == '-'):
				K+= (1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(temp-25.0)**2 / (2 * sigma**2) )) *self.position[i*self.num_cols+j]
	for i in range(self.num_rows):
		for j in range(self.num_cols):
			if(self.config['pipe_map'][i][j] == 'H'):
				self.position[i*self.num_cols+j]= (1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(temp-40.0)**2 / (2 * sigma**2) )) *self.position[i*self.num_cols+j]/K
			if(self.config['pipe_map'][i][j] == 'C'):
				self.position[i*self.num_cols+j]= (1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(temp-20.0)**2 / (2 * sigma**2) )) *self.position[i*self.num_cols+j]/K
			if(self.config['pipe_map'][i][j] == '-'):
				self.position[i*self.num_cols+j]= (1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(temp-25.0)**2 / (2 * sigma**2) )) *self.position[i*self.num_cols+j]/K




    def cal_move(self,move):
	pos = [0.0]*(self.num_rows*self.num_cols)
	goDown = (1.0-self.config['prob_move_correct'])/4
	goUp = 	(1.0-self.config['prob_move_correct'])/4
	goLeft = (1.0-self.config['prob_move_correct'])/4
	goRight = (1.0-self.config['prob_move_correct'])/4
	stay = (1.0-self.config['prob_move_correct'])/4
	if(move == [0,0]):
		stay = self.config['prob_move_correct']
	if(move == [1,0]):
		goUp = self.config['prob_move_correct']
	if(move == [-1,0]):
		goDown = self.config['prob_move_correct']
	if(move == [0,1]):
		goRight = self.config['prob_move_correct']
	if(move == [0,-1]):
		goLeft = self.config['prob_move_correct']
	for i in range(self.num_rows):
		for j in range(self.num_cols):
			pos[i*self.num_cols+j] = self.position[i*self.num_cols+j]*stay+self.position[i*self.num_cols+(j+1)%(self.num_cols)]*goLeft+self.position[i*self.num_cols+(j-1)%(self.num_cols)]*goRight+self.position[((i+1)%(self.num_rows))*self.num_cols+j]*goDown+self.position[((i-1)%(self.num_rows))*self.num_cols+j]*goUp
	for i in range(self.num_rows):
		for j in range(self.num_cols):
			self.position[i*self.num_cols+j] = pos[i*self.num_cols+j]


	
    def cal_bay(self,tex):
	K = 0.0
	for i in range(len(self.config['texture_map'])):
		for j in range(len(self.config['texture_map'][0])):
			if(tex == self.config['texture_map'][i][j]):
				K += self.config['prob_tex_correct']*self.position[i*(len(self.config['texture_map'][0]))+j]
			else:
				K += (1-self.config['prob_tex_correct'])*self.position[i*(len(self.config['texture_map'][0]))+j]					 
	for i in range(len(self.config['texture_map'])):
		for j in range(len(self.config['texture_map'][0])):
			if(tex == self.config['texture_map'][i][j]):
				self.position[i*(len(self.config['texture_map'][0]))+j]= (self.config['prob_tex_correct']*self.position[i*(len(self.config['texture_map'][0]))+j])/K
			else:
				self.position[i*(len(self.config['texture_map'][0]))+j]= ((1-self.config['prob_tex_correct'])*self.position[i*(len(self.config['texture_map'][0]))+j])/K


if __name__ == '__main__':
    robot = Robot()
