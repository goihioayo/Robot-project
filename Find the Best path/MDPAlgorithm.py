#!/usr/bin/env python

import rospy
from read_config import read_config

map_y = (read_config()["map_size"][0])
map_x = (read_config()["map_size"][1])
move_list = (read_config()["move_list"])
start = (read_config()["start"])
goal = (read_config()["goal"])
walls = (read_config()["walls"])
pits = (read_config()["pits"])
move_list = (read_config()["move_list"])
max_iterations = (read_config()["max_iterations"])
threshold_difference = (read_config()["threshold_difference"])
reward_step = (read_config()["reward_for_each_step"])
reward_wall = (read_config()["reward_for_hitting_wall"])
reward_goal = (read_config()["reward_for_reaching_goal"])
reward_pit = (read_config()["reward_for_falling_in_pit"])

discount = (read_config()["discount_factor"])
prob_forward = (read_config()["prob_move_forward"])
prob_back = (read_config()["prob_move_backward"])
prob_left = (read_config()["prob_move_left"])
prob_right = (read_config()["prob_move_right"])

    

def getMdp(mapV):   
    tmapV =[[0 for x in range(map_x)] for y in range(map_y)]
    tmapP =[["" for x in range(map_x)] for y in range(map_y)]
    tmapV[goal[0]][goal[1]] = reward_goal
    tmapP[goal[0]][goal[1]] = "GOAL"
    for pit in pits:
	tmapV[pit[0]][pit[1]] = reward_pit
	tmapP[pit[0]][pit[1]] = "PIT"
    for wall in walls:
	tmapP[wall[0]][wall[1]] = "WALL"

    for x in range(map_y):
	for y in range(map_x):
	    if([x,y] != goal and [x,y] not in walls and [x,y] not in pits):
		best = -100000
		policy = ""
		if([0,1] in move_list):
		    total = 0
		    forward = 0
		    backward = 0
		    left = 0
		    right = 0
		    if(y+1>=map_x or [x,y+1] in walls):
			forward = prob_forward*(reward_wall+discount*mapV[x][y])
		    else:
			forward = prob_forward*(reward_step+discount*mapV[x][y+1])
		    if(y-1<0 or [x,y-1] in walls):
			backward = prob_back*(reward_wall+discount*mapV[x][y])
		    else:
			backward = prob_back*(reward_step+discount*mapV[x][y-1])
		    if(x+1>=map_y or [x+1,y] in walls):
			right = prob_right*(reward_wall+discount*mapV[x][y])
		    else:
			right = prob_right*(reward_step+discount*mapV[x+1][y])
		    if(x-1<0 or [x-1,y] in walls):
			left = prob_left*(reward_wall+discount*mapV[x][y])
		    else:
			left = prob_left*(reward_step+discount*mapV[x-1][y])
		    total =forward+backward+left+right
		    if(total >= best):
			best = total
			policy = "E"
		if([0,-1] in move_list):
		    total = 0
		    forward = 0
		    backward = 0
		    left = 0
		    right = 0
		    if(y+1>=map_x or [x,y+1] in walls):
			backward = prob_back*(reward_wall+discount*mapV[x][y])
		    else:
			backward = prob_back*(reward_step+discount*mapV[x][y+1])
		    if(y-1<0 or [x,y-1] in walls):
			forward = prob_forward*(reward_wall+discount*mapV[x][y])
		    else:
			forward = prob_forward*(reward_step+discount*mapV[x][y-1])
		    if(x+1>=map_y or [x+1,y] in walls):
			left = prob_left*(reward_wall+discount*mapV[x][y])
		    else:
			left = prob_left*(reward_step+discount*mapV[x+1][y])
		    if(x-1<0 or [x-1,y] in walls):
			right = prob_right*(reward_wall+discount*mapV[x][y])
		    else:
			right = prob_right*(reward_step+discount*mapV[x-1][y])
		    total =forward+backward+left+right
		    if(total >= best):
			best = total
			policy = "W"
		if([-1,0] in move_list):
		    total = 0
		    forward = 0
		    backward = 0
		    left = 0
		    right = 0
		    if(y+1>=map_x or [x,y+1] in walls):
			right = prob_right*(reward_wall+discount*mapV[x][y])
		    else:
			right = prob_right*(reward_step+discount*mapV[x][y+1])
		    if(y-1<0 or [x,y-1] in walls):
			left = prob_left*(reward_wall+discount*mapV[x][y])
		    else:
			left = prob_left*(reward_step+discount*mapV[x][y-1])
		    if(x+1>=map_y or [x+1,y] in walls):
			backward = prob_back*(reward_wall+discount*mapV[x][y])
		    else:
			backward = prob_back*(reward_step+discount*mapV[x+1][y])
		    if(x-1<0 or [x-1,y] in walls):
			forward = prob_forward*(reward_wall+discount*mapV[x][y])
		    else:
			forward = prob_forward*(reward_step+discount*mapV[x-1][y])
		    total =forward+backward+left+right
		    if(total >= best):
			best = total
			policy = "N"
		if([1,0] in move_list):
		    total = 0
		    forward = 0
		    backward = 0
		    left = 0
		    right = 0
		    if(y+1>=map_x or [x,y+1] in walls):
			left = prob_left*(reward_wall+discount*mapV[x][y])
		    else:
			left = prob_left*(reward_step+discount*mapV[x][y+1])
		    if(y-1<0 or [x,y-1] in walls):
			right = prob_right*(reward_wall+discount*mapV[x][y])
		    else:
			right = prob_right*(reward_step+discount*mapV[x][y-1])
		    if(x+1>=map_y or [x+1,y] in walls):
			forward = prob_forward*(reward_wall+discount*mapV[x][y])
		    else:
			forward = prob_forward*(reward_step+discount*mapV[x+1][y])
		    if(x-1<0 or [x-1,y] in walls):
			backward = prob_back*(reward_wall+discount*mapV[x][y])
		    else:
			backward = prob_back*(reward_step+discount*mapV[x-1][y])
		    total =forward+backward+left+right
		    if(total >= best):
			best = total
			policy = "S"
		tmapV[x][y] = best
		tmapP[x][y] = policy
    return tmapV,tmapP

			
				
		     
