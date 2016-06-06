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
mapH =([[0 for x in range(map_x)] for y in range(map_y)])
move_list = (read_config()["move_list"])

def initialize_map():
    for y in range(map_y):
	for x in range(map_x):
	    mapH[y][x] = abs(y-goal[0]) + abs(x-goal[1])

def getAstar():
    initialize_map()
    closedSet = []
    openSet = [start]
    cameFrom =([[[0,0] for x in range(map_x)] for y in range(map_y)])
    gScore = ([[10000000 for x in range(map_x)] for y in range(map_y)])
    gScore[start[0]][start[1]] = 0
    fScore = ([[10000000 for x in range(map_x)] for y in range(map_y)])
    fScore[start[0]][start[1]] = mapH[start[0]][start[1]]

    while(len(openSet) != 0):
	current = openSet[0]
	for grid in openSet:
	    if(fScore[grid[0]][grid[1]] < fScore[current[0]][current[1]]):
		current = grid
	if(current == goal):
	    return reconstruct_path(cameFrom)

	openSet.remove(current)
	closedSet.append(current)
	for poss in move_list:
	    neighbor = [current[0]+poss[0],current[1]+poss[1]]
	    if(neighbor in walls):
		continue
	    if(neighbor in pits):
		continue
	    if(neighbor in closedSet):
		continue
	    if(neighbor[0] <0 or neighbor[0] >= map_y or neighbor[1] <0 or neighbor[1] >= map_x):
		continue
	    tentative_gScore = gScore[current[0]][current[1]] + 1
	    if(neighbor not in openSet):
		openSet.append(neighbor)
	    elif(tentative_gScore >= gScore[neighbor[0]][neighbor[1]]):
		continue

	    cameFrom[neighbor[0]][neighbor[1]] = current
	    gScore[neighbor[0]][neighbor[1]] = tentative_gScore
	    fScore[neighbor[0]][neighbor[1]] = tentative_gScore + mapH[neighbor[0]][neighbor[1]]

    return []

def reconstruct_path(cameFrom):
    path = [goal]
    current = goal
    while(current != start):
	current = cameFrom[current[0]][current[1]]
        path = [current] + path
    return path
