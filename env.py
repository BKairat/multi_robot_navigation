from __future__ import annotations
import cv2
import numpy as np
import pygame as pg
from robots import CircleRobot, CarLikeBot
from core import angle_vector, rotation
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import split
from robots import CarLikeBot, CircleRobot
import json
from history import History, MyHistory
# import copy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def laser_intersection(laser, object):
    segment_start, segment_end = laser
    segment = LineString([segment_start, segment_end])
    if object.shape == (3,):
        circle_center, radius = object[:-1], object[-1]
        point = Point(circle_center)
        circle = point.buffer(radius).boundary
        intersection = circle.intersection(segment)
    else:
        polygon = Polygon(object)
        intersection = polygon.intersection(segment)

    if intersection.geom_type == "Point":
        return intersection.coords[0]
    elif intersection.geom_type == "MultiPoint":
        return min([intersection.geoms[0].coords[0], intersection.geoms[1].coords[0]], key=lambda x: np.linalg.norm(np.array(x)-segment_start))
    elif intersection.geom_type == 'LineString':
        if len(intersection.coords):
            return intersection.coords[0]
        else:
            return segment_end
    elif intersection.geom_type == "MultiLineString":
        return intersection.geoms[0].coords[0]
    else:
        return segment_end

def ray_cast(robot: CarLikeBot | CircleRobot, objects: [np.ndarray]) -> list:
    laser_pos = robot.laser_position()
    laser = rotation(np.array([0, robot.laser_lenght]), robot.orientation-robot.laser_range/2) + laser_pos
    ret = []
    for _ in range(robot.lasers):
        laser = rotation(laser, robot.resolution, laser_pos)
        intersections = [laser_intersection((laser_pos, laser), obj) for obj in objects]
        ret.append(min(intersections, key=lambda x: np.linalg.norm(laser_pos - x)))
    return ret 

def collision(object1: np.array, object2: np.array):
    if object1.shape == (3,):
        circle_center, radius = object1[:-1], object1[-1]
        point = Point(circle_center)
        obj1 = point.buffer(radius).boundary
    else:
        obj1 = Polygon(object1)
    if object2.shape == (3,):
        circle_center, radius = object2[:-1], object2[-1]
        point = Point(circle_center)
        obj2 = point.buffer(radius).boundary
    else:
        obj2 = Polygon(object2)
    return bool(obj1.intersection(obj2)) 
    
class Environment:
    def __init__(
        self,
        agent_type: CircleRobot | CarLikeBot,
        map: str | int = "maps/map1.png",
        surface: pg.surface.Surface | None = None,
        font: pg.font.Font | None = None,
        observation_: bool = False,
        policy = None,
        reward = None,
        discrete: bool = True,
        threshhold: int = 40
        ):
        if type(map) == str:
            self.map_bgr = cv2.imread(map)
        elif type(map) == int:
            self.map_bgr = cv2.imread(f"maps/map{map}.png")
        self.agent_type = agent_type
        self.shape = self.map_bgr.shape[:-1]
        self.surface = surface
        self.font = font
        self.observation_ = observation_
        self.policy = policy
        self.reward = reward
        self.discrete = discrete
        self.threshhold = threshhold
        self.addObstacles()
        self.addAgents()
        self.addGoals()
        self.get_observation()
        self.disc_act = {
            0: (0, 0.5),        # 0
            1: (1, 0.5),        # 0
            2: (1, 0.625),      # 0.25
            3: (1, 0.375),      #-0.25
            4: (1, 0.75),       # 0.5
            5: (1, 0.25),       #-0.5
            6: (1, 0.875),      # 0.75
            7: (1, 0.125),      #-0.75
            8: (1, 1),          # 1
            9: (1, 0),          #-1
        }
        
    def addObstacles(self):
        self.obstalces = []
        THRESH = cv2.inRange(self.map_bgr, np.array([0,0,0]), np.array([0,0,0]))
        CONTOURS, _  = cv2.findContours(THRESH.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in CONTOURS:
            self.obstalces.append(np.reshape(contour, (contour.shape[0], 2)))
    
    def addAgents(self):
        self.agents = []
        thresh_red = cv2.inRange(self.map_bgr, np.array([0,0,255]), np.array([0,0,255]))
        contours_red, _  = cv2.findContours(thresh_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        thresh_blue = cv2.inRange(self.map_bgr, np.array([255,0,0]), np.array([255,0,0]))
        contours_blue, _  = cv2.findContours(thresh_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours_red) == len(contours_blue), "Map must contain same amount of red contours as bule ones"
        red_centers = [np.mean(c[0], axis=0) for c in contours_red]
        blue_centers = [np.mean(c[0], axis=0) for c in contours_blue]
        
        red_blue = [(red, min(blue_centers, key = lambda x: np.linalg.norm(x-red)))  for red in red_centers]
        for i, rb  in enumerate(red_blue):
            orient = angle_vector(rb[1]-rb[0])
            self.agents.append(self.agent_type(
                position = rb[0] + np.random.randn(2),
                orientation = orient + np.random.uniform(-np.pi/12, np.pi/12),
                velocity=np.array([0.0, 0.0]),
                index=i,
                ))
    
    def addGoals(self):
        self.goals = {}
        THRESH = cv2.inRange(self.map_bgr, np.array([0,255,0]), np.array([0,255,0]))
        CONTOURS, _  = cv2.findContours(THRESH.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        order = [i for i in range(len(CONTOURS))]
        np.random.shuffle(self.agents)
        for index, contour in enumerate(CONTOURS):
            self.goals[self.agents[len(self.agents)-index-1]] = np.mean(np.reshape(contour, (contour.shape[0], 2)), axis=0)
        
    def get_observation(self) -> [(int, np.ndarray)]:
        ret = []
        for agent in self.agents:
            
            obj = self.obstalces.copy()
            for ag in self.agents:
                if ag != agent:
                    obj.append(ag.get_())
            
            # obs = {
            #     #  lasers
            #     "ol": np.linalg.norm(np.array(ray_cast(agent, obj)) - agent.laser_position(), axis=1),
            #     #  relative orientation to the goal as dir vector
            #     "og": (self.goals[agent] - agent.position)/np.linalg.norm(self.goals[agent] - agent.position),
            #     #  linear and angular velocity 
            #     "ov": agent.velocity,
            #     #  euclidian dstance to goal.
            #     "od": np.reshape(np.linalg.norm(agent.position-self.goals[agent]), (1))
            # }
            obs = {
                #  lasers
                "ol": np.linalg.norm(np.array(ray_cast(agent, obj)) - agent.laser_position(), axis=1) / agent.laser_lenght,
                # 
                "op": np.concatenate((self.goals[agent] - agent.position, np.array([agent.orientation])))
            }
            
            if agent.history:
                agent.history.add(obs)
            else:
                agent.history = MyHistory(obs)
            ret.append(MyHistory(obs))

        return ret[0].get_vectors()
    
    def step(self, actions):
        if self.discrete:
            if not actions.shape:
                actions = [self.disc_act[actions.item()]]
            else:
                actions = np.array([self.disc_act[a.item()] for a in actions])
        else:
            # print("env", type(actions))
            if len(actions.shape) == (1, 2):
                actions = actions.reshape(2)
            # actions = actions.detach().numpy()
            actions = sigmoid(actions)
            # print(actions)
        r = []
        done = False
        cnt = 0
        for i, agent in enumerate(self.agents):
            agent.velocity = actions[i]
            if not agent.collision_w and not agent.collision_a and  not agent.reached:
                agent.act()
                obj = agent.get_()
                for obst in self.obstalces:
                    if collision(obj, obst):
                        agent.collision_w = True
                        break
                for ag in self.agents:
                    if collision(obj, ag.get_()) and ag != agent:
                        agent.collision_a = True
                        ag.collision_a = True
                        break
                if np.linalg.norm(self.goals[agent]-agent.position) <= self.threshhold:
                    agent.reached = True
            else:
                cnt += 1
        if cnt >= len(self.agents):
            done = True
        
        self.get_observation()
        
        if self.reward:
            for ag in self.agents:
                r.append(float(self.reward.reward(ag, self.goals[ag]))) 
        
        return r, done
                        
    def draw(self, text_ = None):
        for goal in self.goals:
            color = goal.color
            pg.draw.circle(self.surface, color, self.goals[goal], self.threshhold, 8)
            text = self.font.render(
                str(goal.index),
                True,
                (0, 0, 0)
            )
            self.surface.blit(
                text,
                text.get_rect(center=self.goals[goal])
            )
        
        for obst in self.obstalces:
            pg.draw.polygon(self.surface, (0, 0, 0), obst,  width=0)
        
        for agent in self.agents:
            if agent.reached:
                text_ = "SUCCESS"
            agent.draw(self.surface, self.font, text_)
        
        if self.observation_: 
            for agent in self.agents:
                obj = self.obstalces.copy()
                for ag in self.agents:
                    if ag != agent:
                        obj.append(ag.get_())
                for o in ray_cast(agent, obj):
                    pg.draw.circle(self.surface, agent.color, o, 5)
        
    def reset(self, map_: int):
        del self.agents
        del self.goals
        del self.obstalces
        self.map_bgr = cv2.imread(f"maps/map{map_}.png")
        self.shape = self.map_bgr.shape[:-1]
        self.addObstacles()
        self.addAgents()
        self.addGoals()
        self.get_observation()
        return
                    