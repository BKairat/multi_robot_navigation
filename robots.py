from __future__ import annotations
import numpy as np
import pygame as pg
from shapely.geometry import Polygon
import core
from history import History

class CircleRobot:
    # 5 pix = 100 mm 
    radius = 30
    save_zone = 40
    laser_lenght = 500
    laser_range = np.pi
    lasers = 16 # 1080
    resolution = laser_range/(lasers+1)
    collision_w = False
    collision_a = False
    reached = False
    v_limit = 5
    w_limit = np.pi/24

    def __init__(
        self,
        position: np.array,
        orientation: float,
        velocity: np.array,
        index: int,
        history: History | None = None
        ):
        self.position = np.array(position).astype(np.float64)
        self.orientation = orientation
        self.velocity = velocity
        self.color = tuple(np.random.rand(3) * 250)
        self.index = index
        self.history = history
        self.path = [np.array(self.position)]

    def act(self):
        if not self.collision_w and not self.collision_a:
            v = (self.velocity[0] + 1) * 0.5 * self.v_limit
            w = (self.velocity[1] - 0.5) * 2 * self.w_limit
            # print(w)

            self.orientation = core.normalize_angle(self.orientation + w)
            
            delta = core.rotation(np.array([0, v])+self.position, self.orientation, self.position)
            self.position = delta
        self.path.append(np.array(self.position))
    
    def draw(self, surface: pg.surface.Surface, font: pg.font.Font, text_ = None) -> None:
        axis = core.rotation(np.array([0, self.radius*1.6])+self.position, self.orientation, self.position)
        pg.draw.circle(surface, (0,0,0), self.position, self.radius)
        pg.draw.circle(surface, self.color, self.position, self.radius*0.89)
        pg.draw.line(surface, (0, 0, 0), self.position, axis)
        text = font.render(
            str(self.index),
            True,
            (255, 255, 255)
        )
        surface.blit(
            text,
            text.get_rect(center=self.position)
        )

    def get_(self) -> np.ndarray:
        return np.concatenate((self.position, np.array([self.radius])))
    
    def laser_position(self) -> np.ndarray:
        return self.position
    
class CarLikeBot:
    # 5 pix = 100 mm 
    W = 93          
    L = 159
    save_zone = 93
    laser_lenght = 500
    laser_range = np.pi
    lasers = 16 # 1080
    resolution = laser_range/(lasers+1)
    collision_w = False
    collision_a = False
    reached = False
    v_limit = 20
    w_limit = np.pi/6
    
    def __init__(
        self,
        position: np.array,
        orientation: float,
        velocity: np.array,
        index: int,
        history: History | None = None
        ):
        self.position = np.array(position).astype(np.float64)
        self.orientation = orientation
        self.velocity = velocity
        self.color = tuple(np.random.rand(3) * 250)
        self.index = index
        self.history = history
        self.path = [np.array(self.position)]
        
    def act(self) -> None:
        if not self.collision_w and not self.collision_a:
            v = self.velocity[0] * self.v_limit
            w = (self.velocity[1] - 0.5) * 2 * self.w_limit
            # print(self.velocity[0], self.velocity[1])
            self.position += np.array([
                v*np.sin(self.orientation),
                v*np.cos(self.orientation)
            ])
            self.orientation += (v/self.L) * np.tan(w)
        self.path.append(np.array(self.position))

    def draw(self, surface: pg.surface.Surface, font: pg.font.Font, text: str|None = None):
        lenght = self.L + 0.4*self.L
        width = self.W
        
        car = np.array([
            [-width/2, -lenght/2],
            [-width/2,  lenght/2],
            [ width/2,  lenght/2],
            [ width/2, -lenght/2]
        ])
        weel = np.array([
            [-width*0.1, -lenght*0.1],
            [-width*0.1,  lenght*0.1],
            [ width*0.1,  lenght*0.1],
            [ width*0.1, -lenght*0.1]
        ])
        car += self.position
        car = core.rotation(car, self.orientation)
        weel_1, weel_2, weel_3, weel_4 = (car - self.position)*0.65
        
        weel_1 = core.rotation(weel + weel_1 + self.position, self.orientation)
        weel_2 = core.rotation(weel + weel_2 + self.position, self.orientation)
        weel_3 = core.rotation(weel + weel_3 + self.position, self.orientation)
        weel_4 = core.rotation(weel + weel_4 + self.position, self.orientation)

        weel_2 = core.rotation(weel_2, (self.velocity[1] - 0.5) * 2 * self.w_limit)
        weel_3 = core.rotation(weel_3, (self.velocity[1] - 0.5) * 2 * self.w_limit)
        
        pg.draw.polygon(surface, self.color, car,  width=5)
        pg.draw.polygon(surface, (0, 0, 0), weel_1,  width=0)
        pg.draw.polygon(surface, (0, 0, 0), weel_2,  width=0)
        pg.draw.polygon(surface, (0, 0, 0), weel_3,  width=0)
        pg.draw.polygon(surface, (0, 0, 0), weel_4,  width=0)
        text = font.render(
            "%.3f, %.3f" % (self.velocity[0] * self.v_limit, (self.velocity[1] - 0.5) * 2 * self.w_limit) if not text else text,
            True,
            (0, 0, 0)
        )
        surface.blit(
            text,
            text.get_rect(center=self.position)
        )
        
    def collision(self, other: CarLikeBot | np.ndarray) -> bool:
        lenght = self.L + 0.4*self.L
        width = self.W
        polygon = np.array([
            [-width/2, -lenght/2],
            [-width/2,  lenght/2],
            [ width/2,  lenght/2],
            [ width/2, -lenght/2]
        ])
        polygon += self.position
        polygon = core.rotation(polygon, self.orientation)
        polygon = Polygon(polygon)
        
        if type(other) == np.ndarray:
            other_polygon = Polygon(other)
        else:
            lenght = other.L + 0.4*other.L
            width = other.W
            other_polygon = np.array([
                [-width/2, -lenght/2],
                [-width/2,  lenght/2],
                [ width/2,  lenght/2],
                [ width/2, -lenght/2]
            ])
            other_polygon += other.position
            other_polygon = core.rotation(other_polygon, other.orientation)
            other_polygon = Polygon(other_polygon)
        
        return bool(polygon.intersects(other_polygon))

    def get_(self) -> np.ndarray:
        lenght = self.L + 0.4*self.L
        width = self.W
        
        car = np.array([
            [-width/2, -lenght/2],
            [-width/2,  lenght/2],
            [ width/2,  lenght/2],
            [ width/2, -lenght/2]
        ])
        car += self.position
        car = core.rotation(car, self.orientation)
        return car
    
    def laser_position(self) -> np.ndarray:
        return self.position + core.rotation(np.array([0, self.L/2]), self.orientation)