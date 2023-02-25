import pygame
import numpy as np
from LocalSearch import dist


class Obstacle:
    def __init__(self, center, radius):
        self.center = center
        self.center_X, self.center_Y = self.center
        self.radius = radius

    def showObstacle(self, canvas):
        pygame.draw.circle(canvas, (0, 0, 127), self.center, self.radius, 2)

    def showOriginalObstacle(self, canvas, rad):
        new_radius = self.radius - rad
        pygame.draw.circle(canvas, (255, 127, 0), self.center, new_radius, 2)

    def move(self):
        pass


class LinearlyMovingObstacle(Obstacle):
    def __init__(self, center, radius, velocity, slope, move_range):
        super().__init__(center, radius)
        self.velocity = velocity
        self.slope = slope
        self.move_range = move_range
        self.slope_X, self.slope_Y = self.slope
        self.min_X_range, self.max_X_range = self.move_range

    def move(self):
        self.center_X += self.velocity * self.slope_X
        # Check bound
        if self.center_X < self.min_X_range:
            self.center_X = self.min_X_range
            self.velocity = -self.velocity
        if self.center_X > self.max_X_range:
            self.center_X = self.max_X_range
            self.velocity = -self.velocity
        self.center_Y -= self.velocity * self.slope_Y
        # Update center
        self.center = (self.center_X, self.center_Y)


class CircularlyMovingObstacle(Obstacle):
    def __init__(self, center, radius, angular_velocity, c, angle_range):
        super().__init__(center, radius)
        self.angular_velocity = angular_velocity
        self.c = c
        self.angle_range = angle_range
        self.c_X, self.c_Y = self.c
        self.r = dist(self.center, self.c)
        if self.center_X > self.c_X:
            self.angle = np.arccos((self.center_X - self.c_X) / self.r)
        else:
            self.angle = 2 * np.pi - np.arccos((self.center_X - self.c_X) / self.r)
        self.min_angle_range, self.max_angle_range = self.angle_range

    def move(self):
        self.angle += self.angular_velocity
        # Check bound
        if self.angle < self.min_angle_range:
            self.angle = self.min_angle_range
            self.angular_velocity = -self.angular_velocity
        if self.angle > self.max_angle_range:
            self.angle = self.max_angle_range
            self.angular_velocity = -self.angular_velocity
        # Update position
        self.center_X = self.c_X + self.r * np.cos(self.angle)
        self.center_Y = self.c_Y - self.r * np.sin(self.angle)
        self.center = (self.center_X, self.center_Y)


test = [[Obstacle((136, 562), 30), Obstacle((316, 460), 48), Obstacle((496, 580), 72),
         Obstacle((478, 280), 50), Obstacle((196, 202), 42)],
        [LinearlyMovingObstacle((466, 574), 32, 16, (np.cos(1.2217), np.sin(1.2217)), (466, 496)),
         LinearlyMovingObstacle((482, 202), 32, 13, (1, 0), (482, 580)),
         CircularlyMovingObstacle((350, 400), 32, 0.25, (316, 400), (0, 0.25 * np.pi)),
         CircularlyMovingObstacle((250, 400), 32, 0.25, (316, 400), (np.pi, 1.25 * np.pi)),
         CircularlyMovingObstacle((286, 250), 32, 0.25, (316, 400), (0.5 * np.pi, 0.75 * np.pi)),
         CircularlyMovingObstacle((346, 550), 32, 0.25, (316, 400), (1.5 * np.pi, 1.75 * np.pi))],
        [LinearlyMovingObstacle((300, 300), 40, 14, (0.7, -0.7), (300, 400)),
         LinearlyMovingObstacle((400, 400), 40, 14, (0.7, 0.7), (400, 500)),
         LinearlyMovingObstacle((500, 300), 40, 14, (-0.7, 0.7), (400, 500)),
         LinearlyMovingObstacle((400, 200), 40, 14, (-0.7, -0.7), (300, 400)),
         CircularlyMovingObstacle((400, 325), 25, 0.7, (400, 300), (-np.inf, np.inf))],
        [LinearlyMovingObstacle((155, 450), 32, 15, (0.5, 0.86), (100, 155)),
         CircularlyMovingObstacle((416, 350), 32, 0.3, (416, 300), (-np.inf, np.inf)),
         LinearlyMovingObstacle((490, 300), 32, 10, (1, 0), (490, 510)), Obstacle((416, 300), 43),
         LinearlyMovingObstacle((416, 398), 32, 10, (0.1, -1), (416, 418)),
         LinearlyMovingObstacle((416, 202), 32, 10, (0.1, 1), (416, 418)),
         LinearlyMovingObstacle((320, 300), 32, 10, (1, 0), (320, 340))],
        [Obstacle((333, 427), 44), Obstacle((250, 375), 43), Obstacle((425, 452), 41),
         Obstacle((488, 164), 42), Obstacle((494, 415), 38), Obstacle((125, 361), 45),
         Obstacle((117, 516), 44), Obstacle((395, 255), 47), Obstacle((600, 176), 46)],
        [LinearlyMovingObstacle((390, 195), 50, 12, (1, 0), (360, 400)), Obstacle((507, 314), 49),
         CircularlyMovingObstacle((316, 455), 45, 0.25, (316, 400), (0, np.pi)), Obstacle((528, 137), 74),
         LinearlyMovingObstacle((263, 594), 57, 5, (0.5, -0.86), (189, 263)), Obstacle((413, 364), 31)]]

test2 = []
for i in range(7):
    test2.append(Obstacle((700-50*i, 200), 30))
test.append(test2)
test.append([Obstacle((480, 220), 22), Obstacle((450, 190), 22), Obstacle((510, 250), 22),
             Obstacle((420, 220), 22), Obstacle((390, 250), 22), Obstacle((480, 280), 22),
             Obstacle((450, 310), 22)])
