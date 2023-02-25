import numpy as np
import copy
import pygame.draw
# from PSO_MFB import PSO_MFB, Bat
from PurePSO import PSO
from Obstacles import test
from LocalSearch import dist, solve_quadratic
import time


# Robot
robot_X = 16
robot_Y = 700
robot_X_prev, robot_Y_prev = robot_X, robot_Y
robot_radius = 16
SR = 48
threshold = 4


def draw_robot():
    pygame.draw.circle(screen, (255, 0, 0), (robot_X, robot_Y), robot_radius)


# Goal
goal_point_X = 616
goal_point_Y = 100
goal = pygame.image.load("flag.png")


def draw_goal():
    screen.blit(goal, (goal_point_X, goal_point_Y - 64))


# Preliminaries
# Input
number = int(input('Enter test number: '))
obs = test[number]


# Smoothness
def smooth(x1, y1, x2, y2):
    if x1 == x2:
        if y1 < y2:
            angle1 = -np.pi / 2
        else:
            angle1 = np.pi / 2
    else:
        angle1 = np.arctan((y1 - y2) / (x2 - x1))
    angle2 = np.arctan((y1 - goal_point_Y) / (goal_point_X - x1))
    return np.abs(angle1 - angle2)


def is_interior(x, y):
    for obstacle in obs:
        if dist((x, y), obstacle.center) < obstacle.radius:
            return 1
    return 0


# Optimization function
w1 = 0.7
w2 = 0.3


def optimization(pos):
    x, y = pos
    return w1 * dist((x, y), (goal_point_X, goal_point_Y)) / 100 + w2 * smooth(robot_X_prev, robot_Y_prev, x, y) \
           + 10000 * is_interior(x, y)


# Obstacle detection
def obstacles_detection(x, y, obstacles, sensor_range):
    collision = []
    for obstacle in obstacles:
        current_dist = dist((x, y), obstacle.center)
        if current_dist <= sensor_range + obstacle.radius:
            collision.append(obstacle)
    return collision


# Find tangent point for two sides (left and right) to an obstacle
# Add additional da and ds to avoid computational errors
def calculate_tangent_point(x, y, obstacle, side, da=0.1, ds=2):
    distance = dist((x, y), obstacle.center)
    if distance <= obstacle.radius + threshold:
        # If obstacle is too close then the distance moved is too small so set distance = robot_radius
        tangential_distance = robot_radius
    else:
        tangential_distance = np.sqrt(distance ** 2 - obstacle.radius ** 2)
    theta = np.arcsin(obstacle.radius / distance)
    if x == obstacle.center_X:
        if y > obstacle.center_Y:
            phi = 1.5 * np.pi
        else:
            phi = 0.5 * np.pi
    elif x < obstacle.center_X:
        phi = np.arctan((obstacle.center_Y - y) / (obstacle.center_X - x))
    else:
        phi = np.pi + np.arctan((obstacle.center_Y - y) / (obstacle.center_X - x))
    if side == 'left':
        angle = phi - theta - da
    elif side == 'right':
        angle = phi + theta + da
    point = np.array([x, y]) + (tangential_distance + ds) * np.array([np.cos(angle), np.sin(angle)])
    return point


# Find the closest obstacle blocking the straight movement towards goal
def detect_nearest_obstacle(start_x, start_y, goal_x, goal_y, obs_list):
    best_ans = 2
    nearest_obs = None
    for obstacle in obs_list:
        a = (start_x - goal_x) ** 2 + (start_y - goal_y) ** 2
        b = 2 * ((start_x - obstacle.center_X) * (goal_x - start_x)
                 + (start_y - obstacle.center_Y) * (goal_y - start_y))
        c = (start_x - obstacle.center_X) ** 2 + (start_y - obstacle.center_Y) ** 2 - obstacle.radius ** 2
        ans = solve_quadratic(a, b, c)
        if len(ans) < 2:
            continue
        else:
            current_ans = np.min(ans)
            if 0 < current_ans < 1:
                if current_ans < best_ans:
                    best_ans = current_ans
                    nearest_obs = obs_list.index(obstacle)
    return nearest_obs


# Find intersection of two obstacles and add a small number ds to avoid computational errors
def find_intersection(x, y, obstacle_0, obstacle_1, ds):
    d = dist(obstacle_0.center, obstacle_1.center)
    a = (obstacle_0.radius ** 2 - obstacle_1.radius ** 2 + d ** 2) / (2 * d)
    p_0 = np.array(obstacle_0.center)
    p_1 = np.array(obstacle_1.center)
    h = np.sqrt(obstacle_0.radius ** 2 - a ** 2)
    midpoint = p_0 + a / d * (p_1 - p_0)
    intersect_1 = midpoint + h / d * np.array([p_0[0] - p_1[0], p_1[1] - p_0[1]])
    intersect_2 = midpoint - h / d * np.array([p_0[0] - p_1[0], p_1[1] - p_0[1]])
    if dist((x, y), intersect_1) <= dist((x, y), intersect_2):
        return intersect_1 + ds * (intersect_1 - intersect_2) / (2 * h)
    else:
        return intersect_2 + ds * (intersect_2 - intersect_1) / (2 * h)


def locate_initial_position(x, y, obs_list, ds=4):
    collision = []
    for obstacle in obs_list:
        if dist((x, y), obstacle.center) <= obstacle.radius:
            collision.append(obstacle)
            if len(collision) == 2:
                break
    if len(collision) == 2:
        x, y = find_intersection(x, y, collision[0], collision[1], ds)
    elif len(collision) == 1:
        distance = dist((x, y), collision[0].center)
        t = (collision[0].radius + ds) / distance
        x = collision[0].center_X + t * (x - collision[0].center_X)
        y = collision[0].center_Y + t * (y - collision[0].center_Y)
    return x, y


# Tangent + PSO
def find_direction(start_x, start_y, goal_x, goal_y, obs_list, side):
    obstacle_index = detect_nearest_obstacle(start_x, start_y, goal_x, goal_y, obs_list)
    if obstacle_index is not None:
        obstacle = obs_list[obstacle_index]
        distance = dist((start_x, start_y), obstacle.center)
        if distance <= np.sqrt(SR ** 2 + obstacle.radius ** 2):
            new_obs_list = copy.deepcopy(obs_list)
            new_obs_list.pop(obstacle_index)
            point = calculate_tangent_point(start_x, start_y, obstacle, side)
            goal_x, goal_y = find_direction(start_x, start_y, point[0], point[1], new_obs_list, side)
        else:
            goal_x, goal_y = PSO(20, 10, robot_X - 8, robot_X + 8, robot_Y - 8,
                                 robot_Y + 8, 0.25, 0.2, 0.8, optimization)
    else:
        if goal_x == goal_point_X and goal_y == goal_point_Y:
            d = dist((start_x, start_y), (goal_x, goal_y))
            goal_x = start_x + robot_radius * (goal_x - start_x) / d
            goal_y = start_y + robot_radius * (goal_y - start_y) / d
    return goal_x, goal_y


# Tangent only
'''def find_direction_org(start_x, start_y, goal_x, goal_y, obs_list, side):
    obstacle_index = detect_nearest_obstacle(start_x, start_y, goal_x, goal_y, obs_list)
    if obstacle_index is not None:
        new_obs_list = copy.deepcopy(obs_list)
        new_obs_list.pop(obstacle_index)
        point = calculate_tangent_point(start_x, start_y, obs_list[obstacle_index], side)
        goal_x, goal_y = find_direction_org(start_x, start_y, point[0], point[1], new_obs_list, side)
    else:
        if goal_x == goal_point_X and goal_y == goal_point_Y:
            goal_x = start_x + robot_radius * (goal_x - start_x) / dist((start_x, start_y), (goal_x, goal_y))
            goal_y = start_y + robot_radius * (goal_y - start_y) / dist((start_x, start_y), (goal_x, goal_y))
    return goal_x, goal_y'''


# Initializing Screen
pygame.init()
# extra padding for visualization
screen = pygame.display.set_mode((716, 716))
pygame.display.set_caption("Environment Simulation")
running = True


while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw goal
    draw_goal()

    # Draw obstacles
    for ob in obs:
        ob.move()
        ob.showOriginalObstacle(screen, robot_radius)
        ob.showObstacle(screen)

    if dist((robot_X, robot_Y), (goal_point_X, goal_point_Y)) > robot_radius:
        '''# Original approach
        robot_X_prev, robot_Y_prev = robot_X, robot_Y
        d_min, obstacles_list = obstaclesDetection(robot_X, robot_Y, obs, SR)
        robot_X, robot_Y = locate_initial_position(robot_X, robot_Y, obstacles_list)
        if not obstacles_list:
            # Hybridized PSO-MFB
            robot_X, robot_Y = PSO_MFB(5, 5, 5, 5, robot_X - 8, robot_X + 8, robot_Y - 8, robot_Y + 8, optimization)
            # Local search
            robot_X, robot_Y = OCH(robot_X_prev, robot_Y_prev, robot_X, robot_Y, obs)
            robot_X, robot_Y = locate_initial_position(robot_X, robot_Y, obs)
        else:
            gap_vector = constructGapVectors(robot_X, robot_Y, obstacles_list)
            robot_X, robot_Y = obstaclesAvoidance(robot_X, robot_Y, gap_vector, d_min, obstacles_list,
                                                  SR, optimization)
        time.sleep(0.2)'''
        # New approach
        robot_X_prev, robot_Y_prev = robot_X, robot_Y
        obstacles_list = obstacles_detection(robot_X_prev, robot_Y_prev, obs, SR)
        robot_X, robot_Y = locate_initial_position(robot_X_prev, robot_Y_prev, obstacles_list)
        if not obstacles_list:
            robot_X += robot_radius * (goal_point_X - robot_X) / dist((robot_X, robot_Y), (goal_point_X, goal_point_Y))
            robot_Y += robot_radius * (goal_point_Y - robot_Y) / dist((robot_X, robot_Y), (goal_point_X, goal_point_Y))
            time.sleep(0.2)
        else:
            '''# Greedy tangential movement
            robot_X_left, robot_Y_left = find_direction_org(robot_X, robot_Y, goal_point_X, goal_point_Y,
                                                            obstacles_list, 'left')
            robot_X_right, robot_Y_right = find_direction_org(robot_X, robot_Y, goal_point_X, goal_point_Y,
                                                              obstacles_list, 'right')
            if optimization((robot_X_right, robot_Y_right)) < optimization((robot_X_left, robot_Y_left)):
                robot_X, robot_Y = robot_X_right, robot_Y_right
            else:
                robot_X, robot_Y = robot_X_left, robot_Y_left
            robot_X, robot_Y = locate_initial_position(robot_X, robot_Y, obs)
            time.sleep(0.2)'''
            # Hybrid
            robot_X_left, robot_Y_left = find_direction(robot_X, robot_Y, goal_point_X, goal_point_Y,
                                                        obstacles_list, 'left')
            robot_X_right, robot_Y_right = find_direction(robot_X, robot_Y, goal_point_X, goal_point_Y,
                                                          obstacles_list, 'right')
            if optimization((robot_X_right, robot_Y_right)) < optimization((robot_X_left, robot_Y_left)):
                robot_X, robot_Y = robot_X_right, robot_Y_right
            else:
                robot_X, robot_Y = robot_X_left, robot_Y_left
            time.sleep(0.2)
        # Check bound
        if robot_X > 700:
            robot_X = 700
        elif robot_X < 16:
            robot_X = 16
        if robot_Y > 700:
            robot_Y = 700
        elif robot_Y < 16:
            robot_Y = 16
    else:
        running = False
    # Draw robot
    draw_robot()

    pygame.display.update()
