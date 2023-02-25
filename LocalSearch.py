import numpy as np


# Distance
def dist(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(sum((x-y) ** 2))


def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                return []
            else:
                return []
        else:
            return [-c/b]
    else:
        determinant = b ** 2 - 4*a*c
        if determinant < 0:
            return []
        elif determinant == 0:
            return [-b/(2*a)]
        else:
            return [(-b-np.sqrt(determinant))/(2*a), (-b+np.sqrt(determinant))/(2*a)]


# Cross detection
def cross(old_robot_X, old_robot_Y, new_robot_X, new_robot_Y, obs):
    a = (new_robot_X - old_robot_X) ** 2 + (new_robot_Y - old_robot_Y) ** 2
    b = 2*((old_robot_X - obs.center_X) * (new_robot_X - old_robot_X)
           + (old_robot_Y - obs.center_Y) * (new_robot_Y - old_robot_Y))
    c = (old_robot_X - obs.center_X) ** 2 + (old_robot_Y - obs.center_Y) ** 2 - obs.radius ** 2
    ans = solve_quadratic(a, b, c)
    if len(ans) < 2:
        return False
    else:
        for an in ans:
            if 0 < an < 1:
                return True
        return False


# Midpoints inside obstacles handling
def MiOH(robot_X, robot_Y, obs_list, ds=1):
    new_robot_X, new_robot_Y = robot_X, robot_Y
    for obs in obs_list:
        if dist((robot_X, robot_Y), obs.center) < obs.radius:
            distance = dist((robot_X, robot_Y), obs.center)
            t = (obs.radius + ds) / distance
            new_robot_X = obs.center_X + t * (robot_X - obs.center_X)
            new_robot_Y = obs.center_Y + t * (robot_Y - obs.center_Y)
    return new_robot_X, new_robot_Y


# Obstacles crossed handling
def OCH(old_robot_X, old_robot_Y, new_robot_X, new_robot_Y, obs_list, scale=0.1):
    for obs in obs_list:
        if cross(old_robot_X, old_robot_Y, new_robot_X, new_robot_Y, obs):
            new_robot_X = old_robot_X + scale * (new_robot_X - old_robot_X)
            new_robot_Y = old_robot_Y + scale * (new_robot_Y - old_robot_Y)
    return new_robot_X, new_robot_Y
