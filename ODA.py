import numpy as np
from LocalSearch import dist


# Obstacle detection
def obstaclesDetection(x, y, obs_list, sensor_range):
    collision = []
    d_min = np.inf
    for ob in obs_list:
        current_dist = dist((x, y), ob.center)
        if current_dist <= np.sqrt(sensor_range ** 2 + ob.radius ** 2):
            collision.append(ob)
            if current_dist < d_min:
                d_min = current_dist
    return d_min, collision


# Calculate tangent angle
def calculateTangentAngles(x, y, obs):
    theta = np.arcsin(obs.radius/dist((x, y), obs.center))
    if x == obs.center_X:
        if y > obs.center_Y:
            phi = 0.5 * np.pi
        else:
            phi = 1.5 * np.pi
    elif x < obs.center_X:
        phi = np.arctan((y - obs.center_Y)/(obs.center_X - x))
    else:
        phi = np.pi + np.arctan((y - obs.center_Y)/(obs.center_X - x))
    angles = np.array([phi-theta, phi+theta]) % (2*np.pi)
    return angles


# Construct gap vector
def constructGapVectors(x, y, obs_list):
    Vs = np.ones(12, dtype=bool)
    Vg = np.ones(12, dtype=bool)
    for ob in obs_list:
        section1, section2 = np.floor(calculateTangentAngles(x, y, ob) * 6/np.pi)
        section1 = int(section1)
        section2 = int(section2)
        if section1 <= section2:
            for i in range(section1, section2 + 1):
                Vs[i] = False
        else:
            for i in range(section1, 12):
                Vs[i] = False
            for i in range(section2 + 1):
                Vs[i] = False
    for i in range(12):
        Vg[i] = (Vs[i] and Vs[(i-1) % 12])
    return Vg


# Obstacle Avoidance
def obstaclesAvoidance(x, y, gap_vector, d_min, obs_list, sensor_range, func):
    best_record = np.inf
    best_X, best_Y = x, y
    min_radius = np.inf
    for obs in obs_list:
        if obs.radius < min_radius:
            min_radius = obs.radius
    Q = sensor_range + min_radius
    t = Q * 0.5
    for i, b in enumerate(gap_vector):
        if b:
            angle = np.pi / 6 * i
            if d_min < t:
                current_X = x + 1.5 * d_min * np.cos(angle)
                current_Y = y - 1.5 * d_min * np.sin(angle)
            else:
                current_X = x + d_min * np.cos(angle)
                current_Y = y - d_min * np.sin(angle)
            current = func((current_X, current_Y))
            if current < best_record:
                best_record = current
                best_X = current_X
                best_Y = current_Y
    return best_X, best_Y
