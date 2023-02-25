import numpy as np


def PSO(iteration, population, x_low, x_high, y_low, y_high, w, rp, rg, function):
    # Initialize
    b_low = 0
    b_up = 5
    positions = []
    velocities = []
    p_best = []
    for i in range(population):
        x = np.array([np.random.uniform(x_low, x_high), np.random.uniform(y_low, y_high)])
        positions.append(x)
        p_best.append(x)
        v = np.random.uniform(low=b_low-b_up, high=b_up-b_low, size=2)
        velocities.append(v)
    g_best = min(p_best, key=function)

    # Update positions and velocities
    for t in range(iteration):
        for i in range(population):
            randp = np.random.uniform(0, 1, 2)
            randg = np.random.uniform(0, 1, 2)
            velocities[i] = velocities[i]*w + rp*randp*(p_best[i]-positions[i]) + rg*randg*(g_best-positions[i])
            positions[i] = positions[i] + velocities[i]
        for i in range(population):
            if function(positions[i]) < function(p_best[i]):
                p_best[i] = positions[i]
                if function(p_best[i]) < function(g_best):
                    g_best = p_best[i]
    return g_best
