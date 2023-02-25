import numpy as np


# Bat Algorithm
def Bat(iteration, population, x_low, x_high, y_low, y_high, f_min, f_max, sigma, alpha, gamma, func):
    # Initialize
    z = []
    v = []
    a = []
    r = []
    pr = []
    for i in range(population):
        x = np.random.uniform(x_low, x_high)
        y = np.random.uniform(y_low, y_high)
        z.append(np.array([x, y]))
        v.append(np.zeros(2))
        a.append(np.random.uniform(1, 5))
        rand = np.random.rand()
        r.append(rand)
        pr.append(rand)

    z_best = min(z, key=func)

    p = [0] * population
    for t in range(1, iteration + 1):
        # Update positions and velocities
        for i in range(population):
            f = f_min + (f_max-f_min) * np.random.rand()
            v[i] = v[i] * 0.01 + (z[i] - z_best) * f
            p[i] = z[i] + v[i]
            if func(p[i]) < func(z[i]):
                z[i] = p[i]
        # Update loudness and pulse rate
        for i in range(population):
            rand = np.random.rand()
            if rand > pr[i]:
                p[i] = p[i] + sigma * np.random.uniform(-1, 1) * np.mean(a)
                if func(p[i]) < func(z[i]):
                    z[i] = p[i]
            if (rand < a[i]) & (func(z[i]) < func(z_best)):
                z_best = z[i]
                a[i] = a[i] * alpha
                pr[i] = r[i] * (1 - np.exp(-gamma * t))
        current_best = min(z, key=func)
        if func(z_best) > func(current_best):
            z_best = current_best
    return z_best, func(z_best)


def PSO_MFB(iter_PSO, population_PSO, iter_Bat, population_Bat, x_low, x_high, y_low, y_high, func):
    alpha_gamma = []
    velocities = []
    p_best = []
    p_best_value = []
    p_best_position = []
    # Parameters for PSO
    b_low = 0
    b_up = 1
    w = 0.1
    rp = 0.25
    rg = 0.75
    # BA parameters
    f_min = 0
    f_max = 1
    sigma = 0.3
    # Initialize PSO
    for i in range(population_PSO):
        x = np.random.uniform(0, 1, 2)
        alpha_gamma.append(x)
        p_best.append(x)
        v = np.random.uniform(b_low-b_up, b_up-b_low, 2)
        velocities.append(v)

    # Initialize p_best & g_best
    for i in range(population_PSO):
        alpha, gamma = alpha_gamma[i]
        pos, val = Bat(iter_Bat, population_Bat, x_low, x_high, y_low, y_high,
                       f_min, f_max, sigma, alpha, gamma, func)
        p_best_position.append(pos)
        p_best_value.append(val)
    g_best_value = min(p_best_value)
    best_index = p_best_value.index(g_best_value)
    g_best = p_best[best_index]
    g_best_position = p_best_position[best_index]

    for t in range(iter_PSO):
        # Update alpha & gamma
        for i in range(population_PSO):
            rand_p = np.random.uniform(0, 1, 2)
            rand_g = np.random.uniform(0, 1, 2)
            velocities[i] = velocities[i] * w + rp * rand_p * (p_best[i] - alpha_gamma[i]) \
                            + rg * rand_g * (g_best - alpha_gamma[i])
            alpha_gamma[i] = alpha_gamma[i] + velocities[i]
        # Call BA for each alpha, gamma and update best values
        for i in range(population_PSO):
            alpha, gamma = alpha_gamma[i]
            pos, val = Bat(iter_Bat, population_Bat, x_low, x_high, y_low, y_high,
                           f_min, f_max, sigma, alpha, gamma, func)
            if val < p_best_value[i]:
                p_best[i] = alpha_gamma[i]
                p_best_value[i] = val
                p_best_position[i] = pos
                if val < g_best_value:
                    g_best = p_best[i]
                    g_best_value = val
                    g_best_position = pos
    return g_best_position


'''# Test function
def func1(nd_array):
    norm = np.sqrt(0.5*np.sum(nd_array ** 2))
    angle = np.cos((2*np.pi*nd_array[0])) + np.cos((2*np.pi*nd_array[1]))
    result = -20*np.exp(-0.2*norm) - np.exp(0.5*angle) + np.e + 20
    return result


def func2(nd_array):
    return sum((nd_array-np.array([900,350]))**2)

x,y = 64,16
i = 1
while func1(np.array([x,y])) > 0.1:
    skew = (900-x)/(350-y)
    x,y = PSO_MFB(5,10,10,5,x-3*skew,x+3*skew,y-3,y+3,func2)
    print(i,x,y)
    i += 1 '''