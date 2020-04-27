import numpy as np
from matplotlib import pyplot as plt

def base_sisd_model(init_vals, params, t):
    s, i, d = init_vals
    S, I, D = [s], [i], [d]
    alpha, beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s = s + (alpha*i - beta*s*i)*dt
        i = i + (beta*s*i - alpha*i - gamma*i)*dt
        d = d + (gamma*i)*dt
        S.append(s)
        I.append(i)
        D.append(d)
    return np.stack([S, I, D]).T

# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0
alpha = 0.2
beta = 1.75
gamma = 0.02
params = alpha, beta, gamma
# Run simulation
results = base_sisd_model(init_vals, params, t)
plt.plot(results)
plt.show()
