import numpy as np
from matplotlib import pyplot as plt

def base_sisd_model(init_vals, params, t):
    s, i, d = init_vals
    S, I, D = [s], [i], [d]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s = s + (alpha*i - rho*beta*s*i)*dt
        i = i + (rho*beta*s*i - alpha*i - gamma*i)*dt
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
mortality_rate = 0.02
# the recovery rate includes death and actual recovery
recovery_rate = 0.2
# this is social distancing parameter
rho = 0.2
gamma = mortality_rate * recovery_rate
alpha = recovery_rate - gamma
beta = 2

params = alpha, beta, gamma, rho
# Run simulation
results = base_sisd_model(init_vals, params, t)
grph = plt.plot(results)
plt.legend(grph, ('Susceptible', 'Infectious', 'Deaths'))
plt.show()
