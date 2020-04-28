import numpy as np
from matplotlib import pyplot as plt

def base_sixd_model(init_vals, params, t):
    s, i, x, d = init_vals
    S, I, X, D = [s], [i], [x], [d]
    alpha, beta, gamma, theta = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s += -(beta*s*i)*dt
        i += (beta*s*i - alpha*i - gamma*i + theta*x)*dt
        x += (alpha*i - theta*x)*dt
        d += (gamma*i)*dt
        S.append(s)
        I.append(i)
        X.append(x)
        D.append(d)
    return np.stack([S, I, X, D]).T

# Define parameters
t_max = 1000
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0
mortality_rate = 0.02
# the recovery rate includes death and actual recovery
recovery_rate = 0.2
# this is the reactivation parameter
theta = 0.00085
gamma = mortality_rate * recovery_rate
alpha = recovery_rate - gamma
beta = 2

params = alpha, beta, gamma, theta
# Run simulation
results = base_sixd_model(init_vals, params, t)
grph = plt.plot(results)
plt.legend(grph, ('Susceptible', 'Infectious', 'Ex-infectious', 'Deaths'))
plt.show()
