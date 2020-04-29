import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def xscale(x, pos):
    return int(x/10)

def yscale(y, pos):
    return int(y*100)

def base_sixd_model(init_vals, params, t):
    s, i, x, d = init_vals
    S, I, X, D = [s], [i], [x], [d]
    alpha, beta, gamma, theta = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s += -(beta*S[-1]*I[-1])*dt
        i += (beta*S[-1]*I[-1] - alpha*I[-1] - gamma*I[-1] + theta*X[-1])*dt
        x += (alpha*I[-1] - theta*X[-1])*dt
        d += (gamma*I[-1])*dt
        S.append(s)
        I.append(i)
        X.append(x)
        D.append(d)
    return np.stack([S, I, X, D]).T

# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0
mortality_rate = 0.02
# the recovery rate includes death and actual recovery
recovery_rate = 0.05
# this is the reactivation parameter
theta = 0.00085
gamma = mortality_rate * recovery_rate
alpha = recovery_rate - gamma
beta = 0.2

params = alpha, beta, gamma, theta
results = base_sixd_model(init_vals, params, t)

# Simulate
fig, ax = plt.subplots()
grph = ax.plot(results)

x_formatter = FuncFormatter(xscale)
y_formatter = FuncFormatter(yscale)

ax.xaxis.set_major_formatter(x_formatter)
ax.yaxis.set_major_formatter(y_formatter)

plt.xlabel('Number of days')
plt.ylabel('Percentage of Population')

plt.legend(grph, ('Susceptible', 'Infectious', 'Ex-infectious', 'Deaths'))
plt.show()
