import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def xscale(x, pos):
    return int(x/10)

def yscale(y, pos):
    return int(y*100)

def base_sir_model(init_vals, params, t):
    s, i, r = init_vals
    S, I, R = [s], [i], [r]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s += (alpha*I[-1] - rho*beta*S[-1]*I[-1])*dt
        i += (rho*beta*S[-1]*I[-1] - gamma*I[-1] - alpha*I[-1])*dt
        r += (gamma*I[-1])*dt
        S.append(s)
        I.append(i)
        R.append(r)
    return np.stack([S, I, R]).T

def base_seir_model(init_vals, params, t):
    s, e, i, r = init_vals
    S, E, I, R = [s], [e], [i], [r]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s += - (rho*beta*S[-1]*I[-1])*dt
        e += (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        i += (alpha*E[-1] - gamma*I[-1])*dt
        r += (gamma*I[-1])*dt
        S.append(s)
        E.append(e)
        I.append(i)
        R.append(r)
    return np.stack([S, E, I, R]).T

def base_sisd_model(init_vals, params, t):
    s, i, d = init_vals
    S, I, D = [s], [i], [d]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s += (alpha*I[-1] - rho*beta*S[-1]*I[-1])*dt
        i += (rho*beta*S[-1]*I[-1] - alpha*I[-1] - gamma*I[-1])*dt
        d += (gamma*I[-1])*dt
        S.append(s)
        I.append(i)
        D.append(d)
    return np.stack([S, I, D]).T

def base_sixd_model(init_vals, params, t):
    s, i, x, d = init_vals
    S, I, X, D = [s], [i], [x], [d]
    alpha, beta, gamma, theta, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        s += -(rho*beta*S[-1]*I[-1])*dt
        i += (rho*beta*S[-1]*I[-1] - alpha*I[-1] - gamma*I[-1] + theta*X[-1])*dt
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
recovery_rate = 0.2
# this is social distancing parameter
rho = 0.4
# this is the reactivation parameter
theta = 0.00085
gamma = mortality_rate * recovery_rate
alpha = recovery_rate - gamma
beta = 2

params = alpha, beta, gamma, theta, rho
results_sir = base_sir_model(init_vals[:-1], (recovery_rate, beta, 0.5, rho), t)
results_seir = base_seir_model(init_vals, (recovery_rate, beta, 0.5, rho), t)
results_sisd = base_sisd_model(init_vals[:-1], (alpha, beta, gamma, rho), t)
results_sixd = base_sixd_model(init_vals, (alpha, beta, gamma, theta, rho), t)


# Simulate
fig, axs = plt.subplots(1, 3)
fig.suptitle('Social Distancing ' + str(round((1-rho)*100)) + '%')
grph_sir = axs[0].plot(results_sir)
# grph_seir = axs[0].plot(results_seir)
grph_sisd = axs[1].plot(results_sisd)
grph_sixd = axs[2].plot(results_sixd)


x_formatter = FuncFormatter(xscale)
y_formatter = FuncFormatter(yscale)

for ax in axs:
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    
    ax.set_xlabel('Number of days')
    ax.set_ylabel('Percentage of Population')

axs[0].legend(grph_sir, ('Susceptible', 'Infectious', 'Recovered'))
# axs[0].legend(grph_seir, ('Susceptible', 'Exposed', 'Infectious', 'Recovered'))
axs[1].legend(grph_sisd, ('Susceptible', 'Infectious', 'Deaths'))
axs[2].legend(grph_sixd, ('Susceptible', 'Infectious', 'Ex-infectious', 'Deaths'))

axs[0].set_title("SIR")
# axs[0].set_title("SEIR")
axs[1].set_title("SISD")
axs[2].set_title("SIXD")


plt.show()
