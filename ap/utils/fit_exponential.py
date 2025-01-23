import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# def exp_func(x, a, b, c):
#     return a*np.exp(b*x) + c

# def exp_func(x, a, b, c):
#     return a*x**2 + b*x + c

def exp_func(x, a, b):
    return a*x + b


endmember_mixing_ratio = 0.7
if endmember_mixing_ratio == 0.8:
    # for 80%
    x = np.array([0, 0.1, 0.13, 0.15, 0.17, 0.18, 0.2])
    y = np.array([0, 10.6, 36.8, 47.8, 66.1, 80.9, 100])
elif endmember_mixing_ratio == 0.7:
    # for 70%
    x = np.array([0, 0.1, 0.2, 0.23, 0.25, 0.27, 0.28, 0.3])
    y = np.array([0, 16.3, 23.7, 46, 56.3, 71, 84.7, 100])
else:
    x = np.array([0, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 1])
    y = np.array([0, 57.8, 64.2, 67.4, 76.1, 80, 86.8, 93.2, 100])

c, cov = curve_fit(exp_func, xdata=x, ydata=y)

if endmember_mixing_ratio == 0.7:
    x_lim = 0.3
elif endmember_mixing_ratio == 0.8:
    x_lim = 0.2
else:
    x_lim = 1

plt.figure(figsize=(10, 10))
x_scale = np.arange(0, x_lim + x_lim/10, x_lim/100)
y_pred = exp_func(x, *c)
r2_s = r2_score(y, y_pred)
plt.xlim(-0.01, x_lim)
plt.xticks(ticks=x_scale[::10], labels=(100*x_scale/x_lim)[::10])
plt.ylim(-1, 100)
plt.scatter(x, y)
plt.plot(x_scale, exp_func(x_scale, *c), label=f"R²={r2_s}")
plt.legend()
plt.show()
