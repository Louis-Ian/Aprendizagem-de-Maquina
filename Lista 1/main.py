import numpy as np
import matplotlib.pyplot as plt

artificial1d_dataset = np.genfromtxt("./artificial1d.csv", delimiter=",")
boston_dataset = np.genfromtxt("./boston.csv", delimiter=",")

X_art = np.c_[np.ones(artificial1d_dataset.shape[0]), artificial1d_dataset[:, [0]]]
y_art = artificial1d_dataset[:, [1]]

w_art = np.linalg.inv(X_art.T @ X_art) @ X_art.T @ y_art

pred_art = X_art @ w_art

mse_art = ((y_art - pred_art) ** 2).mean()

X_line = np.c_[np.ones(10), np.linspace(X_art[:, 1].min(), X_art[:, 1].max(), 10) ]

plt.scatter(X_art[:,1], y_art)
plt.plot(X_line[:, 1], X_line @ w_art, color="red")
plt.show()