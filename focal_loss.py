import numpy as np

alpha = 0.25
gama = 2

y_true = np.array([1,    0,    1,   0])
y_pred = np.array([0.95, 0.05, 0.5, 0.5])

alpha_weights = [alpha if y == 1 else 1 - alpha for y in y_true]
print(alpha_weights)

pt = np.zeros(4)

index1 = np.argwhere(y_true == 1)
index0 = np.argwhere(y_true == 0)

pt[index1] = (1 - y_pred[index1])**gama
pt[index0] = (y_pred[index0])**gama

weights = pt * alpha_weights

print(weights)
