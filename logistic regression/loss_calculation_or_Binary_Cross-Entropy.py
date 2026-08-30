# Binary Cross-Entropy because of Binary Classification (0 or 1)
# Also called Log Loss

import numpy as np

# Input
x = 2

# Actual answer (0 or 1)
y = 1

# Current weight and bias
w = 0.5
b = 0

# --------------------------------------------------
# FORWARD PROPAGATION
# --------------------------------------------------

# Step 1: Linear calculation
z = w * x + b
print("z:", z)

# Step 2: Convert z into probability
# p is the predicted probability of class 1
p = 1 / (1 + np.exp(-z))
print("Probability:", p)

# Step 3: Calculate Log Loss
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
print("Loss:", loss)


# --------------------------------------------------
# BACKWARD PROPAGATION
# --------------------------------------------------

# Step 4: Partial derivative of Loss with respect to p
dL_dp = -y / p + (1 - y) / (1 - p)
print("dL/dp:", dL_dp)

# Step 5: Partial derivative of p with respect to z
# Derivative of sigmoid
dp_dz = p * (1 - p)
print("dp/dz:", dp_dz)

# Step 6: Partial derivative of z with respect to w
# z = w*x + b
dz_dw = x
print("dz/dw:", dz_dw)

# Step 7: Chain Rule
dL_dw = dL_dp * dp_dz * dz_dw
print("dL/dw:", dL_dw)