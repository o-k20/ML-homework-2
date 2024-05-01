import numpy as np

# Function to compute F(x, y)
def F(x, y):
    return np.exp(1j * y) + 2j * np.exp(-1j * y / 2) * np.cos(np.sqrt(3) / 2 * x)

# Generate a grid of x and y values
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)

# Array to store eigenvalues
eigenvalues = np.zeros((2, 100, 100), dtype=complex)

# Calculate eigenvalues for each (x, y) on the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        F_val = F(X[i, j], Y[i, j])
        F_conj = np.conj(F_val)
        M = np.array([[0, F_val], [F_conj, 0]], dtype=complex)
        vals, _ = np.linalg.eig(M)  # Diagonalize M using numpy.linalg.eig
        eigenvalues[:, i, j] = vals


