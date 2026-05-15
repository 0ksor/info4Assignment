import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation, PillowWriter


# Add a bias term to
# the input so that each
# input vector becomes [1, x_1, x_2, ...]
def add_bias(x):
    n = x.shape[0]
    d = x.shape[1]
    out = np.zeros((n, d + 1), dtype=float)
    i = 0
    while i < n:
        out[i, 0] = 1.0
        j = 0
        while j < d:
            out[i, j + 1] = x[i, j]
            j = j + 1
        i = i + 1
    return out


# predicting values based on hypothesis function
# t shows linear regression parameters
def predicted_values(x, t):
    raise NotImplementedError


# Calculate cost value
def cost(x, y, t):
    raise NotImplementedError


# Fit Batch Gradient Descent
# return cost history and theta
def fit_bgd(x, y, lr, steps, seed=0):
    raise NotImplementedError


# Fit Stochastic Gradient Descent
# return cost history and theta
def fit_sgd(x, y, lr, epochs, seed=0):
    raise NotImplementedError


# creating a rotating 3D animation of the updated regression plane
def plot_3d_slice_gif(
    x_no_bias,
    y,
    t,
    f1,
    f2,
    name1,
    name2,
    out_gif="rotating_3d_plane.gif",
    fps=20,
    dpi=120,
    elev=25,
):
    from mpl_toolkits.mplot3d import Axes3D

    # Selected features to be shown
    a = x_no_bias[:, f1]
    b = x_no_bias[:, f2]

    a0 = float(a.min())
    a1 = float(a.max())
    b0 = float(b.min())
    b1 = float(b.max())

    ag = np.linspace(a0, a1, 25)
    bg = np.linspace(b0, b1, 25)
    A, B = np.meshgrid(ag, bg)

    z = t[0] + t[f1 + 1] * A + t[f2 + 1] * B

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(a, b, y, s=60, marker="o", label="data")
    ax.plot_surface(A, B, z, alpha=0.25)

    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_zlabel("y")
    ax.set_title("trained plane slice (other features = 0)")
    ax.legend()

    pick = [0, 10]
    r = 0
    while r < len(pick):
        i = pick[r]
        ya = t[0] + t[f1 + 1] * a[i] + t[f2 + 1] * b[i]
        ax.quiver(
            a[i], b[i], ya, 0.0, 0.0, y[i] - ya, arrow_length_ratio=0.12, linewidth=2
        )
        r = r + 1

    def update(angle):
        ax.view_init(elev=elev, azim=angle)
        return (fig,)

    angles = np.arange(0, 360, 2)  # full rotation
    ani = FuncAnimation(fig, update, frames=angles, interval=50, blit=False)

    ani.save(out_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    print("Saved GIF:", out_gif)

    plt.show()


# Main part

# read dataset
df = pd.read_csv("Real_estate.csv")
y = df["Y house price of unit area"].to_numpy(dtype=float)
x = df.drop("Y house price of unit area", axis=1).to_numpy(dtype=float)
names = df.drop("Y house price of unit area", axis=1).columns.tolist()

# Standardization
sc = StandardScaler()
x = sc.fit_transform(x)

# Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

x_train_b = add_bias(x_train)
x_test_b = add_bias(x_test)

# Learning rates
bgd_lr_list = [0.01, 0.05, 0.1]
sgd_lr_list = [0.001, 0.005, 0.01]

# Steps and Epochs
bgd_steps = 600
sgd_epochs = 60

# Apply BGD on test dataset
bgd_hist_list = []
bgd_lr_used = []
bgd_test_list = []
bgd_t_list = []

i = 0
while i < len(bgd_lr_list):
    lr = bgd_lr_list[i]
    t, h = fit_bgd(x_train_b, y_train, lr, bgd_steps, seed=0)

    test_cost = cost(x_test_b, y_test, t)

    bgd_lr_used.append(lr)
    bgd_hist_list.append(h)
    bgd_test_list.append(test_cost)
    bgd_t_list.append(t)
    i = i + 1

print("BGD test Cost")
i = 0
while i < len(bgd_lr_used):
    print(bgd_lr_used[i], float(bgd_test_list[i]))
    i = i + 1


# Apply SGD on test dataset
sgd_hist_list = []
sgd_lr_used = []
sgd_test_list = []
sgd_t_list = []

i = 0
while i < len(sgd_lr_list):
    lr = sgd_lr_list[i]
    t, h = fit_sgd(x_train_b, y_train, lr, sgd_epochs, seed=0)

    test_cost = cost(x_test_b, y_test, t)

    sgd_lr_used.append(lr)
    sgd_hist_list.append(h)
    sgd_test_list.append(test_cost)
    sgd_t_list.append(t)
    i = i + 1

print("SGD test Cost")
i = 0
while i < len(sgd_lr_used):
    print(sgd_lr_used[i], float(sgd_test_list[i]))
    i = i + 1


# Convergence Plots
# BGD
plt.figure(figsize=(7, 4))
i = 0
while i < len(bgd_lr_used):
    plt.plot(bgd_hist_list[i], label=str(bgd_lr_used[i]))
    i = i + 1
plt.xlabel("step")
plt.ylabel("cost value")
plt.title("BGD convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# SGD
plt.figure(figsize=(7, 4))
i = 0
while i < len(sgd_lr_used):
    plt.plot(sgd_hist_list[i], label=str(sgd_lr_used[i]))
    i = i + 1
plt.xlabel("epoch")
plt.ylabel("cost")
plt.title("SGD convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Best parameters
# BGD
best_i = 0
i = 1
while i < len(bgd_test_list):
    if bgd_test_list[i] < bgd_test_list[best_i]:
        best_i = i
    i = i + 1
t_best = bgd_t_list[best_i]

print("Best BGD learning rate:", bgd_lr_used[best_i])
print("Best parameter vector (theta):")
print(t_best)

# SGD
best_sgd_i = 0
i = 1
while i < len(sgd_test_list):
    if sgd_test_list[i] < sgd_test_list[best_sgd_i]:
        best_sgd_i = i
    i = i + 1
t_best_sgd = sgd_t_list[best_sgd_i]

print("Best SGD learning rate:", sgd_lr_used[best_sgd_i])
print("Best parameter vector (theta):")
print(t_best_sgd)

# Comparison
print("Best BGD test cost:", bgd_test_list[best_i])
print("Best SGD test cost:", sgd_test_list[best_sgd_i])

# Chosen features for 3D plot by BGD method
f1 = 2
f2 = 3
plot_3d_slice_gif(
    x_train,
    y_train,
    t_best,
    f1,
    f2,
    names[f1],
    names[f2],
    out_gif="real_estate_3d_rotation.gif",
)

