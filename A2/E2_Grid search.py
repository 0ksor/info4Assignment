import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Generating random uniform distribution datapoints x from a range [x_min, x_max].
# The inputs a and b are slope and intercept of the linear function respectively and noise_std is the standard deviation of the Gaussian noise.
def make_data(n=60, x_min=0.0, x_max=5.0, a=1.5, b=2.0, noise_std=0.6, seed=10):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n)
    y_true = a * x + b
    y = y_true + rng.normal(0.0, noise_std, size=n)
    return x, y, a, b


# splitting the dataset into training and testing sets with test data ratio of 25%
# and shuffle the data using the given seed
# Returns the training inputs, training labels, testing inputs, and testing labels
def split_train_test(x, y, test_ratio=0.25, seed=123):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    split_index = int(len(x) * (1 - test_ratio))

    train_X = x[indices[:split_index]]
    train_y = y[indices[:split_index]]
    test_X = x[indices[split_index:]]
    test_y = y[indices[split_index:]]
    return train_X, train_y, test_X, test_y


# Hypothesis function with input parameters theta0, theta1 and x and returns the predicted y values.
def hypothesis(theta0, theta1, x):
    return theta0 + theta1 * x


# Cost function between y values and yhat(predicted) values
def cost(y, yhat):
    return np.mean((yhat - y) ** 2) / 2


# Grid search function with input parameters x, y, theta0_values and theta1_values
# returns the best_theta0, best_theta1 and cost_grid(2D array of all combination of cost values)
def grid_search(x, y, theta0_values, theta1_values):
    raise NotImplementedError


# Plotting function to visualize the training and testing data points
def plot_before_after(x_train, y_train, x_test, y_test, a, b, theta0, theta1, title):
    x_grid = np.linspace(
        min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 200
    )
    y_true_grid = a * x_grid + b
    y_pred_grid = hypothesis(theta0, theta1, x_grid)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, color="blue", label="Training set")
    plt.scatter(x_test, y_test, color="red", label="Testing set")
    plt.plot(
        x_grid,
        y_pred_grid,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Hypothesis",
    )
    plt.ylim(-1, 12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()


# Plotting function to visualize the cost values for each combination of theta0 and theta1 as a 2D heatmap.
# The best point is highlighted on the heatmap.
def plot_2d_heatmap(theta0_values, theta1_values, cost_grid, best_theta0, best_theta1):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        cost_grid,
        origin="lower",
        aspect="auto",
        extent=[
            theta1_values[0],
            theta1_values[-1],
            theta0_values[0],
            theta0_values[-1],
        ],
    )
    plt.colorbar(label="cost")
    plt.scatter(
        best_theta1,
        best_theta0,
        color="white",
        edgecolors="black",
        s=100,
        label="Best point",
    )
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_0$")
    plt.title(r"cost value for each $(\theta_0, \theta_1)$")
    plt.legend()


# Plotting function to visualize the cost values for each combination of theta0 and theta1 as a 3D plot.
def plot_3d_cost_surface(
    theta0_values, theta1_values, cost_grid, best_theta0, best_theta1, best_cost
):
    T1, T0 = np.meshgrid(theta1_values, theta0_values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(T0, T1, cost_grid, alpha=0.8)
    fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1, label="cost")
    ax.scatter(
        best_theta0, best_theta1, best_cost, color="red", s=80, label="Best point"
    )
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel("cost")
    ax.set_title(r"Cost Surface over $\theta_0$ and $\theta_1$")
    ax.legend()


# Main function to execute the steps of data generation, splitting, training, evaluation and plotting.
def main():
    # Generate data
    x, y, a, b = make_data()

    # Split data
    x_train, y_train, x_test, y_test = split_train_test(x, y)

    # Plot before training
    plot_before_after(
        x_train,
        y_train,
        x_test,
        y_test,
        a,
        b,
        theta0=0.0,
        theta1=0.0,
        title="Before training",
    )

    # Candidate parameter values
    theta0_values = np.linspace(0.0, 4.0, 60)
    theta1_values = np.linspace(0.5, 2.5, 60)

    # Grid search
    best_theta0, best_theta1, cost_grid = grid_search(
        x_train, y_train, theta0_values, theta1_values
    )

    # Evaluate best model
    train_cost = cost(y_train, hypothesis(best_theta0, best_theta1, x_train))
    test_cost = cost(y_test, hypothesis(best_theta0, best_theta1, x_test))

    print("Best parameters")
    print(r"theta0 =", float(best_theta0))
    print(r"theta1 =", float(best_theta1))
    print("Train cost =", float(train_cost))
    print("Test cost  =", float(test_cost))

    # Plot after training
    plot_before_after(
        x_train,
        y_train,
        x_test,
        y_test,
        a,
        b,
        theta0=best_theta0,
        theta1=best_theta1,
        title="After training",
    )

    # Plot 2D heatmap
    plot_2d_heatmap(theta0_values, theta1_values, cost_grid, best_theta0, best_theta1)

    # Plot 3D cost surface
    plot_3d_cost_surface(
        theta0_values, theta1_values, cost_grid, best_theta0, best_theta1, train_cost
    )

    plt.show()


if __name__ == "__main__":
    main()
