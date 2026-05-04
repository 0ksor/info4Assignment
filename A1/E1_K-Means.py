import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Random initialization of centroids with inputs
# x(iris dataset) and K(number of clusters)
# and returns the initial centroid values.
def initialize_centroids(x, K, seed=None):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    return x[rng.choice(n, size=K, replace=False)]


# squared Euclidean Distance calculation with inputs a and b
def calculate_distances(a, b):
    size = len(a)
    c = a - b
    temp = 0
    for i in range(size):
        temp += c[i] ** 2
    return temp


# Assigning the values to clusters with inputs
# x(Iris datapoints) and c(centroids) and returns the best cluster label
def assign(x, c):
    y = np.empty(x.shape[0], dtype=int)
    for i in range(len(x)):
        dists = [calculate_distances(x[i], cj) for cj in c]
        print(dists)
        y[i] = int(np.argmin(dists))
    return y


# Update and move centroids with inputs
# x(iris dataset), y(cluster label) and K(number of clusters)
# and returns the updated centroids
def move_centroids(x, y, K, seed=None):
    _, dim = x.shape
    new_centroids = np.zeros((K, dim), dtype=float)
    for i in range(K):
        assigned_values = x[y == i]
        if assigned_values.size == 0:
            new_centroids[i] = initialize_centroids(x, 1, seed)[0]
        else:
            new_centroids[i] = np.mean(assigned_values, axis=0)
    return new_centroids


# Cost function with input
# x(Iris Dataset), c(centroids) and y(cluster label) and returns the cost value
def cost(x, c, y):
    total_cost = 0
    centroids_for_points = c[y]
    for i in range(len(x)):
        total_cost += calculate_distances(x[i], centroids_for_points[i])
    return total_cost


# Function to print the number of interation,
# Euclidean distance between points, cluster labels and centroids  with inputs
# it(number of iteration) and
# show_n(number of points to be printed with default set as 8)
def show_iter(x, c, y, it, show_n=8):
    m = min(show_n, x.shape[0])
    K = c.shape[0]
    dmat = np.zeros((m, K))
    for i in range(m):
        for j in range(K):
            dmat[i, j] = calculate_distances(x[i], c[j])

    print("")
    print("iteration", it + 1)
    print("distances (first", m, "points)")
    print(dmat)
    print("cluster labels (first", m, "points)")
    print(y[:m])
    print("centroids")
    print(c)


# K-Means algorithm function with maximum iteration set to 100, tolerance limit
# to 1e-6 and show_steps set to 2 and show_n set to 8
def kmeans(x, K, max_iter=100, tol=1e-6, seed=None, show_steps=2, show_n=8):
    np.set_printoptions(precision=3, suppress=True)

    c = initialize_centroids(x, K, seed)
    cost_list = []
    y = None

    for it in range(max_iter):
        y = assign(x, c)

        if it < show_steps:
            show_iter(x, c, y, it, show_n=show_n)

        c_new = move_centroids(x, y, K, seed)
        shift = np.max(np.linalg.norm(c_new - c, axis=1))
        c = c_new

        j = cost(x, c, y)
        cost_list.append(j)

        if it < show_steps:
            print("updated centroids")
            print(c)
            print("cost")
            print(float(j))

        if shift <= tol:
            break

    return c, y, cost_list


# runing k-means multiple times with different random starting points
# and recording the best result.
def best_run(x, K, runs=10, max_iter=100, tol=1e-6, show_steps=2, show_n=8):
    best = None

    for seed in range(runs):
        c, y, cost_list = kmeans(
            x,
            K,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            show_steps=show_steps,
            show_n=show_n,
        )
        final_j = cost_list[-1]

        if (best is None) or (final_j < best["final_j"]):
            best = {
                "seed": seed,
                "iters": len(cost_list),
                "final_j": final_j,
                "c": c,
                "y": y,
                "cost_list": cost_list,
            }

    return best


# dataset
data = load_iris()
x = data.data  # type: ignore[attr-defined]

# Number of clusters
K = 3

best: dict = best_run(x, K, runs=10, max_iter=100, tol=1e-6,
                      show_steps=2, show_n=8)  # type: ignore[attr-defined]

print("")
print("best labels:", best["y"])
print("iterations:", best["iters"])
print("final cost:", float(best["final_j"]))
print("centroids:")
print(best["c"])

plt.figure(figsize=(6, 4))

x_iter = np.arange(1, len(best["cost_list"]) + 1)
y_cost = np.array(best["cost_list"])


plt.plot(x_iter, y_cost, "o", label="cost")

plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("k-means convergence (best run)")
plt.grid(True)
plt.legend()
plt.show()
