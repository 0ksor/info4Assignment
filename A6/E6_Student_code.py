import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int64)

X = X / 255.0

num_classes = 10
Y = np.eye(num_classes, dtype=np.float32)[y]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.14285, random_state=42
)

input_size = 784
h1_size = 64
h2_size = 64
output_size = 10

np.random.seed(42)
Theta1 = np.random.randn(h1_size, input_size + 1).astype(np.float32) * np.sqrt(
    2.0 / input_size
)
Theta2 = np.random.randn(h2_size, h1_size + 1).astype(np.float32) * np.sqrt(
    2.0 / h1_size
)
Theta3 = np.random.randn(output_size, h2_size + 1).astype(np.float32) * np.sqrt(
    2.0 / h2_size
)

Theta1[:, 0] = 0.0
Theta2[:, 0] = 0.0
Theta3[:, 0] = 0.0


# ── Activation functions ──────────────────────────────────────────────────────


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(np.float32)


def softmax(Z):
    # Subtract row-wise max for numerical stability
    Z_shift = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


# ── Forward propagation ───────────────────────────────────────────────────────


def forward(X, Theta1, Theta2, Theta3):
    """
    X : (m, 784)
    Returns y_hat : (m, 10)
    """
    m = X.shape[0]

    # Layer 1
    X_bias = np.hstack([np.ones((m, 1), dtype=np.float32), X])  # (m, 785)
    Z1 = X_bias @ Theta1.T  # (m, 64)
    A1 = relu(Z1)  # (m, 64)

    # Layer 2
    A1_bias = np.hstack([np.ones((m, 1), dtype=np.float32), A1])  # (m, 65)
    Z2 = A1_bias @ Theta2.T  # (m, 64)
    A2 = relu(Z2)  # (m, 64)

    # Output layer
    A2_bias = np.hstack([np.ones((m, 1), dtype=np.float32), A2])  # (m, 65)
    Z3 = A2_bias @ Theta3.T  # (m, 10)
    y_hat = softmax(Z3)  # (m, 10)

    return y_hat


# ── Cost function ─────────────────────────────────────────────────────────────


def cross_entropy_cost(y_hat, Y_true):
    """
    y_hat  : (m, 10) — predicted probabilities
    Y_true : (m, 10) — one-hot ground truth
    Returns scalar average cross-entropy loss.
    """
    m = Y_true.shape[0]
    # Clip for numerical safety
    log_probs = np.log(np.clip(y_hat, 1e-12, 1.0))
    cost = -np.sum(Y_true * log_probs) / m
    return float(cost)


# ── Backpropagation ───────────────────────────────────────────────────────────


def backward(X, Y_true, Theta1, Theta2, Theta3):
    """
    Returns gradients dTheta1, dTheta2, dTheta3 — same shapes as the weights.
    """
    m = X.shape[0]

    # ---- Forward pass (save intermediates) ----------------------------------
    X_bias = np.hstack([np.ones((m, 1), dtype=np.float32), X])  # (m, 785)
    Z1 = X_bias @ Theta1.T  # (m, 64)
    A1 = relu(Z1)  # (m, 64)

    A1_bias = np.hstack([np.ones((m, 1), dtype=np.float32), A1])  # (m, 65)
    Z2 = A1_bias @ Theta2.T  # (m, 64)
    A2 = relu(Z2)  # (m, 64)

    A2_bias = np.hstack([np.ones((m, 1), dtype=np.float32), A2])  # (m, 65)
    Z3 = A2_bias @ Theta3.T  # (m, 10)
    y_hat = softmax(Z3)  # (m, 10)

    # ---- Output layer delta (softmax + cross-entropy combined) ---------------
    delta3 = (y_hat - Y_true) / m  # (m, 10)
    dTheta3 = delta3.T @ A2_bias  # (10, 65)

    # ---- Hidden layer 2 delta -----------------------------------------------
    # Back-prop through Theta3, drop the bias column (index 0)
    delta2 = (delta3 @ Theta3[:, 1:]) * relu_derivative(Z2)  # (m, 64)
    dTheta2 = delta2.T @ A1_bias  # (64, 65)

    # ---- Hidden layer 1 delta -----------------------------------------------
    delta1 = (delta2 @ Theta2[:, 1:]) * relu_derivative(Z1)  # (m, 64)
    dTheta1 = delta1.T @ X_bias  # (64, 785)

    return dTheta1, dTheta2, dTheta3


# ── Training loop ─────────────────────────────────────────────────────────────

learning_rate = 0.1
epochs = 100
costs = []

for epoch in range(epochs):
    y_hat = forward(X_train, Theta1, Theta2, Theta3)
    cost = cross_entropy_cost(y_hat, Y_train)
    costs.append(cost)

    dTheta1, dTheta2, dTheta3 = backward(X_train, Y_train, Theta1, Theta2, Theta3)

    # Gradient descent update
    Theta1 -= learning_rate * dTheta1
    Theta2 -= learning_rate * dTheta2
    Theta3 -= learning_rate * dTheta3

    print(f"Epoch {epoch + 1:3d} / {epochs}  cost = {cost:.4f}")

plt.figure()
plt.plot(costs)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Training cost")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/training_cost.png", dpi=120)
plt.show()


# ── Evaluation ────────────────────────────────────────────────────────────────


def predict(X, Theta1, Theta2, Theta3):
    y_hat = forward(X, Theta1, Theta2, Theta3)
    return np.argmax(y_hat, axis=1)


y_test_true = np.argmax(Y_test, axis=1)
y_test_pred = predict(X_test, Theta1, Theta2, Theta3)
acc = np.mean(y_test_pred == y_test_true)
print("Test Accuracy =", round(acc * 100, 2), "%")


def show_predictions(X, y_true, y_pred, num=10):
    plt.figure(figsize=(12, 3))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        plt.title("T:" + str(y_true[i]) + "\nP:" + str(y_pred[i]))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/predictions.png", dpi=120)
    plt.show()


show_predictions(X_test, y_test_true, y_test_pred, num=10)
