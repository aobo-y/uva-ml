# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np
# more imports

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    # your code
    with open(filename) as data_file:
        data = data_file.read()
        data = np.array([
            [float(t) for t in line.split('\t')]
            for line in data.split('\n')
            if line.strip()
        ])

    x, y = data[:, :2], data[:, 2]
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    # your code
    xt = x.transpose()
    theta = np.linalg.inv(xt @ x) @ xt @ y
    return theta

# Find thetas using stochiastic gradient descent
# Don't forget to shuffle
def stochiastic_gradient_descent(x, y, learning_rate, num_iterations):
    # your code
    thetas = []
    theta = np.random.randn(2)

    for iteration in range(num_iterations):
        # shuffle
        idx = np.random.permutation(len(x))

        for i in idx:
            theta -= learning_rate * (x[i] @ theta - y[i]) * x[i]

        thetas.append(theta.copy())

    return thetas

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    # your code
    thetas = []
    theta = np.random.randn(2)

    for iteration in range(num_iterations):
        theta -= learning_rate * (x @ theta - y) @ x

        thetas.append(theta.copy())

    return thetas

# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_iterations, batch_size):
    # your code
    # your code
    thetas = []
    theta = np.random.randn(2)

    for iteration in range(num_iterations):
        # shuffle
        idx = np.random.permutation(len(x))

        for i in range(0, len(idx), batch_size):
            indices = idx[i: i + batch_size]
            x_batch, y_batch = x[indices], y[indices]
            theta -= learning_rate * (x_batch @ theta - y_batch) @ x_batch

        thetas.append(theta.copy())

    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
   # your code
   y_predict = x @ theta
   return y_predict

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    loss = ((y - y_predict) ** 2).mean()
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_loss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()

def main():
    x, y = load_data_set('regression-data.txt')
    plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = gradient_descent(x, y, 1e-3, 100)
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = stochiastic_gradient_descent(x, y, 1e-3, 100) # Try different learning rates and number of iterations
    plot(x, y, thetas[-1], "Stochiastic Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Stochiastic Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = minibatch_gradient_descent(x, y, 1e-3, 100, 10)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Mean Epoch vs Training Loss")


if __name__ == "__main__":
    main()
