# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np

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

    x, y = data[:, :-1], data[:, -1]
    return x, y

# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    # your code
    sample_size = len(x)
    train_size = int(len(x) * train_proportion)
    x_train, x_test, y_train, y_test = x[:train_size], x[train_size:], y[:train_size], y[train_size:]

    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python
def normal_equation(x, y, lambdaV):
    # your code
    I = np.identity(x.shape[1])
    beta = np.linalg.inv(x.T @ x + lambdaV * I) @ (x.T @ y)
    return beta

# Extra Credit: Find theta using gradient descent
def gradient_descent(x, y, lambdaV, num_iterations, learning_rate):
    # your code
    thetas = []
    theta = np.zeros(len(x[0]))

    for iteration in range(num_iterations):
        gradient = (x @ theta - y) @ x + lambdaV * theta
        gradient /= len(x)
        theta -= learning_rate * gradient
        thetas.append(theta)

    return thetas


# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    loss = ((y - y_predict) ** 2).mean()
    return loss

# Given an array of x and theta predict y
def predict(x, theta):
    # your code
   y_predict = x @ theta
   return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv
def cross_validation(x_train, y_train, lambdas):
    valid_losses = [0] * len(lambdas)
    training_losses = [0] * len(lambdas)
    # your code

    fold_size = int(len(x_train) / 4)
    for i in range(4):
        x_t = np.concatenate([
            x_train[:i * fold_size],
            x_train[(i + 1) * fold_size:]
        ])
        y_t = np.concatenate([
            y_train[:i * fold_size],
            y_train[(i + 1) * fold_size:]
        ])

        x_v = x_train[i * fold_size:(i + 1) * fold_size]
        y_v = y_train[i * fold_size:(i + 1) * fold_size]

        for j, lbd in enumerate(lambdas):
            beta = normal_equation(x_t, y_t, lbd)

            train_loss = get_loss(y_t, predict(x_t, beta))
            val_loss = get_loss(y_v, predict(x_v, beta))

            training_losses[j] += train_loss
            valid_losses[j] += val_loss

    valid_losses = [l / 4 for l in valid_losses]
    training_losses = [l / 4 for l in training_losses]

    return np.array(valid_losses), np.array(training_losses)

def bar_plot(beta):
    plt.bar(list(range(1, len(beta) + 1)), beta)
    plt.title("learned beta")
    plt.show()

if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt") # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss")
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]


    # step 2: analysis
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = np.linalg.norm(normal_beta) # your code get l2 norm of normal_beta
    best_beta_norm = np.linalg.norm(best_beta) # your code get l2 norm of best_beta
    large_lambda_norm = np.linalg.norm(large_lambda_beta) # your code get l2 norm of large_lambda_beta
    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " + str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " + str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " + str(get_loss(y_test, predict(x_test, large_lambda_beta))))
    bar_plot(best_beta)


    # Step3: Retrain a new model using all sampling in training, then report error on testing set
    # your code !


    # Step Extra Credit: Implement gradient descent, analyze and show it gives the same or very similar beta to normal_equation
    # to prove that it works
    betas = gradient_descent(x_train, y_train, best_lambda, 1000, 0.05)
    beta = betas[-1]
    print('first 5 dims of normal equation beta:', best_beta[:5])
    print('first 5 dims of gradient descent beta:', beta[:5])
    beta_norm = np.linalg.norm(beta)
    print("L2 norm of gradient descent beta:  " + str(beta_norm))

    print("Average testing loss for gradient descent beta:  " + str(get_loss(y_test, predict(x_test, beta))))
