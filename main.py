"""
I'm going to emulate the example of MLP in COMP5511
"""

import numpy as np

inputlayer = np.array([[1, 0.6, 0.1]])         # shape is (1, 3)
desired = np.array([[1, 0]])
theta1 = np.array([[0.1, 0.2, 0.5], [0.1, 0, 0.3], [-0.2, 0.2, -0.4]])   # (3, 3)
theta2 = np.array([[-0.1, 0.6], [-0.4, 0.2], [0.1, -0.1], [0.6, -0.2]])  # (4, 2)
#theta3 = np.random.random((3, 1))
dummyfeature = np.ones((1, 1))
learning_rate = 0.1
"""
# for test
print(dummyfeature.shape)
features = np.vstack((dummyfeature, inputlayer))
print(features)
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feedforward():
    # inputfeatures = np.hstack((dummyfeature, inputlayer))   # shape is (4, 3)
    hidden1 = np.dot(inputlayer, theta1)                    # shape is (1, 3)
    hidden1out = sigmoid(hidden1)                           # shape is (1, 3)
    hidden1features = np.hstack((dummyfeature, hidden1out)) # shape is (1, 4)
    output = np.dot(hidden1features, theta2)               # shape is (1, 2)
    output = sigmoid(output)                               # shape is (1, 2)
    """
    hidden2features = np.hstack((dummyfeature, hidden2out)) # shape is (4, 3)
    output = np.dot(hidden2features, theta3)                # shape is (4, 1)
    output = sigmoid(output)                                # shape is (4, 1)
    """
    return inputlayer, hidden1out, hidden1features, output


def cost(output):
    return np.sum(np.square(output - desired)) / 2


def backpropagation():
    global theta3, theta2, theta1
    loss = 4
    epoch = 0
    best = None
    while loss >= 0.001 or epoch < 2000:
        inputfeatures, hidden1out, hidden1features, output = feedforward()
        loss = cost(output)
        print(hidden1features)
        print(output)
        """
        outputlocalgradient = (desired - output) * output * (1 - output)       # (4, 1)
        gradient3 = np.dot(hidden2features.T, outputlocalgradient)             # (3, 1)

        hidden2localgradient = np.dot(outputlocalgradient, theta3.T)[:, -2:]   # (4, 2)
        hidden2localgradient = hidden2localgradient * hidden2out * (1 - hidden2out)
        gradient2 = np.dot(hidden1features.T, hidden2localgradient)            # (3, 2)

        hidden1localgradient = np.dot(hidden2localgradient, theta2.T)[:, -2:]  # (4, 2)
        hidden1localgradient = hidden1localgradient * hidden1out * (1 - hidden1out)
        gradient1 = np.dot(inputfeatures.T, hidden1localgradient)


        theta3 += learning_rate * gradient3
        theta2 += learning_rate * gradient2
        theta1 += learning_rate * gradient1
        """
        outputlocalgradient = (desired - output) * output * (1 - output)  # (1, 2)
        gradient2 = np.dot(hidden1features.T, outputlocalgradient)        # (4, 2)

        hidden1localgradient = np.dot(outputlocalgradient, theta2.T)[:, -3:]      # (1, 3)
        hidden1localgradient = hidden1localgradient * hidden1out * (1 - hidden1out)
        gradient1 = np.dot(inputfeatures.T, hidden1localgradient)

        theta2 += learning_rate * gradient2
        theta1 += learning_rate * gradient1

        print(theta2)

        print(theta1)

        best = output
        print(loss)
        epoch += 1

    print(theta1)
    print("----")
    print(theta2)
    print("----")
    print(theta3)
    print("----")
    print(best)


if __name__ == "__main__":
    backpropagation()