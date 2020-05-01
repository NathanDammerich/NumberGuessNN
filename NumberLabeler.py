from random import randint
from PIL import Image
import numpy as np
import chainer
from NN import NeuralNetwork

global training_inputs, temp_training_outputs, test_inputs, test_outputs

# Initialize NumberNet as an object of NeuralNetwork class
NumberNet = NeuralNetwork()

# Load data from MNIST database
data = chainer.datasets.get_mnist(withlabel=True)


def reformatData(data):
    global training_inputs, temp_training_outputs, test_inputs, test_outputs

    # Get training inputs and outputs into the correct format (numpy arrays)
    training_input = [None for i in range(60000)]
    temp_training_outputs = [None for i in range(60000)]
    test_input = [None for i in range(60000)]
    test_outputs = [None for i in range(60000)]

    for i in range(60000):
        interim = data[0][i]
        test_interim = data[0][i]
        training_input[i] = interim[0]
        temp_training_outputs[i] = interim[1]
        test_input[i] = test_interim[0]
        test_outputs[i] = test_interim[1]

    training_inputs = np.array(training_input)
    training_inputs = training_inputs.T

    test_inputs = np.array(test_input)
    test_inputs = test_inputs.T


def train(iterations, rate):
    global training_inputs, temp_training_outputs, test_inputs, test_outputs
    NeuralNetwork.train(NumberNet, training_inputs=training_inputs, training_outputs=temp_training_outputs,
                        training_iterations=iterations, learning_rate=rate)


def test(num_tests):
    global training_inputs, temp_training_outputs, test_inputs, test_outputs
    correct = 0
    incorrect = 0

    # Test NN amount of times user specifies with num_tests
    for i in range(num_tests):
        # Get random test data
        r = randint(0, 59999)
        # Get test data in correct format
        new_test_input = np.array([[0.0 for x in range(784)]])
        temp = test_inputs[:, r]
        for j in range(784):
            new_test_input[0][j] = temp[j]
        new_test_input = new_test_input.T

        # Get NN's guess for input
        nnGuess = NeuralNetwork.think(NumberNet, test_input=new_test_input)

        # Log test results
        if test_outputs[r] == nnGuess:
            correct += 1
        else:
            incorrect += 1

    # Print test results
    print("Correct: " + str(correct))
    print("Incorrect: " + str(incorrect))


def testWithInput(input_from_user):
    nnGuess = NeuralNetwork.think(NumberNet, test_input=input_from_user)
    print("My guess is... " + str(nnGuess))


def convert_image():  # Converts from image_input.png, 700x700 pixels, returns [784, 1] array
    # Open image, converting to grayscale
    img = Image.open('image_input.png').convert('L')
    WIDTH, HEIGHT = img.size
    pic = list(img.getdata())

    # Convert that to 2D list (list of lists of integers)
    pic = [pic[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    # Convert to grayscale between 0 and 1
    temp_input_from_user = np.array(pic)
    for i in range(700):
        for j in range(700):
            temp_input_from_user[i][j] = (255 - temp_input_from_user[i][j]) / 255

    # Take average of pixels in 25x25 grids, and assign value to corresponding location of input
    input_from_user = np.empty(shape=(784, 1))
    for i in range(28):
        for j in range(28):
            totalPix = 0
            for a in range(25):
                for b in range(25):
                    totalPix += temp_input_from_user[i * 25 + a][j * 25 + b]
            totalPix = totalPix / 625
            input_from_user[(28 * i) + j] = totalPix

    return input_from_user

# Let NN decide what number user enters by running testWithInput
myNum = convert_image()
testWithInput(myNum)
