from random import randint
import numpy as np


class NeuralNetwork:

    def __init__(self):
        self.bias1 = np.array([[0.0 for x in range(64)]]).T
        self.bias2 = np.array([[0.0 for x in range(64)]]).T
        self.bias3 = np.array([[0.0 for x in range(64)]]).T
        self.bias4 = np.array([[0.0 for x in range(10)]]).T

        self.sw_i_h1 = 2 * np.random.random((64, 784)) - 1
        self.sw_h1_h2 = 2 * np.random.random((64, 64)) - 1
        self.sw_h2_h3 = 2 * np.random.random((64, 64)) - 1
        self.sw_h3_o = 2 * np.random.random((10, 64)) - 1

        # np.save("sw_i_h1.npy", self.sw_i_h1)
        # np.save("sw_h1_h2.npy", self.sw_h1_h2)
        # np.save("sw_h2_h3.npy", self.sw_h2_h3)
        # np.save("sw_h3_o.npy", self.sw_h3_o)
        # np.save("bias1.npy", self.bias1)
        # np.save("bias2.npy", self.bias2)
        # np.save("bias3.npy", self.bias3)
        # np.save("bias4.npy", self.bias4)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations, learning_rate):

        # Load current synaptic weights and biases
        sw_i_h1 = np.load("sw_i_h1.npy")
        sw_h1_h2 = np.load("sw_h1_h2.npy")
        sw_h2_h3 = np.load("sw_h2_h3.npy")
        sw_h3_o = np.load("sw_h3_o.npy")
        bias1 = np.load("bias1.npy")
        bias2 = np.load("bias2.npy")
        bias3 = np.load("bias3.npy")
        bias4 = np.load("bias4.npy")

        # Train training_iterations times, which is passed to the function
        for iteration in range(training_iterations):

            # Choose random value from training set and use it to train
            r = randint(0, 59999)
            new_input = np.array([[0.0 for x in range(784)]])
            new_output = np.array([[0.0 for x in range(10)]])
            temp = training_inputs[:, r]
            for j in range(784):
                new_input[0][j] = temp[j]
            new_input = new_input.T
            for j in range(10):
                if training_outputs[r] == j:
                    new_output[0][j] = 1.0
            new_output = new_output.T

            # Feedforward values
            first_hidden_values = self.sigmoid(np.dot(sw_i_h1, new_input) + bias1)
            second_hidden_values = self.sigmoid(np.dot(sw_h1_h2, first_hidden_values) + bias2)
            third_hidden_values = self.sigmoid(np.dot(sw_h2_h3, second_hidden_values) + bias3)
            outputs = self.sigmoid(np.dot(sw_h3_o, third_hidden_values) + bias4)

            # Calc Error then Gradient H3-O and Adjustment
            output_errors = new_output - outputs
            G_H3_O = learning_rate * self.sigmoid_derivative(outputs) * output_errors
            delta_sw_h3_o = np.dot(G_H3_O, third_hidden_values.T)
            sw_h3_o += delta_sw_h3_o
            bias4 += G_H3_O

            # Calc Error then Gradient H2-H3 and Adjustment
            third_hidden_errors = np.dot(sw_h3_o.T, output_errors)
            G_H2_H3 = learning_rate * self.sigmoid_derivative(third_hidden_values) * third_hidden_errors
            delta_sw_hh = np.dot(G_H2_H3, second_hidden_values.T)
            sw_h2_h3 += delta_sw_hh
            bias3 += G_H2_H3

            # Calc Error then Gradient H1-H2 and Adjustment
            second_hidden_errors = np.dot(sw_h2_h3.T, third_hidden_errors)
            G_H1_H2 = learning_rate * self.sigmoid_derivative(second_hidden_values) * second_hidden_errors
            delta_sw_h1_h2 = np.dot(G_H1_H2, first_hidden_values.T)
            sw_h1_h2 += delta_sw_h1_h2
            bias2 += G_H1_H2

            # Calc Error then Gradient I-H1 and Adjustment
            first_hidden_errors = np.dot(sw_h1_h2.T, second_hidden_errors)
            G_I_H1 = learning_rate * self.sigmoid_derivative(first_hidden_values) * first_hidden_errors
            delta_sw_i_h1 = np.dot(G_I_H1, new_input.T)
            sw_i_h1 += delta_sw_i_h1
            bias1 += G_I_H1

        # Save new synaptic weights and biases
        np.save("sw_i_h1.npy", sw_i_h1)
        np.save("sw_h1_h2.npy", sw_h1_h2)
        np.save("sw_h2_h3.npy", sw_h2_h3)
        np.save("sw_h3_o.npy", sw_h3_o)
        np.save("bias1.npy", bias1)
        np.save("bias2.npy", bias2)
        np.save("bias3.npy", bias3)
        np.save("bias4.npy", bias4)

    def think(self, test_input):

        # Load synaptic weights and biases
        sw_i_h1 = np.load("sw_i_h1.npy")
        sw_h1_h2 = np.load("sw_h1_h2.npy")
        sw_h2_h3 = np.load("sw_h2_h3.npy")
        sw_h3_o = np.load("sw_h3_o.npy")
        bias1 = np.load("bias1.npy")
        bias2 = np.load("bias2.npy")
        bias3 = np.load("bias3.npy")
        bias4 = np.load("bias4.npy")

        # Feedforward input with preloaded synaptic weights and biases
        first_hidden_values = self.sigmoid(np.dot(sw_i_h1, test_input) + bias1)
        second_hidden_values = self.sigmoid(np.dot(sw_h1_h2, first_hidden_values) + bias2)
        third_hidden_values = self.sigmoid(np.dot(sw_h2_h3, second_hidden_values) + bias3)
        outputs = self.sigmoid(np.dot(sw_h3_o, third_hidden_values) + bias4)
        maxNum = outputs[0]

        # Return the max output
        max_index = 0
        for i in range(10):
            if outputs[i] > maxNum:
                maxNum = outputs[i]
                max_index = i
        return max_index
