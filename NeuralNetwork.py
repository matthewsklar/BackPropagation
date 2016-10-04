# Imports
import random
import TransferFunction
import Utils

# The neurons making up neuron layers
class Neuron(object):
    def __init__(self, inputs):
        # Number if inputs in the neuron
        self.inputs = inputs

        # Weights in the neuron
        self.weights = []

        # Bias of the neuron
        self.bias = random.random() * 2 - 1

        # The error of the neuron
        self.delta = 0

        # The output of the neuron
        self.output = 0

        # Create the weights
        self.create_weights()

    '''
    Add a random number between -1 and 1 for each weight
    '''
    def create_weights(self):
        for x in range(0, self.inputs):
            self.weights.append(random.random() * 2 - 1)


class NeuronLayer(object):
    def __init__(self, num_neurons, inputs_per_neuron):
        # Number of neurons in the layer
        self.num_neurons = num_neurons

        # Neurons in the layer
        self.neurons = []

        # Create the neurons
        self.create_neurons(inputs_per_neuron)

    '''
    Add the neurons to the layer
    '''
    def create_neurons(self, inputs):
        for i in range(0, self.num_neurons):
            self.neurons.append(Neuron(inputs))


class NeuralNetwork(object):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_neurons_per_hidden_layer):
        # Number of input neurons
        self.num_inputs = num_inputs

        # Number of output neurons
        self.num_outputs = num_outputs

        # Number of hidden layers
        self.num_hidden_layers = num_hidden_layers

        # Number of neurons in a hidden layer
        self.num_neurons_per_hidden_layer = num_neurons_per_hidden_layer

        # Layers
        self.layers = []

    '''
    Create the neural network
    '''
    def create_network(self):
        # Create the input layer
        self.layers.append(NeuronLayer(self.num_inputs, 1))

        # Create the hidden layers
        for i in range(0, self.num_hidden_layers):
            self.layers.append(NeuronLayer(self.num_neurons_per_hidden_layer, self.layers[i - 1].num_neurons - 1))

        # Create the output layer
        self.layers.append(NeuronLayer(self.num_outputs, self.num_neurons_per_hidden_layer))

    '''
    Calculate the output layer
    '''
    def calculate_network(self, inputs):
        # Stores the resultant outputs from each layer
        outputs = []

        # Check that the correct amount of inputs are available
        if len(inputs) != self.num_inputs:
            print("Aborting network: Inconsistent inputs for input size")

            return outputs

        for i in range(0, self.num_hidden_layers + 1):
            # If it is not the first layer, the inputs are the previous outputs
            if i > 0:
                inputs.clear()

                for o in outputs:
                    inputs.append(o)

            outputs.clear()
            corr_weight = 0

            # For each neuron in the layer
            for j in self.layers[i + 1].neurons:
                sum = 0

                # For each weight in the neuron
                for k in j.weights:
                    # Add the product (weight * input) to the sum
                    sum += k * inputs[corr_weight]

                    corr_weight += 1

                # Add the bias to the sum
                # TODO: Multiply bias by constant
                sum += j.bias

                # Filter the output of the neuron then add it to the outputs
                j.output = TransferFunction.Sigmoid(sum)
                outputs.append(j.output)

                corr_weight = 0

        return outputs

    def back_propagation(self, expected_output):
        # For each neuron layer in reverse order (0 is output layer)
        for i in range(len(self.layers) - 1, 0, -1):
            # The current layer in reverse order
            layer = self.layers[i]

            # For each neuron in layer i
            for j in layer.neurons:
                error_factor = 0

                if i == len(self.layers) - 1:
                    '''
                        The neuron is in the output layer
                        error factor = expected output - actual output
                    '''
                    error_factor = expected_output - j.output
                else:
                    '''
                        The neuron is in a hidden layer
                        error factor = sum of all (delta of connected neurons * connecting bias)
                    '''
                    next_layer = self.layers[i + 1]

                    for k in range(0, len(j.weights)):
                        error_factor += next_layer.neurons[k].delta * j.weights[k]

                # Set the delta (error) of the neuron
                # delta = output * (1 - output) * error factor
                j.delta = j.output * (1 - j.output) * error_factor

        # For each neuron layer in the network
        for i in range(0, len(self.layers)):
            layer = self.layers[i]

            # For each neuron in the layer
            for j in range(0, layer.num_neurons):
                # The neuron
                neuron = layer.neurons[j]

                neuron.bias += Utils.LEARNING_RATE * neuron.delta

                # For each weight in the neuron
                for k in range(0, len(neuron.weights)):
                    neuron.weights[k] += Utils.LEARNING_RATE * layer.neurons[j - 1].output * neuron.delta
