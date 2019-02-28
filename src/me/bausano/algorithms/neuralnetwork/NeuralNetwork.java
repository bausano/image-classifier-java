package me.bausano.algorithms.neuralnetwork;

import me.bausano.Settings;
import me.bausano.algorithms.Classifier;

import java.util.Arrays;
import java.util.PrimitiveIterator;
import java.util.Random;

public class NeuralNetwork implements Classifier {

    /**
     * Array of network layers. The input layer is not included and is only abstract. Therefore a network that has
     * structure [64, 36, 10] will have two layers behind the scenes.
     */
    public Layer[] layers;

    /**
     * Used to map neurons to classes. Indices associated neurons and values are output classes.
     */
    public final int[] mapNeuronToDigit;

    /**
     * Used to map neurons to classes. Indices associated neurons and values are output classes.
     */
    public final int[] mapDigitToNeuron;

    /**
     * @param layers Array of network layers
     * @param mapNeuronToDigit Converts neurons to classes
     * @param mapDigitToNeuron Converts classes to neurons
     */
    private NeuralNetwork(Layer[] layers, int[] mapNeuronToDigit, int[] mapDigitToNeuron) {
        this.layers = layers;
        this.mapNeuronToDigit = mapNeuronToDigit;
        this.mapDigitToNeuron = mapDigitToNeuron;
    }

    /**
     * Generates new neural network with randomly assigned weights from given schema. Schema is a list where it's length
     * represents number of layers and each element number of neurons within the layer.
     *
     * @param schema Layers and neurons including input and output layer
     * @param mapNeuronToDigit Holds information about which output neuron represents which class
     * @param mapDigitToNeuron Holds information about which class is bound to which neuron
     * @return New instance of an untrained network
     */
    public static NeuralNetwork fromBlueprint(int[] schema,  int[] mapNeuronToDigit, int[] mapDigitToNeuron) {
        PrimitiveIterator.OfDouble rng = new Random().doubles().iterator();

        Layer[] layers = new Layer[schema.length - 1];

        // Using loops instead of streams for better performance.
        for (int layerIndex = 1; layerIndex < schema.length; layerIndex++) {
            // We create a new layer array (collection of neurons). It's going to have number of neurons according to
            // the schema and each neuron will have weight according to number of neurons in previous layer plus a bias.
            double[][] weights = new double[schema[layerIndex]][schema[layerIndex - 1]];
            double[] biases = new double[schema[layerIndex]];

            // Generate each layer neuron.
            for (int neuronIndex = 0; neuronIndex < schema[layerIndex]; neuronIndex++) {
                // Generate each layer weight based on number of neurons in previous layer.
                for (int weightIndex = 0; weightIndex < schema[layerIndex - 1]; weightIndex++) {
                    // Weight is a double in range <-0.25;0.25).
                    weights[neuronIndex][weightIndex] = (rng.next() - 0.5d) / 2;
                }

                // Sets the bias to be 0. It does not matter how we initially set bias as the activation is always 1.
                biases[neuronIndex] = 0d;
            }

            // Since we are not creating the input layer, we have to decrement layer index by one when assigning it to
            // the layers array.
            layers[layerIndex - 1] = new Layer(weights, biases);
        }

        return new NeuralNetwork(layers, mapNeuronToDigit, mapDigitToNeuron);
    }

    /**
     * Overloading the init method to default the class mapping.
     *
     * @param schema Layers and neurons including input and output layer
     * @return New instance of an untrained network
     */
    public static NeuralNetwork fromBlueprint(int[] schema) {
        return NeuralNetwork.fromBlueprint(
                schema,
                new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }
        );
    }

    /**
     * @inheritDoc
     */
    public int classify (double[] digit) {
        int candidate = 0;
        double candidateProbability = Double.MIN_VALUE;

        // Feeds forward the inputs and gathers the results on output neurons.
        double[] probabilities = feedForward(digit);

        // Each of the probabilities corresponds to one output neuron.
        for (int neuronIndex = 0; neuronIndex < probabilities.length; neuronIndex++) {
            // If candidate's probability is higher than that of iterated neuron, skip.
            if (probabilities[neuronIndex] < candidateProbability ) {
                continue;
            }

            // Set this neuron as the new candidate.
            candidate = neuronIndex;
            candidateProbability = probabilities[neuronIndex];
        }

        return mapNeuronToDigit[candidate];
    }

    /**
     * @inheritDoc
     */
    public double[] estimate(double[] digit) {
        // Default each class with -1, which represents "I don't know".
        double[] digitProbabilities = new double[Settings.OUTPUT_CLASSES_COUNT];

        // Feeds forward the inputs and gathers the results on output neurons.
        double[] neuronProbabilities = feedForward(digit);

        // If the last digit represents IDK (I don't know this class) value (-1), then it prefills all class values with
        // the probability of IDK neuron.
        Arrays.fill(
                digitProbabilities,
                mapNeuronToDigit[mapNeuronToDigit.length - 1] == -1
                        ? neuronProbabilities[neuronProbabilities.length - 1]
                        : -1
        );

        for (int neuronIndex = 0; neuronIndex < neuronProbabilities.length; neuronIndex++) {
            // Neuron -1 has a special meaning. It is a shadow neuron that represents IDK value.
            if (mapNeuronToDigit[neuronIndex] == -1) {
                continue;
            }

            digitProbabilities[mapNeuronToDigit[neuronIndex]] = neuronProbabilities[neuronIndex];
        }

        return digitProbabilities;
    }

    /**
     * Clones the last hidden layer and creates a new layer array with size layers.length + 1. This can be used to
     * iteratively add layers and avoid gradient fading problem.
     */
    public void expand () {
        Layer[] newLayers = new Layer[layers.length + 1];

        // Moves over the output layer.
        newLayers[newLayers.length - 1] = layers[layers.length - 1];
        // Clones last hidden layer and adds it before the output layer.
        newLayers[newLayers.length - 2] = layers[layers.length - 2].copy();

        // Moves over the rest of the layers.
        System.arraycopy(layers, 0, newLayers, 0, layers.length - 1);

        // Assigns this newly created array to the network.
        this.layers = newLayers;
    }

    /**
     * Folds the layers starting with input values and finishing with output layer's activations.
     *
     * @param digit Input digit with pixels
     * @return Activations for each output neuron
     */
    private double[] feedForward (double[] digit) {
        // Clones the input digit.
        double[] carry = Arrays.stream(digit).toArray();

        // Folds the layers array feeding forward the outputs from one layer to next as inputs.
        for (Layer layer : layers) {
            carry = layer.activation(carry);
        }

        return carry;
    }

}
