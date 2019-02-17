package me.bausano.algorithms.neuralnetwork;

import me.bausano.algorithms.Classifier;

import java.util.Arrays;
import java.util.PrimitiveIterator;
import java.util.Random;

public class NeuralNetwork implements Classifier {

    /**
     * Array of network layers. The input layer is not included and is only abstract. Therefore a network that has
     * structure [64, 36, 10] will have two layers behind the scenes.
     */
    private Layer[] layers;

    /**
     * Used to map neurons to classes. Indices are target classes (digits 0-9) and values are associated neurons.
     */
    private final int[] classMapping;

    /**
     * @param layers Array of network layers
     * @param classMapping Converts neurons to classes
     */
    private NeuralNetwork(Layer[] layers, int[] classMapping) {
        this.layers = layers;
        this.classMapping = classMapping;
    }

    /**
     * Generates new neural network with randomly assigned weights from given schema. Schema is a list where it's length
     * represents number of layers and each element number of neurons within the layer.
     *
     * @param schema Layers and neurons including input and output layer
     * @param classMapping Holds information about which output neuron represents which class
     * @return New instance of an untrained network
     */
    public static NeuralNetwork fromBlueprint(int[] schema, int[] classMapping) {
        PrimitiveIterator.OfDouble rng = new Random().doubles().iterator();

        Layer[] layers = new Layer[schema.length - 1];

        // Using loops instead of streams for better performance.
        for (int layerIndex = 1; layerIndex < schema.length; layerIndex++) {
            // We create a new layer array (collection of neurons). It's going to have number of neurons according to
            // the schema and each neuron will have weight according to number of neurons in previous layer plus a bias.
            double[][] weights = new double[schema[layerIndex]][schema[layerIndex - 1]];
            double[] biasses = new double[schema[layerIndex]];

            // Generate each layer neuron.
            for (int neuronIndex = 0; neuronIndex < schema[layerIndex]; neuronIndex++) {
                // Generate each layer weight based on number of neurons in previous layer.
                for (int weightIndex = 0; weightIndex < schema[layerIndex - 1]; weightIndex++) {
                    // Weight is a double in range <-1;1).
                    weights[neuronIndex][weightIndex] = rng.next() * 2d - 1d;
                }

                // Sets the bias to be 0. It does not matter how we initially set bias as the activation is always 1.
                biasses[neuronIndex] = 0d;
            }

            // Since we are not creating the input layer, we have to decrement layer index by one when assigning it to
            // the layers array.
            layers[layerIndex - 1] = new Layer(weights, biasses);
        }

        return new NeuralNetwork(layers, classMapping);
    }

    /**
     * Overloading the init method to default the class mapping.
     *
     * @param schema Layers and neurons including input and output layer
     * @return New instance of an untrained network
     */
    public static NeuralNetwork fromBlueprint(int[] schema) {
        return NeuralNetwork.fromBlueprint(schema, new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    }

    /**
     * @inheritDoc
     */
    public int classify (int[] digit) {
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

        return candidate;
    }

    /**
     * Folds the layers starting with input values and finishing with output layer's activations.
     *
     * @param digit Input digit with pixels
     * @return Activations for each output neuron
     */
    private double[] feedForward (int[] digit) {
        // Clones and converts input digit integers into doubles.
        double[] carry = Arrays.stream(digit).asDoubleStream().toArray();

        // Folds the layers array feeding forward the outputs from one layer to next as inputs.
        for (Layer layer : layers) {
            carry = layer.activation(carry);
        }

        return carry;
    }

}
