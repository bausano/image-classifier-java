package me.bausano.algorithms.neuralnetwork;

import me.bausano.Settings;

public class Layer {

    /**
     * Array of neurons with weights and bias. Each weight is semantically connected to one neuron in previous layer.
     * Neuron would be preferably expressed as tuple of weights and one bias, however Java does not support tuples
     * neither zero cost abstractions, hence we have separate bias array.
     */
    private double[][] neurons;

    /**
     * Each neuron has one bias weight. This bias is always included in computing the net and output of a neuron.
     */
    private double[] biasses;

    /**
     * @param neurons Neurons with their associated weights
     * @param biasses Bias associated with each neuron
     */
    public Layer (double[][] neurons, double[] biasses) {
        this.biasses = biasses;
        this.neurons = neurons;
    }

    /**
     * Computes the activation vector for the layer.
     *
     * @param inputs Outputs from the previous layer
     * @return Sets of inputs for next layer
     */
    public double[] activation (double[] inputs) {
        double[] outputs = new double[neurons.length];

        // Calculate output for each neuron by multiplying its weights by outputs from last layer.
        for (int neuronIndex = 0; neuronIndex < neurons.length; neuronIndex++) {
            double product = biasses[neuronIndex];

            for (int weightIndex = 0; weightIndex < neurons[neuronIndex].length; weightIndex++) {
                product += neurons[neuronIndex][weightIndex] * inputs[weightIndex];
            }

            outputs[neuronIndex] = Settings.activation.function.apply(product);
        }

        return outputs;
    }

}
