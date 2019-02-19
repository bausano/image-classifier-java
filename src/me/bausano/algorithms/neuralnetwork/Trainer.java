package me.bausano.algorithms.neuralnetwork;

import me.bausano.Settings;

public class Trainer {

    /**
     * Sets number of iteration so that the network always finishes the training when the learning rate is lowest.
     */
    private final int iterations = Settings.CYCLES * (Settings.STEP_SIZE * 2) + Settings.STEP_SIZE + 1;

    /**
     * Maximum learning rate equals to mean learning rate with an upper bound of oscillation.
     */
    private final double maxLR = Settings.MEAN_LEARNING_RATE + Settings.OSCILLATION;

    /**
     * Maximum learning rate equals to mean learning rate with a lower bound of oscillation.
     */
    private final double minLR = Settings.MEAN_LEARNING_RATE - Settings.OSCILLATION;

    /**
     * Used to map neurons to classes. Indices are target classes (digits 0-9) and values are associated neurons.
     */
    private final int[] classMapping;

    /**
     * Training data.
     */
    private final double[][] data;

    /**
     * Network to train.
     */
    private final NeuralNetwork network;

    /**
     * Learning rate is going to be updated each iteration.
     */
    private double LR;

    /**
     * @param network Neural network to train
     * @param data Training data
     * @param classMapping Maps target classes to neurons
     */
    public Trainer(NeuralNetwork network, double[][] data, int[] classMapping) {
        this.data = data;
        this.network = network;
        this.classMapping = classMapping;
    }

    /**
     * @param network Neural network to train
     * @param data Training data
     */
    public Trainer(NeuralNetwork network, double[][] data) {
        this.data = data;
        this.network = network;
        this.classMapping = new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    }

    /**
     * Trains the network.
     */
    public void train() {
        for (int iteration = 0; iteration < iterations; iteration++) {
            // Changes the learning rate with each iteration. Is it scaled down and cycled.
            this.LR = calculateLearningRate(iteration);

            for (int sampleIndex = 0; sampleIndex < data.length; sampleIndex++) {
                // Calculates the nudges for given sample and saves them to a temporary vector.
                learnSample(data[sampleIndex]);

                // Updates the weights of all layers every nth sample.
                if (sampleIndex % Settings.BATCH_SIZE == 0) {
                    commitNudges();
                }
            }

            // After each iteration, it updates the weight by the leftover nudges.
            commitNudges();
        }
    }

    /**
     * Updates the learning rate based on epoch. Learning rate cycles around a descending value.
     *
     * @param epoch Which iteration of learning is it
     * @return New value that the learning rate should take
     */
    private double calculateLearningRate(int epoch) {
        // The step including floating point based on step size.
        double step = 1d + (double) epoch / (2d * Settings.STEP_SIZE);
        // Precise cycle number integer.
        double cycle = Math.floor(step);
        // How much is missing to next 50 % of a cycle.
        double progress = Math.abs(0.5d - (step - cycle));

        // Learning rate change based on current cycle with oscillation.
        double newLearningRate = Math.abs(maxLR - (2d * (maxLR - minLR) * (0.5d - progress)));

        // Makes the learning rate descend with each epoch.
        return newLearningRate * (iterations - epoch + 1) / iterations;
    }

    /**
     * Feeds forward the sample, calculates the error and saves nudges that are to be committed to the network.
     *
     * @param sample Digit
     */
    private void learnSample(double[] sample) {
        // Converts digit class to expected neuron.
        int target = classMapping[(int) sample[sample.length - 1]];

        // Matrix with each neuron's activation. If we want to implement other activation functions, this would have to
        // include net (pre squashed by activation function) as well as out values.
        double[][] activationsMatrix = calculateActivations(sample);

        // Calculates the error of the output layer. This does not include learning rate or previous neuron activations.
        // We will use this variable to fold the layers and propagate the error backwards.
        double[] partialError = calculateOutputLayerError(target, activationsMatrix[activationsMatrix.length - 1]);

        // Folding the layers array starting from the last layer.
        for (int layerIndex = network.layers.length - 1; layerIndex >= 0; layerIndex--) {
            // Has a side effect of updating local nudges cache and returns errors of each neurons from layer which is
            // used in layer n - 1.
            partialError = updateWeightsAndReturnError(layerIndex, partialError, activationsMatrix);
        }
    }

    private double[][] calculateActivations(int[] sample) {
        double[][] activationsMatrix = new double[network.layers.length][];
        activationsMatrix[0] = network.layers[0].activation(sample);

        for (int layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {

        }
    }

}
