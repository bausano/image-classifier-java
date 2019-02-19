package me.bausano.algorithms.neuralnetwork;

import me.bausano.Settings;

public class Trainer {

    /**
     * Sets number of iteration so that the network always finishes the training when the learning rate is lowest.
     */
    private final int iterations = Settings.CYCLES * (Settings.STEP_SIZE * 3) + 1;

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
     * Caches the updates to weights.
     */
    private double[][][] weightNudges;

    /**
     * Caches the updates to biases.
     */
    private double[][] biasNudges;

    /**
     * We have to take an average from all nudges updates, therefore we need a counter.
     */
    private int nudgesSinceLastCommit;

    /**
     * @param network Neural network to train
     * @param data Training data
     */
    public Trainer(NeuralNetwork network, double[][] data) {
        this.data = data;
        this.network = network;
        clearCache();
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
     * Resets nudges.
     */
    private void clearCache() {
        weightNudges = new double[network.layers.length][][];
        biasNudges = new double[network.layers.length][];
        nudgesSinceLastCommit = 0;
    }

    /**
     * Updates the learning rate based on epoch. Learning rate cycles around a descending value.
     *
     * @param epoch Which iteration of learning is it
     * @return New value that the learning rate should take
     */
    private double calculateLearningRate(int epoch) {
        // Maximum learning rate equals to mean learning rate with an upper bound of oscillation.
        double maxLR = Settings.MEAN_LEARNING_RATE + Settings.OSCILLATION;
        // Maximum learning rate equals to mean learning rate with a lower bound of oscillation.
        double minLR = Settings.MEAN_LEARNING_RATE - Settings.OSCILLATION;

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
        int target = network.mapDigitToNeuron[(int) sample[sample.length - 1]];

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
            partialError = addNudgesAndReturnErrors(layerIndex, partialError, activationsMatrix);
        }
    }

    /**
     * Calculates and caches the activations values for each neuron of each layer. Works just like the classify method
     * on network with the exception that here we actually save the outputs of each layer.
     *
     * @param sample Input digit
     * @return Activations for each layer
     */
    private double[][] calculateActivations(double[] sample) {
        double[][] activationsMatrix = new double[network.layers.length + 1][];
        // Activation matrix includes inputs, so all layer indices are shifted to n + 1. Wish there were well
        // performable streams in Java as all of these computations are made to be done in a functional way.
        // Unfortunately streams has about 5 times worse performance in Java, which along with the fact that it has non
        // zero cost abstractions brings me to conclusion that it's not a good language to be doing machine learning in.
        activationsMatrix[0] = sample.clone();

        // Folding the layer array by inputting outputs from previous layers into the next one.
        for (int layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
            activationsMatrix[layerIndex + 1] = network.layers[layerIndex].activation(activationsMatrix[layerIndex]);
        }

        return activationsMatrix;
    }

    /**
     * Calculates the errors for each neuron in the output layer.
     *
     * @param target What is the desired class of the digit
     * @param activations The activations of each output layer neuron
     * @return Vector of deltas for each neuron that is to be mapped over activation from previous layer and LR
     */
    private double[] calculateOutputLayerError(int target, double[] activations) {
        double[] deltas = new double[activations.length];

        for (int neuronIndex = 0; neuronIndex < activations.length; neuronIndex++) {
            // Formula -(target - output) that emerges from the chain rule.
            double totalToOutputError = -((target == neuronIndex ? 1d : 0d) - activations[neuronIndex]);
            // The derivative of activation function computed from the value of the activation function over the net.
            // Functions with steeper derivatives converge faster.
            double derivative = Settings.activation.derivative.apply(activations[neuronIndex]);
            // We cache the value.
            deltas[neuronIndex] = totalToOutputError * derivative;
        }

        return deltas;
    }

    /**
     * For each layer it computes the error that is sent to previous layer, and nudges each neuron's weights and bias
     * in direction to reduce the error. The functionality differs a little for output layer where we don't recompute
     * the error sent to previous layer.
     *
     * @param layerIndex Layer to perform the updates for
     * @param previousErrors Errors from previous layer that are used to follow the chain rule
     * @param activationMatrix Activations
     * @return Layer contribution to total error
     */
    private double[] addNudgesAndReturnErrors(int layerIndex, double[] previousErrors, double[][] activationMatrix) {
        Layer layer = network.layers[layerIndex];

        double[] currentErrors = new double[layer.neurons.length];

        for (int neuronIndex = 0; neuronIndex < layer.neurons.length; neuronIndex++) {
            double currentError;

            // For the output layer, we have already computed the errors. We don't need to consider any weights for this
            // layer as there are not any connecting it to the output, there's just a single activation number.
            if (layerIndex == network.layers.length - 1) {
                currentError = previousErrors[neuronIndex];
            } else {
                // Derivative of activation output for current neuron. Note that in activation matrix, layer indices are
                // shifted by one.
                double derivative = Settings.activation.derivative.apply(activationMatrix[layerIndex + 1][neuronIndex]);

                // Calculates the neurons participation on the total error of next layer.
                double totalError = 0d;
                for (int errorIndex = 0; errorIndex < previousErrors.length; errorIndex++) {
                    // Probably the hardest part to get your head around. We want to calculate overall neuron error.
                    // Each neuron is connected to each neuron in the next layer, and errorIndex will serve us as an
                    // iterator as it's coming from previous layer, therefore has same length as neurons in that layer.
                    // And last but not least, we take the weight from that neuron that connects that neuron in the next
                    // layer to the currently iterated over in this layer. It's funny how much neater this looks with
                    // functional programming style of folding the arrays.
                    totalError += network.layers[layerIndex + 1].neurons[errorIndex][neuronIndex] * previousErrors[errorIndex];
                }

                currentError = derivative * totalError;
            }

            currentErrors[neuronIndex] = currentError;

            // Calculates the nudge for each neuron weight by following the chain rule and scaling it with learning rate.
            double[] nudges = new double[activationMatrix[layerIndex].length];
            for (int activation = 0; activation < nudges.length; activation++) {
                // Use the activations from previous layer.
                nudges[activation] = activationMatrix[layerIndex][activation] * currentError * LR;
            }

            // Caches nudges to local vector before committing them to the layer.
            addNudgesForNeuron(layerIndex, neuronIndex, nudges, currentError * LR);
        }

        return currentErrors;
    }

    /**
     * Caches the nudges that are later on committed in bulk to the layer's neurons.
     *
     * @param layer Layer index
     * @param neuron Neuron index
     * @param nudges Weight changes
     * @param biasNudge Bias change
     */
    private void addNudgesForNeuron(int layer, int neuron, double[] nudges, double biasNudge) {
        // If the array are not initialized, prepare them.
        if (weightNudges[layer] == null) {
            double[][] neurons = network.layers[layer].neurons;
            weightNudges[layer] = new double[neurons.length][neurons[neuron].length];
            biasNudges[layer] = new double[neurons.length];
        }

        biasNudges[layer][neuron] = biasNudge;

        // Adds all weight nudges to the temporary vector.
        for (int weightIndex = 0; weightIndex < weightNudges[layer][neuron].length; weightIndex++) {
            weightNudges[layer][neuron][weightIndex] += nudges[weightIndex];
        }

        nudgesSinceLastCommit++;
    }

    /**
     * Commits all cached nudges to the layers weights and biases.
     */
    private void commitNudges() {
        // Avoid division by zero.
        if (nudgesSinceLastCommit == 0) {
            return;
        }

        // For each layer, each layer's neuron and each neuron's weight, perform an update.
        for (int layerIndex = 0; layerIndex < network.layers.length; layerIndex++) {
            Layer layer = network.layers[layerIndex];

            // Updates bias. Since trainer is trying to achieve minimum possible error (we are minimizing the function),
            // we have to deduct the nudges from the current bias and weights.
            for (int neuronIndex = 0; neuronIndex < layer.neurons.length; neuronIndex++) {
                layer.biases[neuronIndex] -= biasNudges[layerIndex][neuronIndex] / nudgesSinceLastCommit;

                // Updating the weights.
                for (int weightIndex = 0; weightIndex < layer.neurons[neuronIndex].length; weightIndex++) {
                    layer.neurons[neuronIndex][weightIndex] -= weightNudges[layerIndex][neuronIndex][weightIndex] / nudgesSinceLastCommit;
                }
            }
        }

        // Clears temporary vector.
        clearCache();
    }

}
