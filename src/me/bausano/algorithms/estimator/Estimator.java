package me.bausano.algorithms.estimator;

import me.bausano.Settings;
import me.bausano.algorithms.Classifier;
import me.bausano.algorithms.nearestneighbour.NearestNeighbour;
import me.bausano.algorithms.neuralnetwork.NeuralNetwork;
import me.bausano.algorithms.neuralnetwork.Trainer;

public class Estimator implements Classifier {

    /**
     * Data set of structured input data to train on.
     */
    private final double[][] data;

    /**
     * The numbers are split into groups. These groups are based on my observations from confusion matrix.
     */
    private NeuralNetwork[] groups = new NeuralNetwork[] {
            // Train network only for numbers 2, 4, 5, 7 and 8.
            NeuralNetwork.fromBlueprint(
                    new int[]{ 191, 42, 6 },
                    new int[]{ 2, 4, 5, 7, 8, -1 },
                    new int[]{ 5, 5, 0, 5, 1, 2, 5, 3, 4, 5 }
            ),
            // Network only for numbers 0, 1, 3, 6 and 9.
            NeuralNetwork.fromBlueprint(
                    new int[] { 191, 42, 6 },
                    new int[] { 0, 1, 3, 6, 9, -1 },
                    new int[] { 0, 1, 5, 2, 5, 5, 3, 5, 5, 4 }
            )
    };

    /**
     * Nearest neighbour instance.
     */
    private final NearestNeighbour nn;

    /**
     * @param data Input data set where last int is the class
     */
    public Estimator(double[][] data) {
        this.data = data;
        this.nn = new NearestNeighbour(data);
    }

    /**
     * Trains the algorithm.
     */
    public void train() {
        for (NeuralNetwork group : groups) {
            new Trainer(group, data).train();
        }
    }

    /**
     * @inheritDoc
     */
    public int classify (double[] digit) {
        double[] estimates = this.estimate(digit);

        double maxEstimate = 0;
        int maxEstimateClass = 0;

        // Finds the maximum estimate class.
        for (int classIndex = 0; classIndex < estimates.length; classIndex++) {
            if (estimates[classIndex] < maxEstimate) {
                continue;
            }

            // Updates the best result.
            maxEstimate = estimates[classIndex];
            maxEstimateClass = classIndex;
        }

        return maxEstimateClass;
    }

    /**
     * @inheritDoc
     */
    public double[] estimate(double[] digit) {
        double[] estimates = nn.estimate(digit);

        for (NeuralNetwork group : groups) {
            addPartialEstimates(estimates, group.estimate(digit));
        }

        return estimates;
    }

    /**
     * Adds estimates array into base array.
     *
     * @param base Array that is going to be summed into
     * @param estimates Array to sum from
     */
    private void addPartialEstimates(double[] base, double[] estimates) {
        for (int classIndex = 0; classIndex < Settings.OUTPUT_CLASSES_COUNT; classIndex++) {
            base[classIndex] += estimates[classIndex];
        }
    }

}
