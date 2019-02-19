package me.bausano.algorithms.nearestneighbour;

import me.bausano.Settings;
import me.bausano.algorithms.Classifier;

public class NearestNeighbour implements Classifier {

    /**
     * Data set of structured input data to match against.
     */
    private final double[][] neighbours;

    /**
     * @param neighbours Input data set where last int is the class
     */
    public NearestNeighbour(double[][] neighbours) {
        this.neighbours = neighbours;
    }

    /**
     * @inheritDoc
     */
    public int classify (double[] digit) {
        // How close was the closest neighbour to the digit.
        double closestDistance = Double.MAX_VALUE;
        // Which class had the closest distance.
        int closestClass = 0;

        for (double[] neighbour : neighbours) {
            // Calculates how similar the digit is to iterated neighbour.
            double distance = calculateDistance(digit, neighbour);

            // If there was a neighbour with lower distance, go to next iteration.
            if (distance > closestDistance) {
                continue;
            }

            // The last element of array is the classification.
            closestClass = (int) neighbour[neighbour.length - 1];
            closestDistance = distance;
        }

        return closestClass;
    }

    /**
     * Calculates distance between two vectors. To find the Euclidean distance, the result needs to be square rooted.
     * This is however not necessary to do for this algorithm, therefore we can avoid the computation.
     *
     * @param from Point which has at least Settings.INPUT_PARAMETERS_LENGTH length
     * @param to Point which has at least Settings.INPUT_PARAMETERS_LENGTH length
     * @return Distance between the two multi dimensional points
     */
    private double calculateDistance (double[] from, double[] to) {
        int sum = 0;

        // We assume both arrays will have adequate number of elements. These assumptions might possibly result in
        // better overall performance.
        for (int pixel = 0; pixel < Settings.INPUT_PARAMETERS_LENGTH; pixel++) {
            double difference = from[pixel] - to[pixel];

            sum += difference * difference;
        }

        return sum;
    }

}
