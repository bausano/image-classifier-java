package me.bausano.algorithms.nearestneighbour;

import me.bausano.Settings;
import me.bausano.algorithms.Classifier;

import java.util.Arrays;
import java.util.stream.DoubleStream;

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
     * @inheritDoc
     */
    public double[] estimate(double[] digit) {
        // For each digit class stores total distance to all of the neighbours. Defaults them with 1 to avoid zero
        // division (which would never probably happen anyway but it didn't feel right not to do that).
        double[] distances = new double[Settings.OUTPUT_CLASSES_COUNT];
        Arrays.fill(distances, 1);

        for (double[] neighbour : neighbours) {
            // Calculates how similar the digit is to iterated neighbour.
            distances[(int) neighbour[neighbour.length - 1]] += calculateDistance(digit, neighbour);
        }

        // Lowest distance total.
        double min = Arrays.stream(distances).min().getAsDouble();

        // Divides each distance total by the lowest total scaling them to range 0 - 1.
        return Arrays.stream(distances).map(x -> min / x).toArray();
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
