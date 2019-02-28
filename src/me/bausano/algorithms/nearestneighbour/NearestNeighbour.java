package me.bausano.algorithms.nearestneighbour;

import me.bausano.Settings;
import me.bausano.algorithms.Classifier;

import java.util.PriorityQueue;

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
        double bestEstimate = Double.MIN_VALUE;
        // Which class had the closest distance.
        int closestClass = 0;

        // Finds class with the highest estimate.
        double[] estimates = estimate(digit);
        for (int classIndex = 0; classIndex < estimates.length; classIndex++) {
            if (estimates[classIndex] < bestEstimate) {
                continue;
            }

            // Updates the leading estimate.
            closestClass = classIndex;
            bestEstimate = estimates[classIndex];
        }

        return closestClass;
    }

    /**
     * @inheritDoc
     */
    public double[] estimate(double[] digit) {
        PriorityQueue<Neighbour> closestNeighbours = new PriorityQueue<>(Settings.K_NEAREST_NEIGHBOURS);

        // Fills the queue with distances.
        for (double[] neighbour : neighbours) {
            int target = (int) neighbour[neighbour.length - 1];
            double distance = calculateDistance(digit, neighbour);

            closestNeighbours.add(new Neighbour(target, distance));

            // Ensures there's not more than k neighbours in the fit queue.
            if (closestNeighbours.size() > Settings.K_NEAREST_NEIGHBOURS) closestNeighbours.poll();
        }

        double[] classes = new double[Settings.OUTPUT_CLASSES_COUNT];
        for (int classIndex = 0; classIndex < classes.length; classIndex++) {
            // Storage to close compilers mouth.
            int target = classIndex;
            // Counts occurrence of a class and divides it by total queue size.
            classes[classIndex] = (double) closestNeighbours.stream()
                    .filter((Neighbour x) -> x.classification == target)
                    .count() / (double) Settings.K_NEAREST_NEIGHBOURS;
        }

        return classes;
    }

    /**
     * Calculates distance between two vectors. To find the Euclidean distance, the result needs to be square rooted.
     * This is however not necessary to do for this algorithm, therefore we can avoid the computation.
     *
     * @param from Point which has at least Settings.INPUT_PARAMETERS length
     * @param to Point which has at least Settings.INPUT_PARAMETERS length
     * @return Distance between the two multi dimensional points
     */
    private double calculateDistance (double[] from, double[] to) {
        int sum = 0;

        // We assume both arrays will have adequate number of elements. These assumptions might possibly result in
        // better overall performance.
        for (int pixel = 0; pixel < from.length - 1; pixel++) {
            double difference = from[pixel] - to[pixel];

            sum += difference * difference;
        }

        return sum;
    }

}
