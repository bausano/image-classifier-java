package me.bausano.algorithms.nearestneighbour;

public class Neighbour implements Comparable<Neighbour> {

    /**
     * Distance to currently compared number.
     */
    public double distance;

    /**
     * Which digit this neighbour is.
     */
    public int classification;

    /**
     * @param classification Digit
     * @param distance Distance to currently compared
     */
    Neighbour (int classification, double distance) {
        this.distance = distance;
        this.classification = classification;
    }

    /**
     * Compares this neighbour distance to another one.
     *
     * @param another Another neighbour to compare
     * @return Whether this distance is smaller
     */
    @Override
    public int compareTo(Neighbour another) {
        return Double.compare(another.distance, this.distance);
    }
}
