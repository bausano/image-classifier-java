package me.bausano.algorithms;

public interface Classifier {

    /**
     * Classifies given digit based on previously seen data.
     *
     * @param digit Digit we want to find match for
     * @return Class of the neighbour that resembled the digit the most
     */
    int classify (double[] digit);

}
