package me.bausano.algorithms;

public interface Classifier {

    /**
     * Classifies given digit based on previously seen data.
     *
     * @param digit Digit we want to find match for
     * @return Class of the neighbour that resembled the digit the most
     */
    int classify (double[] digit);

    /**
     * Gives probabilities for each class.
     *
     * @param digit Input digit
     * @return Vector of probabilities in range 0 - 1
     */
    double[] estimate (double[] digit);

}
