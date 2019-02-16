package me.bausano;

import me.bausano.algorithms.Classifier;

public class Reporter {

    /**
     * Standard report on how successful the classifier is in digit identification.
     *
     * @param title Name of the report
     * @param classifier Algorithm reference
     * @param data Validation data
     */
    public static void assess (String title, Classifier classifier, int[][] data) {
        int correctlyClassified = 0;
        for (int[] digit : data) {
            if (classifier.classify(digit) == digit[digit.length - 1]) {
                correctlyClassified++;
            }
        }

        System.out.printf(
                "\n> %s\nCorrectly classified %d out of %d (%.2f %%).",
                title.toUpperCase(),
                correctlyClassified,
                data.length,
                (float) correctlyClassified / (float) data.length * 100f
        );
    }

}
