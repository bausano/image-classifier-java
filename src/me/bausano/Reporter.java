package me.bausano;

import me.bausano.algorithms.Classifier;
import me.bausano.algorithms.nearestneighbour.NearestNeighbour;

public class Reporter {

    /**
     * Standard report on how successful the classifier is in digit identification.
     *
     * @param title Name of the report
     * @param classifier Algorithm reference
     * @param data Validation data
     */
    public static void assess (String title, Classifier classifier, double[][] data) {
        int correctlyClassified = 0;
        for (double[] digit : data) {

            if (classifier.classify(digit) == digit[digit.length - 1]) {
                correctlyClassified++;
            } else {
                System.out.printf("\nCorrect: %d", (int) digit[digit.length - 1]);
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

    /**
     * Calculates the confusion matrix for given data. This matrix represents how many times has been each class
     * classified as each other class. This gives us insight on how much different classes resemble each other.
     *
     * @param classifiers Algorithm that classifies digits
     * @param data Data that is preferably not included in the neighbours
     */
    public static void printConfusionMatrix (Classifier[] classifiers, double[][] data) {
        // Matrix where rows are classes and columns their classifications.
        int[][] matrix = new int[Settings.OUTPUT_CLASSES_COUNT][Settings.OUTPUT_CLASSES_COUNT];

        for (double[] digit : data) {
            for (Classifier classifier : classifiers) {
                int classification = classifier.classify(digit);
                int target = (int) digit[digit.length - 1];

                // We don't care about correctly classified digits.
                if (classification == target) {
                    matrix[target][classification] = -1;
                    continue;
                }

                // Rows represent target correct classes and columns represent how many times that class has been classified
                // as certain class. If first row was [10, 0, 1, ...], that would mean that class 0 was classified as 0 ten
                // times (a.k.a. correctly), as 1 zero times, as 2 one time...
                // Then we add up all [x][y] and [y][x], because it's more important to know how many times has zero been
                // identified as 8 plus 8 as 0, rather than having two separate counters.
                matrix[target][classification]++;
                matrix[classification][target]++;

            }
        }

        System.out.println("--- Confusion matrix ----------");

        for (int[] target : matrix) {
            System.out.printf("| ");

            for (int classification : target) {
                System.out.printf("%d, ", classification);
            }

            System.out.print("|\n");
        }

        System.out.println("------------------------------");
    }

}
