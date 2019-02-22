package me.bausano;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public class DataSet {

    /**
     * The loaded data from file that is used to train the network.
     */
    public final double[][] setForTraining;

    /**
     * The loaded data from file that is used to asses the network's performance.
     */
    public final double[][] setForValidation;

    /**
     * Instantiates data for training.
     *
     * @param trainingData Array of digits that are going to be used for training
     * @param validatingData Array of digits that are going to be used for validation
     */
    private DataSet(double[][] trainingData, double[][] validatingData) {
        this.setForTraining = trainingData;
        this.setForValidation = validatingData;
    }

    /**
     * DataSet builder. Factor splits data in ratio:
     * training data FACTOR : 1 validation data
     * For input value 0, all data are considered validating. For input value -1, all data are considered training.
     *
     * @param path Path to the data file
     * @param factor What part of the data is meant to be used for training and what for validating
     * @throws IOException Since reading the files is crucial, we want the program to panic on error in reading the
     *                      input file
     */
    public static DataSet from (Path path, int factor) throws IOException {
        double[][] data = Files.lines(path)
            .filter((String line) -> !line.equals(""))
            .map(String::trim)
            .map(DataSet::convertToDigit)
            .toArray(double[][]::new);

        // If factor is zero, initiates new validation data set. Useful for testing data inputs.
        if (factor == 0) {
            return new DataSet(
                    new double[0][Settings.INPUT_PARAMETERS_LENGTH],
                    data
            );
        }

        // If factor is -1, initiates new training data set. Useful for training data inputs.
        if (factor == -1) {
            return new DataSet(
                    data,
                    new double[0][Settings.INPUT_PARAMETERS_LENGTH]
            );
        }

        // All data with index lower than boundary are training data (exclusive), all above are validating data.
        int boundaryIndex = data.length / Settings.CROSSFOLD_FACTOR * (Settings.CROSSFOLD_FACTOR - 1) ;

        // Splits the data into two arrays.
        return new DataSet(
                Arrays.copyOfRange(data, 0, boundaryIndex),
                Arrays.copyOfRange(data, boundaryIndex, data.length)
        );
    }

    /**
     * Converts line to digit. A digit is represented by an array of 65 integers.
     *
     * @param line A single line from the data set file
     * @return Array of doubles representing pixels and the last integer represents the class
     */
    private static double[] convertToDigit(String line) {
        return Arrays.stream(line.split(","))
                .mapToDouble(Double::parseDouble)
                .toArray();
    }

}
