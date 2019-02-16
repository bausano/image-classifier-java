package me.bausano;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public class DataSet {

    /**
     * The loaded data from file that is used to train the network.
     */
    public final int[][] setForTraining;

    /**
     * The loaded data from file that is used to asses the network's performance.
     */
    public final int[][] setForValidation;

    /**
     * Instantiates data for training.
     *
     * @param trainingData Array of digits that are going to be used for training
     * @param validatingData Array of digits that are going to be used for validation
     */
    private DataSet(int[][] trainingData, int[][] validatingData) {
        this.setForTraining = trainingData;
        this.setForValidation = validatingData;
    }

    /**
     * Instantiates data set for testing;
     *
     * @param validatingData Array of digits that are going to be used for assessing the performance
     */
    private DataSet(int[][] validatingData) {
        this.setForTraining = new int[0][Settings.INPUT_PARAMETERS_LENGTH];
        this.setForValidation = validatingData;
    }

    /**
     * DataSet builder.
     *
     * @param path Path to the data file
     * @throws IOException Since reading the files is crucial, we want the program to panic on error in reading the
     *                      input file
     */
    public static DataSet from (Path path, int factor) throws IOException {
        int[][] data = Files.lines(path)
            .filter((String line) -> !line.equals(""))
            .map(String::trim)
            .map(DataSet::convertToDigit)
            .toArray(int[][]::new);

        if (factor == 0) {
            return new DataSet(data);
        }

        // All data with index lower than boundary are training data (exclusive), all above are validating data.
        int boundaryIndex = (data.length / Settings.CROSSFOLD_FACTOR * (Settings.CROSSFOLD_FACTOR - 1)) ;

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
     * @return Array of integers representing pixels and the last integer represents the class
     */
    private static int[] convertToDigit(String line) {
        return Arrays.stream(line.split(","))
                .mapToInt(Integer::parseInt)
                .toArray();
    }

}
