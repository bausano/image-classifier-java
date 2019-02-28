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
                    new double[0][Settings.INPUT_PARAMETERS],
                    data
            );
        }

        // If factor is -1, initiates new training data set. Useful for training data inputs.
        if (factor == -1) {
            return new DataSet(
                    data,
                    new double[0][Settings.INPUT_PARAMETERS]
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
        return mapDigitThroughFilters(
                Arrays.stream(line.split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray(),
                Settings.FILTERS
        );
    }

    /**
     * Maps digit through given filter. This is usually a 3x3 matrix of weights that highlight certain feature in the
     * image, such as edges.
     *
     * @param digit Original digit
     * @param filters Matrix of weights
     * @return Changed digit
     */
    private static double[] mapDigitThroughFilters(double[] digit, double[][][] filters) {
        // New image will be composed out of the main image and its mutations under each filter plus the target.
        double[] output = new double[Settings.INPUT_PARAMETERS * (filters.length + 1) + 1];
        // Sets the last value as target (classification 0 - 9).
        output[output.length - 1] = digit[digit.length - 1];

        // We copy all values from the original input into the new one.
        System.arraycopy(digit, 0, output, 0, digit.length);

        for (int filterIndex = 0; filterIndex < filters.length; filterIndex++) {
            // Filters each pixel through a matrix (usually 3x3) with weights. The matrix is set to benefit certain shapes
            // such as corners.
            for (int pixel = 0; pixel < Settings.INPUT_PARAMETERS; pixel++) {
                double mappedValue = mapFilterToPixel(digit, pixel, filters[filterIndex]);
                output[pixel + Settings.INPUT_PARAMETERS * (filterIndex + 1)] = mappedValue;
            }
        }

        return output;
    }

    /**
     * Maps given pixel and its neighbours over the filter.
     *
     * @param digit Original input image
     * @param pixel Pixel index to map
     * @param filter Filter matrix to apply
     * @return Value for given pixel in given point
     */
    private static double mapFilterToPixel(double[] digit, int pixel, double[][] filter) {
        // How many pixels are on one row of the digit.
        int rowLength = (int) Math.sqrt(Settings.INPUT_PARAMETERS);

        // New pixel value.
        double checksum = 0d;
        // Multiplies values of each neighbour pixel with one weight from the filter.
        for (int rowShift = 0; rowShift < filter.length; rowShift++) {
            for (int filterColumn = 0; filterColumn < filter[rowShift].length; filterColumn++) {
                int targetPixel = rowLength * rowShift + pixel + filterColumn;

                // Add check for the pixels that are on the edges (both vertically and horizontally). Prevents
                // index out of bounds error.
                if (
                        targetPixel >= (Math.floor((double) targetPixel / rowLength) + 1) * rowLength ||
                        targetPixel > Settings.INPUT_PARAMETERS
                ) {
                    break;
                }

                checksum += digit[targetPixel] * filter[rowShift][filterColumn];
            }
        }

        return checksum;
    }

}
