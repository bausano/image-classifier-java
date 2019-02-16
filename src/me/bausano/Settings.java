package me.bausano;

public class Settings {

    /**
     * Defines over how many input parameters are we working on. This is useful for constructing arrays of static length
     * which brings performance benefits.
     */
    public static final int INPUT_PARAMETERS_LENGTH = 64;

    /**
     * What portion of training data should be used for cross fold validation. The program will omit these data when it
     * trains. Then it uses that data to calculate how successful the program is.
     */
    public static final int CROSSFOLD_FACTOR = 10;

    /**
     * File path to the training file relative to the project root.
     */
    public static final String TRAINING_FILE_PATH = "data/training-data.txt";

    /**
     * File path to the testing file relative to the project root. This data should not be used to parametrize the
     * algorithms, to train it nor to cross validate it.
     */
    public static final String TESTING_FILE_PATH = "data/testing-data.txt";

}
