package me.bausano;

import me.bausano.algorithms.neuralnetwork.ActivationMapper;

public class Settings {

    /**
     * K-nearest neighbours parameter k, which specifies how many closest neighbours get to vote on the classification.
     */
    public static final int K_NEAREST_NEIGHBOURS = 1;

    /**
     * Defines over how many input parameters are we working on. This is useful for constructing arrays of static length
     * which brings performance benefits.
     */
    public static final int INPUT_PARAMETERS_LENGTH = 64;

    /**
     * How many output classes are there. For digit classification, there's 10 (0-9).
     */
    public static final int OUTPUT_CLASSES_COUNT = 10;

    /**
     * What portion of training data should be used for cross fold validation. The program will omit these data when it
     * trains. Then it uses that data to calculate how successful the program is.
     */
    public static final int CROSSFOLD_FACTOR = 2;

    /**
     * After how many data samples should the neural network update its weights.
     */
    public static final int BATCH_SIZE = 10;

    /**
     * Hyper-parameter for cycling learning rate that indicates how many batches does one half of the cycle last.
     */
    public static final int STEP_SIZE = 16;

    /**
     * Used to calculate number of iterations for training. The formula is
     * iterations = CYCLES * (STEP_SIZE * 2) + STEP_SIZE + 1
     * This ensures that the network will stop training the when learning rate is the lowest, which gives the best
     * accuracy.
     */
    public static final int CYCLES = 20;

    /**
     * Initial learning rate that will decrease with each epoch and also is cycled around.
     */
    public static final double MEAN_LEARNING_RATE = 3d;

    /**
     * How much should the learning rate be changed throughout the cycling.
     * For example, if the learning rate is 3 and we oscillate by 0.5, learning rate will change from 3.5 to 2.5 during
     * first step, to 3.5 during second step.
     */
    public static final double OSCILLATION = 2.5d;

    /**
     * File path to the training file relative to the project root.
     */
    public static final String TRAINING_FILE_PATH = "data/training-data.txt";

    /**
     * File path to the testing file relative to the project root. This data should not be used to parametrize the
     * algorithms, to train it nor to cross validate it.
     */
    public static final String TESTING_FILE_PATH = "data/testing-data.txt";

    /**
     * Activation function is used throughout all layers to indulge linearity.
     */
    public static final ActivationMapper activation = ActivationMapper.sigmoid();

    /**
     * Filters used to map the input digit. Each pixel is mapped over each over the filters, therefore the resulting
     * number of inputs the new digit will have is INPUTS + FILTERS.LENGTH * INPUTS.
     */
    public static final double[][][] FILTERS = new double[][][] {
        // Detects horizontal edges.
        new double[][] {
                new double[] { 1, 1, 1, },
                new double[] { 0, 0, 0, },
                new double[] { -1, -1, -1, },
        },
        // Detects vertical edges.
        new double[][] {
                new double[] { 1, 0, -1, },
                new double[] { 1, 0, -1, },
                new double[] { 1, 0, -1, },
        },
    };

}
