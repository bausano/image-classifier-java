package me.bausano;

import me.bausano.algorithms.estimator.Estimator;
import me.bausano.algorithms.nearestneighbour.NearestNeighbour;
import me.bausano.algorithms.neuralnetwork.NeuralNetwork;
import me.bausano.algorithms.neuralnetwork.Trainer;

import java.nio.file.Paths;

public class Main {

    /**
     * The program starts at this point. We can throw an exception straight to the standard output as it's easier to
     * debug and it's acceptable for this use case.
     *
     * @param args Console arguments
     * @throws Exception Exceptions are thrown into stdout
     */
    public static void main(String[] args) throws Exception {
        // Loads the training data input.
        DataSet inputData = DataSet.from(Paths.get(Settings.TRAINING_FILE_PATH), -1);
        System.out.println("Training data set ready.");

        // Loads the testing data input. We don't calibrate the network with this data, we use cross-fold validation
        // instead. I have changed this to the testing data set for the reviewers convenience, so that they don't have to
        // figure out how the DataSet API works.
        DataSet testingData = DataSet.from(Paths.get(Settings.TESTING_FILE_PATH), 0);
        System.out.println("Testing data set ready.");

        // Instantiates a neural network with random weights and and trains it.
        Thread runMlp = new Thread(() -> {
            NeuralNetwork mlp = NeuralNetwork.fromBlueprint(new int[] { Settings.INPUT_NEURONS, 37, 10 });
            new Trainer(mlp, inputData.setForTraining).train();
            Reporter.assess("Neural Network", mlp, testingData.setForValidation);
        });

        // Creates new nearest neighbour instance and runs it.
        Thread runNN = new Thread(() -> {
            NearestNeighbour knn = new NearestNeighbour(inputData.setForTraining);
            Reporter.assess("Nearest neighbour", knn, testingData.setForValidation);
        });

        // Creates and trains new instance of estimator which is combined mlp and knn.
        Thread runEstimator = new Thread(() -> {
            Estimator estimator = new Estimator(inputData.setForTraining);
            estimator.train();
            Reporter.assess("Estimator", estimator, testingData.setForValidation);
        });

        // Comment out any of following lines to prevent algorithm from running (advised on slow machines).
        runMlp.start();
        runNN.start();
        runEstimator.start();
    }
}
