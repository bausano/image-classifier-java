package me.bausano;

import me.bausano.algorithms.Classifier;
import me.bausano.algorithms.collab.Collab;
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
        DataSet data = DataSet.from(Paths.get(Settings.TRAINING_FILE_PATH), Settings.CROSSFOLD_FACTOR);

        Collab collaboration = new Collab(data.setForTraining);
        collaboration.train();

        Reporter.assess("Multiple networks plus nearest neighbour", collaboration, data.setForValidation);
    }
}
