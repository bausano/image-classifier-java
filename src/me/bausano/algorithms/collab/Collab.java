package me.bausano.algorithms.collab;

import me.bausano.algorithms.Classifier;
import me.bausano.algorithms.nearestneighbour.NearestNeighbour;
import me.bausano.algorithms.neuralnetwork.NeuralNetwork;
import me.bausano.algorithms.neuralnetwork.Trainer;

public class Collab implements Classifier {

    /**
     * Data set of structured input data to train on.
     */
    private final double[][] data;

    /**
     * Network only for numbers 0, 1, 3, 6 and 9.
     */
    private NeuralNetwork groupA = NeuralNetwork.fromBlueprint(
            new int[] { 64, 24, 6 },
            new int[] { 0, 1, 3, 6, 9, -1 },
            new int[] { 0, 1, 5, 2, 5, 5, 3, 5, 5, 4 }
    );

    /**
     * Train network only for numbers 2, 4, 5, 7 and 8.
     */
    private NeuralNetwork groupB = NeuralNetwork.fromBlueprint(
            new int[] { 64, 24, 6 },
            new int[] { 2, 4, 5, 7, 8, -1 },
            new int[] { 5, 5, 0, 5, 1, 2, 5, 3, 4, 5 }
    );

    /**
     * Nearest neighbour instance.
     */
    private final NearestNeighbour nn;

    /**
     * @param data Input data set where last int is the class
     */
    public Collab(double[][] data) {
        this.data = data;
        this.nn = new NearestNeighbour(data);
    }

    /**
     * Trains the algorithm.
     */
    public void train() {
        new Trainer(groupA, data).train();
        new Trainer(groupB, data).train();
    }

    /**
     * @inheritDoc
     */
    public int classify (double[] digit) {
        int groupAClass = groupA.classify(digit);
        int groupBClass = groupB.classify(digit);
        int nnClass = nn.classify(digit);

        System.out.printf("\nA: %d, B: %d, NN: %d", groupAClass, groupBClass, nnClass);

        if (groupAClass == -1 && groupBClass != -1) {
            return groupBClass;
        }

        if (groupBClass == -1 && groupAClass != -1) {
            return groupAClass;
        }

        return nnClass;
    }
}
