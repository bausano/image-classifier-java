package me.bausano.algorithms.neuralnetwork;

import java.util.function.Function;

public class ActivationMapper {

    /**
     * Activation function that is used to introduce non linearity to the network.
     */
    public final Function<Double, Double> function;

    /**
     * Function that takes result of previous activation function and computes the derivative for that value. This is
     * not optimal as we are limited on what activation functions can we use. However both relu and sigmoid is well
     * derivable and those are the main functions we use.
     */
    public final Function<Double, Double> derivative;

    /**
     * @param function Non linear function
     * @param derivative Transformer that takes output of the function and calculates the derivative at that point
     */
    public ActivationMapper(Function<Double, Double> function, Function<Double, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    /**
     * Returns an instance of sigmoid activation function.
     *
     * @return Instance with sigmoid
     */
    public static ActivationMapper sigmoid() {
        return new ActivationMapper(
                x -> 1d / (1d + Math.pow(Math.E, -x)),
                x -> x * (1d - x)
        );
    }

}
