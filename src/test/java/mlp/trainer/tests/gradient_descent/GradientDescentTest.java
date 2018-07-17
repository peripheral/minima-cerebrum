package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.trainer.Backpropagation.COST_FUNCTION_TYPE;
import mlp.trainer.gradient_descent.GradientDescent;

public class GradientDescentTest{
	
	/**
	 * System under test
	 */
	private GradientDescent sut;

	@BeforeEach
	void init() {
		sut = new GradientDescent();
	}
	
	/**
	 * Test for function that calculates gradient of input to error in output layer
	 * of the output neuron
	 * Io - neuron input, E = (Predicted - Required)
	 * ∂(E)^2/∂Io = gradient
	 * (∂E^2/∂Io) => 2E  - first step of derivation
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2E * fo'(Io) = gradient
	 */
	@Test
	void testCalculateGradientOfChangeInSquaredErrorVs() {
		COST_FUNCTION_TYPE costFType = COST_FUNCTION_TYPE.SQUARED_ERROR;
		ACTIVATION_FUNCTION outputActivationFunction = ACTIVATION_FUNCTION.SOFTMAX;
		float a = 1;
		float b = 1;
		float[] Io = {3f,4f,6f};
		int neuronIdx = 1;
		float error = 0.5f;
		/* Derivative of softmax  ((e^4)((e^3)+(e^4)+(e^6)) - (e^4)(e^4))/((e^3)+(e^4)+(e^6))^2 = 0.10115465582 */
		float softmaxDerivative = 0.10115465582f;
		/* Gradient 2 * E * derivative =  2 * 0.5 * 0.10115465582 = 0.10115465582  */
		float expected = 2 * error * softmaxDerivative;
		float actual = sut.calculateGradientInputOverError(costFType, outputActivationFunction, error, Io, neuronIdx, a, b);
		assertEquals(expected,actual);
	}
	
	/**
	 * Test for function that calculates gradient of input to error in output layer
	 * of the output neuron
	 * Io - neuron input, E = (Predicted - Required)
	 * ∂(E)^2/∂Io = gradient
	 * (∂E^2/∂Io) => 2E  - first step of derivation
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2E * fo'(Io) = gradient
	 * ∂(E)^2/∂Ih = (∂E^2/∂Io) * (∂Io/∂Oh) * (∂Oh/∂Ih)
	 * ∂Io/∂Oh => ∂(OpWpo + Op+1Wp+1 .. OhWho )/∂Oh => Who 
	 * ∂Oh/∂Ih => fsig'(Ih)
	 *  Gradient_l-1 = Gradient * Who * fsig'(Ih)
	 */
	@Test
	void testCalculateGradientInputToErrorInPreceedingLayers() {
		ACTIVATION_FUNCTION activationFunction = ACTIVATION_FUNCTION.SIGMOID;
		float a = 1;
		float b = 1;
		float[] Io = {3f,4f,6f};
		float Ih = 6f;
		int neuronIdx = 1;
		float error = 0.5f;
		float Who = 0.045f; /* weight from neuron h to o*/
		/* Derivative of softmax  ((e^4)((e^3)+(e^4)+(e^6)) - (e^4)(e^4))/((e^3)+(e^4)+(e^6))^2 = 0.10115465582 */
		float softmaxDerivative = 0.10115465582f;
		/* Derivative of sigmoid (b*a*2*(Math.pow(Math.E,-a*x))/Math.pow(1+Math.pow(Math.E,-a*x),2)) = 
		 * = (1 * 1 * 2 * e^(-6))/(1+ e^(-6))^2 = 0.00493301858 */
		float sigmoidDerivative = 0.00493301858f;
		/* Gradient = 2 * E * derivative =  2 * 0.5 * 0.10115465582 = 0.10115465582  */
		float gradient = 2 * error * softmaxDerivative;
		/* expected = 0.10115465582 * 0.045f * 0.00493301858f = 0.0000224549 */
		float expected = gradient * Who * sigmoidDerivative;
		float actual = sut.calculateGradientInputOverError(activationFunction, gradient, Ih, Who, a, b);
		assertEquals(expected,actual);
	}
}