package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANN_MLP;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.ANN_MLP.WEIGHT_INITIATION_METHOD;
import mlp.trainer.Backpropagation.COST_FUNCTION_TYPE;
import mlp.trainer.TrainingData;
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
	void testCalculateNodeGradient() {
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
		float actual = sut.calculateNodeGradient(costFType, outputActivationFunction, error, Io, neuronIdx, a, b);
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
	void testCalculationNodeGradient() {
		ACTIVATION_FUNCTION activationFunction = ACTIVATION_FUNCTION.SIGMOID;
		float a = 1;
		float b = 1;
		float[] Io = {3f,4f,6f};
		float Ih = 6f;
		int neuronIdx = 1;
		float error = 0.5f;
		float[] Who = {0.045f,0.045f}; /* weight from neuron h to o*/
		/* Derivative of softmax  ((e^4)((e^3)+(e^4)+(e^6)) - (e^4)(e^4))/((e^3)+(e^4)+(e^6))^2 = 0.10115465582 */
		float softmaxDerivative = 0.10115465582f;
		/* Derivative of sigmoid (b*a*2*(Math.pow(Math.E,-a*x))/Math.pow(1+Math.pow(Math.E,-a*x),2)) = 
		 * = (1 * 1 * 2 * e^(-6))/(1+ e^(-6))^2 = 0.00493301858 */
		float sigmoidDerivative = 0.00493301858f;
		/* Gradient = 2 * E * derivative =  2 * 0.5 * 0.10115465582 = 0.10115465582  */
		float[] outputNodeGradients = { 2 * error * softmaxDerivative,  2 * error * softmaxDerivative};
		/* expected = 0.10115465582 * 0.045f * 0.00493301858f = 0.0000224549 */
		float[] expected = {outputNodeGradients[0] * Who[0] * sigmoidDerivative,outputNodeGradients[1] * Who[1] * sigmoidDerivative};
		float[] actual = sut.calculateNodeGradient(activationFunction, outputNodeGradients, Ih, Who);
		assertArrayEquals(expected,actual);
	}
	
	/**
	 * Test for function that calculates delta weight for Who, where h sending neuron 
	 * and o receiver.
	 * Io - neuron input
	 * ∂(E)^2/∂Io = gradient
	 * (∂E^2/∂Io) => 2E  - first step of derivation
	 * (∂f(Io)/∂Io) => f'(Io)
	 * 2E * fo'(Io) = gradient
	 * ∂(E)^2/∂Ih = (∂E^2/∂Io) * (∂Io/∂Oh) * (∂Oh/∂Ih) = gradient
	 * ∂(E)^2/∂Who = (∂E^2/∂Io) * (∂Io/∂Oh) * (∂Oh/∂Ih) * (∂Ih/∂Who)
	 * (∂Ih/∂Who) => ∂(OpWpo+Op+1Wp+1o+ ..+OhWho) = Oh
	 *  delta = gradient * Oh
	 */
	@Test
	void testCalculateDeltaWeight() {
		float[] Oh = {3f,4f,6f};
		int neuronIdx = 1;
		float error = 0.5f;
		float softmaxDerivative = 0.10115465582f;
		float currentWeight = 0.03f;
		float gradient = 2 * error * softmaxDerivative;
		float expected = gradient * Oh[neuronIdx] * currentWeight; 
		/* Oh[neuronIdx] - output of sending neuron */
		float actual = sut.calculateDeltaWeight(gradient, Oh[neuronIdx], currentWeight);
		assertEquals(expected,actual);
	}
	
	/**
	 *Test for update weight rule
	 */
	@Test
	void testUpdateWeightRule() {
		float[] Oh = {3f,4f,6f};
		int neuronIdx = 1;
		float error = 0.5f;
		float oldDelta = 0.004f;
		float softmaxDerivative = 0.10115465582f;
		float currentWeight = 0.03f;
		float oldWeight = currentWeight;
		float momentum = 0;
		float gradient = 2 * error * softmaxDerivative;
		float learningRate = 0.001f;
		float deltaWeight = sut.calculateDeltaWeight(gradient, Oh[neuronIdx], currentWeight);
		/* Oh[neuronIdx] - output of sending neuron */
		float actual =  sut.calculateWeight(deltaWeight,oldDelta, learningRate,momentum,oldWeight);
		float expected = oldWeight - (learningRate * deltaWeight * currentWeight) - (momentum * oldDelta * currentWeight);
		assertEquals(expected,actual);
	}	
	
	/**
	 * Test for whole learning cycle of online learning
	 * test MLP consists of input layer size 3, hidden layer 4 neurons, 3 output neurons
	 * hidden layer activation function sigmoid, output activation softmax
	 */
//	@Test
//	void testGradientDecentOneSample() {
//		int[] layerSizes = new int[] {3,4,3};
//		boolean useSoftmax = true;
//		int trainingRowId = 0;
//		COST_FUNCTION_TYPE costFType = COST_FUNCTION_TYPE.SQUARED_ERROR;
//		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
//		mlp.initiate();
//		sut.setMLP(mlp);
//		float[][] data = {{1f,0f,0f,1f,0f,0f},
//				{0f,1f,0f,1f,0f,0f},
//				{0f,0f,1f,1f,0f,0f},
//				{1f,1f,0f,0f,1f,0f},
//				{0f,1f,1f,0f,1f,0f},
//				{1f,0f,1f,0f,1f,0f},
//				{1f,1f,1f,0f,0f,1f},
//				{0f,0f,0f,0f,0f,1f},
//				{1f,1f,1f,0f,0f,1f}};
//		TrainingData td = new TrainingData(data, 3);
//		sut.setTrainingData(td);	
//		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
//		sut.trainOnSample(td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
//		float[][] weights = mlp.getWeights();
//		float[][] weightsExpected = new float[weights.length][weights[0].length];
//		float[] errorVector = sut.getErrorPerNeuron(costFType, td.getInputRow(trainingRowId), td.getTargetRow(trainingRowId));
//		float[] gradient = new float[errorVector.length]; 
//		float[] weightGradient ;
//		float momentum = sut.getMomentum();
//		int outputLayerIdx = mlp.getLayerSizes().length -1;
//		float[] io = mlp.getLayer(outputLayerIdx).getNetInputs();
//		ACTIVATION_FUNCTION activationFType = ACTIVATION_FUNCTION.SOFTMAX;
//		/* Produce error gradient per neuron */
//		for(int errIdx = 0; errIdx < errorVector.length; errIdx++ ) {
//			gradient[errIdx] = sut.calculateGradientInputOverError(costFType, activationFType, errorVector[errIdx], io, errIdx);
//		}
//		float oldGradient = 0;
//		float[] inputs;
//		float learningRate = sut.getLearningRate();
//		int neuronId = 0;
//		/* for each layer calculate new weights */
//		for(int layerIdx = weights.length-1; layerIdx > 0 ; layerIdx--) {
//			if(layerIdx < weights.length-1) {
//				inputs = mlp.getLayer(layerIdx).getNetInputs();
//				for(int neuronIdx = 0; neuronIdx < errorVector.length; neuronIdx++ ) {
//					//gradient[neuronIdx] = sut.calculateGradientInputOverError(activationFType, gradient, inputs[0], weights[layerIdx][0]);
//				}
//			}
//			for(int weightIdx = 0; weightIdx < weights[layerIdx].length; weightIdx++) {
//				weightsExpected[layerIdx][weightIdx] = sut.calculateWeight(gradient[neuronId], oldGradient, learningRate, momentum, weights[layerIdx][weightIdx]);
//				
//			}
//		}
//	}	
//	
	/**
	 * test of set get cost function
	 */
	@Test
	void testSetGetCostFunction() {
		COST_FUNCTION_TYPE costFunction  = COST_FUNCTION_TYPE.SQUARED_ERROR;
		sut.setCostFunctionType(costFunction);
		COST_FUNCTION_TYPE expected = COST_FUNCTION_TYPE.SQUARED_ERROR;
		COST_FUNCTION_TYPE actual = sut.getCostFunctionType();
		assertEquals(expected, actual);
	}
	
	/**
	 * test of set get learning momentum
	 */
	@Test
	void testSetGetMomentum() {
		float momentum = 0.001f;
		sut.setMomentum(momentum);
		float actual = sut.getMomentum();
		float expected = 0.001f;
		assertEquals(expected, actual);
	}
	
	/**
	 * test of get default learning momentum 
	 */
	@Test
	void testGetDefaultMomentum() {
		float actual = sut.getMomentum();
		float expected = 0.00001f;
		assertEquals(expected, actual);
	}
	
	/**
	 * test of set get learning rate
	 */
	@Test
	void testSetGetLearningRate() {
		float momentum = 0.001f;
		sut.setLearningRate(momentum);
		float actual = sut.getLearningRate();
		float expected = 0.001f;
		assertEquals(expected, actual);
	}
	
	/**
	 * test of get default learning momentum 
	 */
	@Test
	void testGetDefaultLearningRate() {
		float actual = sut.getLearningRate();
		float expected = 0.001f;
		assertEquals(expected, actual);
	}
	
}