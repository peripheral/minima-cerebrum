package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;

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
	 * hidden node gradient = f'(Ih)*(Whi* Gradrient_i + Whi+1* Gradrient_i+1 .. + )
	 * Ih - input to node, Whi - weight from node h to node i, Gradient_i - gradient for neuron 
	 * for node i 
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
		/* Gradient = f'(neuronInput)* (Whi* Gradrient_i + Whi+1* Gradrient_i+1 .. )*/
		float expected = outputNodeGradients[0] * Who[0] * sigmoidDerivative + outputNodeGradients[1] * Who[1] * sigmoidDerivative;
		float actual = sut.calculateNodeGradient(activationFunction, outputNodeGradients, Ih, Who);
		assertEquals(expected,actual);
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
		float gradient = 2 * error * softmaxDerivative;
		float expected = gradient * Oh[neuronIdx]; 
		/* Oh[neuronIdx] - output of sending neuron */
		float actual = sut.calculateDeltaWeight(gradient, Oh[neuronIdx]);
		assertEquals(expected,actual);
	}

	/**
	 *Test for update weight rule
	 * Learning rate 
	 */
	@Test
	void testUpdateWeightRule() {
		float[] Oh = {3f,4f,6f};
		int neuronIdx = 1;
		float error = 0.5f;
		float oldDelta = 0.004f;
		float softmaxDerivative = 0.10115465582f;
		float currentWeight = 0.03f;
		float momentum = 0;
		/* Gradient = 2* 0.5 * 0.10115465582f */
		float gradient = 2 * error * softmaxDerivative;
		float learningRate = 0.001f;
		/* Delta weight = 0.10115465582f * 4f ~ 0.4044  */
		float deltaWeight = sut.calculateDeltaWeight(gradient, Oh[neuronIdx]);
		if(deltaWeight > 0 && learningRate > 0) {
			learningRate = learningRate * -1;
		}else if(deltaWeight < 0 && learningRate < 0) {
			learningRate = learningRate * -1;
		}
		/* Oh[neuronIdx] - output of sending neuron 
		 * case: deltaWeight < 0
		 * new_weight = dletaWeightOld*momentum + learningRate*deltaWeight 
		 * case:deltaWeight > 0
		 * new_weight = dletaWeightOld*momentum + learningRate*deltaWeight * -1
		 */
		float actual =  sut.calculateWeight(deltaWeight,oldDelta, learningRate,momentum,currentWeight);
		float expected = currentWeight + learningRate * deltaWeight + (momentum * oldDelta);
		assertEquals(expected,actual);
	}	

	/**
	 * Test for whole learning cycle of online learning
	 * test MLP consists of input layer size 3, hidden layer 4 neurons, 3 output neurons
	 * hidden layer activation function sigmoid, output activation softmax
	 */
	@Test
	void testGradientDecentOneSample() {
		int[] layerSizes = new int[] {3,4,3};
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);
		boolean useSoftmax = true;
		int trainingRowId = 0;
		float learningRate = 0.01f;
		COST_FUNCTION_TYPE costFType = COST_FUNCTION_TYPE.SQUARED_ERROR;
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);
		sut.trainOnSample(td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
		float[][] weights = mlp.getWeights();
		float[][] expectedWeights = new float[weights.length][];
		float[] errorVector = sut.calculateErrorPerNeuron(costFType, td.getInputRow(trainingRowId), td.getTargetRow(trainingRowId));
		float[][] gradients = new float[mlp.getLayerSizes().length][]; 

		float momentum = sut.getMomentum();
		int outputLayerIdx = mlp.getLayerSizes().length -1;
		float[] io = mlp.getLayer(outputLayerIdx).getNetInputs();

		float oldGradient = 0;
		float[] inputs;

		int neuronId = 0;
		ACTIVATION_FUNCTION activationFType = ACTIVATION_FUNCTION.SOFTMAX;
		/* Produce error gradient for each output neuron */
		gradients[outputLayerIdx] = new float[errorVector.length];
		for(int errIdx = 0; errIdx < errorVector.length; errIdx++ ) {
			gradients[outputLayerIdx][errIdx] = sut.calculateNodeGradient(costFType, activationFType, errorVector[errIdx], io, errIdx);
		}
		activationFType = ACTIVATION_FUNCTION.SIGMOID;
		/* Calculate node gradients for each hidden layers*/
		for(int layerIdx = mlp.getLayerSizes().length-2; layerIdx >= 0 ; layerIdx--) {
			/* Retrieve inputs for the layer */
			inputs = mlp.getLayer(layerIdx).getNetInputs();
			/* initiate gradient array for layerIdx */
			gradients[layerIdx] = new float[inputs.length];
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < gradients[layerIdx].length; neuronIdx++ ) {
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				gradients[layerIdx][neuronIdx] = sut.calculateNodeGradient(activationFType, gradients[layerIdx+1], inputs[neuronIdx], weights[layerIdx]);
			}
		}
		int layerIdx = mlp.getLayerSizes().length-1;
		/* Calculate weights per layer */
		for(int weightLayerIdx = weights.length-1; weightLayerIdx >= 0 ; weightLayerIdx--) {
			expectedWeights[weightLayerIdx] = new float[weights[weightLayerIdx].length];
			for(int weightIdx = 0; weightIdx < weights[weightLayerIdx].length; weightIdx++) {
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
			
				neuronId = weightIdx%mlp.getLayerSizes()[layerIdx];

				
				if(gradients[layerIdx][neuronId] > 0 && learningRate > 0) {
					learningRate = learningRate * -1;
				}else if(gradients[layerIdx][neuronId] < 0 && learningRate < 0){
					learningRate = learningRate * -1;
				}
				
				expectedWeights[weightLayerIdx][weightIdx] = sut.calculateWeight(gradients[layerIdx][neuronId], 
						oldGradient, learningRate, momentum, weights[weightLayerIdx][weightIdx]);				
			}
			layerIdx--;
		}
	
		for(int layerId = 0; layerId < expectedWeights.length;layerId++ ) {
			assertArrayEquals(expectedWeights[layerId],weights[layerId]);
		}
	}
	
	/**
	 * Test for calculating of node gradients within network
	 */
	@Test
	void testCalculateNodeGradients() {
		int[] layerSizes = new int[] {3,4,3};
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);
		boolean useSoftmax = true;
		int trainingRowId = 0;
		float learningRate = 0.01f;
		COST_FUNCTION_TYPE costFType = COST_FUNCTION_TYPE.SQUARED_ERROR;
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);

		float[] errorVector = sut.calculateErrorPerNeuron(costFType, td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
		sut.calculateNetworkNodeGradients(errorVector);
	
		float[][] actualGradients = mlp.getNetworkNodeGradients(); 
		float[][] weights = mlp.getWeights();
		int outputLayerIdx = mlp.getLayerSizes().length -1;
		float[] io = mlp.getLayer(outputLayerIdx).getNetInputs();
		float[][] expectedGradients = new float[actualGradients.length][];
		float[] inputs;

		ACTIVATION_FUNCTION activationFType = ACTIVATION_FUNCTION.SOFTMAX;
		/* Produce error gradient for each output neuron */
		expectedGradients[outputLayerIdx] = new float[errorVector.length];
		for(int errIdx = 0; errIdx < errorVector.length; errIdx++ ) {
			expectedGradients[outputLayerIdx][errIdx] = sut.calculateNodeGradient(costFType, activationFType, errorVector[errIdx], io, errIdx);
		}
		activationFType = ACTIVATION_FUNCTION.SIGMOID;
		/* Calculate node gradients for each hidden layers*/
		for(int layerIdx = mlp.getLayerSizes().length-2; layerIdx >= 0 ; layerIdx--) {
			/* Retrieve inputs for the layer */
			inputs = mlp.getLayer(layerIdx).getNetInputs();
			/* initiate gradient array for layerIdx */
			expectedGradients[layerIdx] = new float[inputs.length];
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < expectedGradients[layerIdx].length; neuronIdx++ ) {
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				expectedGradients[layerIdx][neuronIdx] = sut.calculateNodeGradient(activationFType, expectedGradients[layerIdx+1], inputs[neuronIdx], weights[layerIdx]);
			}
		}
		for(int layerIdx = 0;layerIdx < expectedGradients.length;layerIdx++) {
			assertArrayEquals(expectedGradients[layerIdx], actualGradients[layerIdx]);
		}
		
	}

	private float[][] getTrainingDataGD() {
		/* two dimensional array with 3 input and 3 output elements */
		return new float[][]{{1f,0f,0f,1f,0f,0f},
			{0f,1f,0f,1f,0f,0f},
			{0f,0f,1f,1f,0f,0f},
			{1f,1f,0f,0f,1f,0f},
			{0f,1f,1f,0f,1f,0f},
			{1f,0f,1f,0f,1f,0f},
			{1f,1f,1f,0f,0f,1f},
			{0f,0f,0f,0f,0f,1f},
			{1f,1f,1f,0f,0f,1f}};		
	}

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
	 * 
	 */
	@Test
	void testGetDefaultLearningRate() {
		float actual = sut.getLearningRate();
		float expected = 0.001f;
		assertEquals(expected, actual);
	}

}