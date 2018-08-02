package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;

import javax.management.RuntimeErrorException;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import math.utils.StatisticUtils;
import mlp.ANN_MLP;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.ANN_MLP.WEIGHT_INITIATION_METHOD;
import mlp.NeuronFunctionModels;
import mlp.trainer.Backpropagation.COST_FUNCTION_TYPE;
import mlp.trainer.TerminationCriteria;
import mlp.trainer.TerminationCriteria.TERMINATION_CRITERIA;
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
		float Ih = 6f;
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
	 * Test learning rate adaptation. Concept - each time direction of gradient changes, the learning rate divided by half
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
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);
		float[][] weights = mlp.getWeights();
		sut.trainOnSample(td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
		/* Produce error gradient for each output neuron */
		float[][] gradients = mlp.getNetworkNodeGradients(); 

		float[][] expectedWeights = new float[weights.length][];

		float momentum = sut.getMomentum();

		float oldNodeGradient = 0;
		int neuronId = 0;

		float nodeGradient; 
		/* Calculate node gradients for each hidden layers*/
		int layerIdx = mlp.getLayerSizes().length-1;
		/* For each layer, top down*/
		for(int weightLayerIdx = weights.length-1; weightLayerIdx >= 0 ; weightLayerIdx--) {
			/* initiate array for expected weights in layer weightLayerIdx */
			expectedWeights[weightLayerIdx] = new float[weights[weightLayerIdx].length];
			/* for each weight in layer */
			for(int weightIdx = 0; weightIdx < weights[weightLayerIdx].length; weightIdx++) {

				/*  */
				neuronId = weightIdx%mlp.getLayerSizes()[layerIdx];

				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient  = gradients[layerIdx][neuronId];
				if(nodeGradient  > 0 && learningRate > 0) {
					learningRate = learningRate * -1;
				}else if(nodeGradient  < 0 && learningRate < 0){
					learningRate = learningRate * -1;
				}
				expectedWeights[weightLayerIdx][weightIdx] = sut.calculateWeight(nodeGradient, 
						oldNodeGradient, learningRate, momentum, weights[weightLayerIdx][weightIdx]);				
			}
			layerIdx--;
		}
		weights = mlp.getWeights();
		for(int layerId = 0; layerId < expectedWeights.length;layerId++ ) {
			assertArrayEquals(expectedWeights[layerId],weights[layerId]);
		}
	}

	/**
	 * Test learning with adaptive learningRate, using gain attribute per neuron to adapt learning rate per neuron
	 * test MLP consists of input layer size 3, hidden layer 4 neurons, 3 output neurons
	 * hidden layer activation function sigmoid, output activation softmax
	 */
	@Test
	void testGradientDecentWithGainPerNeuron() {
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		int[] layerSizes = new int[] {3,4,3};
		boolean useSoftmax = true;
		float learningRate = 0.01f;
		int iterations = 10;
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);	
		TerminationCriteria tc = new TerminationCriteria(criteria);	
		tc.setIterations(iterations);
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		mlp.setWeights(getTestWeights());
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);
		sut.setTerminationCriteria(tc);

		/* Initial weights */
		float[][] expectedNodeGains = getExpectedGains();

		sut.initiateNodeGains();

		sut.trainOnSampleWithGainParameter(td.getInputRow(td.size()/2), td.getTargetRow(td.size()/2));

		float[][] actualNodeGains = sut.getNodeGains();
		for(int i = 0; i < actualNodeGains.length;i++ ) {
			for(int ii = 0; ii <actualNodeGains[i].length;ii++) {
				actualNodeGains[i][ii] = Math.signum(actualNodeGains[i][ii] );
			}
		}
		for(int i = 0; i < expectedNodeGains.length;i++ ) {
			for(int ii = 0; ii <expectedNodeGains[i].length;ii++) {
				expectedNodeGains[i][ii] = Math.signum(expectedNodeGains[i][ii] );
			}
		}
		for(int layerId = 0; layerId < expectedNodeGains.length;layerId++ ) {
			assertArrayEquals(expectedNodeGains[layerId],actualNodeGains[layerId]);
		}
	}

	/**
	 * Test learning with adaptive learningRate, halve the gain once the neuron gain changes signum
	 * test MLP consists of input layer size 3, hidden layer 4 neurons, 3 output neurons
	 * hidden layer activation function sigmoid, output activation softmax
	 */
	@Test
	void testGradientDecentWithGainAdaptationGainHalvetAtGradientDirectionChange() {
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		int[] layerSizes = new int[] {3,4,3};
		boolean useSoftmax = true;
		float learningRate = 0.01f;
		int iterations = 10;
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);	
		TerminationCriteria tc = new TerminationCriteria(criteria);	
		tc.setIterations(iterations);
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		mlp.setWeights(getTestWeights());
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);
		sut.setTerminationCriteria(tc);
		/* Initial weights */
		float[][] expectedNodeGains = getExpectedGains();

		sut.initiateNodeGains();
		sut.trainOnSampleWithGainParameter(td.getInputRow(td.size()/2), td.getTargetRow(td.size()/2));

		float[][] actualNodeGains = sut.getNodeGains();
		for(int layerId = 0; layerId < expectedNodeGains.length;layerId++ ) {
			assertArrayEquals(expectedNodeGains[layerId],actualNodeGains[layerId]);
		}
	}

	private float[][] getExpectedGains() {
		float[][] nodeGains = {
				{1.0f, 1.0f, 1.0f},
				{-0.5f, -0.5f, -0.5f, -0.5f},
				{-0.5f, -0.5f, -0.5f}
		};
		return nodeGains;
	}

	private float[][] getTestWeights() {
		float[][] weights = new float[2][];
		//Initial LR:0.01
		weights[0] = new float[] {0.26172805f, -0.023212755f, 0.3237994f, 0.33701196f, -0.027228683f, 0.09190323f, -0.00764591f, 
				0.2830995f, 0.27917573f, -0.18975836f, -0.3373269f, 0.32033712f, -0.15263462f, 0.17369014f, 0.20265998f, -0.115796775f};
		weights[1] = new float[]{0.04440312f, 0.09132995f, -0.14755417f, 0.28749457f, -0.26251328f, 0.26934755f, 0.34488472f, 0.025896106f, -0.30336866f,
				0.04440658f, 0.1995952f, 0.23860021f, -0.019049816f, 0.13698725f, 0.043328285f};
		//After LR:-0.01
		return weights;
	}

	/**
	 * Test for calculating of node gradients within network
	 */
	@Test
	void testCalculateNodeGradientsWithinNetwork() {
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
		sut.calculateNetworkNodeGradients(costFType, td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));

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
		for(int layerIdx = mlp.getLayerSizes().length-2; layerIdx > 0 ; layerIdx--) {
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
		/* Gradients are not computed for input nodes */
		expectedGradients[0] = new float[mlp.getLayerSizes()[0]];
		for(int layerIdx = 0;layerIdx < expectedGradients.length;layerIdx++) {
			assertArrayEquals(expectedGradients[layerIdx], actualGradients[layerIdx]);
		}

	}

	/**
	 * Test for calculating of node gradients within network
	 */
	@Test
	void testCalculateNodeGradientsWithinNetworkLocallyStored() {
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
		sut.calculateNetworkNodeGradientsStoredLocaly(costFType, td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));

		float[][] actualGradients = sut.getLocallyStoredNetworkNodeGradients(); 
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
		for(int layerIdx = mlp.getLayerSizes().length-2; layerIdx > 0 ; layerIdx--) {
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
		/* Gradients are not computed for input nodes */
		expectedGradients[0] = new float[mlp.getLayerSizes()[0]];
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

	float[][] calculateNodeGradients(COST_FUNCTION_TYPE costFType,float[][] weights,int[] layerSizes, float[] io,float[][] netInputs,float[] errorVector) {

		int outputLayerIdx = layerSizes.length -1;

		float[][] nodeGradients = new float[layerSizes.length][];
		float[] inputs;

		ACTIVATION_FUNCTION activationFType = ACTIVATION_FUNCTION.SOFTMAX;
		/* Produce error gradient for each output neuron */
		nodeGradients[outputLayerIdx] = new float[errorVector.length];
		for(int errIdx = 0; errIdx < errorVector.length; errIdx++ ) {
			nodeGradients[outputLayerIdx][errIdx] = sut.calculateNodeGradient(costFType, activationFType, errorVector[errIdx], io, errIdx);
		}
		activationFType = ACTIVATION_FUNCTION.SIGMOID;
		/* Calculate node gradients for each hidden layers*/
		for(int layerIdx = layerSizes.length-2; layerIdx > 0 ; layerIdx--) {
			/* Retrieve inputs for the layer */
			inputs =netInputs[layerIdx];
			/* initiate gradient array for layerIdx */
			nodeGradients[layerIdx] = new float[inputs.length];
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < nodeGradients[layerIdx].length; neuronIdx++ ) {
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				nodeGradients[layerIdx][neuronIdx] = sut.calculateNodeGradient(activationFType, nodeGradients[layerIdx+1], inputs[neuronIdx], weights[layerIdx]);
			}
		}
		return nodeGradients;		
	}

	//	void excuteNetwork(float[][] netInputs,float[][] outputs,float[][] weights,float[] input, ACTIVATION_FUNCTION[][] activatiTypes) {
	//		int firstLayerId = 0;
	//		/* set input*/
	//		for(int i = 0; i<input.length;i++) {
	//			netInputs[firstLayerId][i] = input[i];
	//			outputs[firstLayerId][i] = input[i];
	//		}
	//		/* execute network */
	//		for(int layerId = 1; layerId < netInputs.length;layerId++) {
	//			for(int neuronId = 0; neuronId < netInputs[layerId].length;neuronId++) {
	//				netInputs[layerId][neuronId] = 0;
	//				for(int lowerNeuronId = 0; lowerNeuronId < netInputs[layerId-1].length;lowerNeuronId++) {
	//					netInputs[layerId][neuronId] = netInputs[layerId][neuronId] + 
	//							netInputs[layerId-1][lowerNeuronId]*weights[layerId-1][lowerNeuronId*netInputs[layerId-1].length + neuronId];
	//
	//				}
	//				for(int lowerNeuronId = 0; lowerNeuronId < netInputs[layerId-1].length;lowerNeuronId++) {
	//					outputs[layerId][neuronId] = activate(activatiTypes[layerId][neuronId], 
	//							netInputs[layerId][neuronId],neuronId,netInputs[layerId]);
	//				}
	//			}
	//		}
	//	}
	//
	//	private float activate(ACTIVATION_FUNCTION activatTypes, float value,int neuronId,float[] netInputs) {
	//		float a = 1f,b = 1f;;
	//		switch(activatTypes) {
	//		case SIGMOID:
	//			return NeuronFunctionModels.activate(activatTypes, a, b, value);
	//		case SOFTMAX:
	//			return StatisticUtils.calculateSoftmaxPartialDerivative(netInputs, neuronId);
	//		default:
	//			throw new RuntimeException(" not implemented activation function");
	//		}
	//	}
	//
	//	void calculateError(COST_FUNCTION_TYPE fType,float[][] netInputs,float[][] weights,float[] input,float[] target) {
	//		int firstLayerId = 0;
	//		/* set input*/
	//		for(int i = 0; i<input.length;i++) {
	//			netInputs[firstLayerId][i] = input[i];
	//		}
	//		/* execute network */
	//		
	//	}

	/**
	 * Should return two dimensional array, one entity per neuron with 1th as initial value
	 */
	@Test
	void testInitiationNodeGraidients() {
		int[] layerSizes = new int[] {3,4,3};
		ANN_MLP mlp = new ANN_MLP( layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);		
		float[][] nodeGains = sut.initiateNodeGains();
		float[][] expected = new float[layerSizes.length][];
		for(int layerId = 0; layerId < expected.length; layerId++) {
			expected[layerId] = new float[layerSizes[layerId]];
			for(int i = 0; i < expected[layerId].length;i++) {
				expected[layerId][i] = 1;
			}
		}
		for(int layerId = 0; layerId < nodeGains.length; layerId++) {
			assertArrayEquals(expected,nodeGains);
		}
	}


}