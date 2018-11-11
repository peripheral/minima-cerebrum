package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import junit.framework.Assert;
import math.utils.StatisticUtils;
import mlp.ANNMLP;
import mlp.ANNMLP.ACTIVATION_FUNCTION;
import mlp.ANNMLP.WEIGHT_INITIATION_METHOD;
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
	 * Tests performs test on error calculation from target and observed
	 * Error = (target - observed)
	 */
	@Test
	void testCalculateErrorPerNeuron() {
		int[] layerSizes = {3,4,3};
		WEIGHT_INITIATION_METHOD weightInitiationMethod = WEIGHT_INITIATION_METHOD.RANDOM;
		ANNMLP mlp = new ANNMLP(weightInitiationMethod, layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);
		TrainingData td = new TrainingData(getTrainingDataGD(), 3);
		float[] target = td.getTargetRow(td.size()/2);
		float[] input = td.getInputRow(td.size()/2);
		float[] actual = sut.calculateErrorPerNeuron(input,target );		
		float[] expected = 	mlp.predict(input);
		for(int i = 0; i < expected.length;i++) {
			expected[i] =target[i] - expected[i];
		}
		assertArrayEquals(expected,actual);
	}

	/**
	 * Test for function that calculates gradient of input to error in output layer
	 * of the output neuron. with attributes a and b
	 * Io - neuron input, E = (Predicted - Required)
	 * ∂(E)^2/∂Io = gradient
	 * (∂E^2/∂Io) => 2E  - first step of derivation
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2E * fo'(Io) = gradient
	 */
	@Test
	void testCalculateNodeGradientOutputNeuronWithAtrAB() {
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
		float expected = 2 * error * -softmaxDerivative;
		float actual = sut.calculateNodeGradient(costFType, outputActivationFunction, error, Io, neuronIdx, a, b);
		assertEquals(expected,actual);
	}

	/**
	 * Test for function that calculates gradient of input to error in output neurons according to delta rule
	 * Io - neuron input, E = (Required - Predicted)^2
	 * Oo - output of output neuron, 
	 * Io - input of output neuron
	 * ∂(E)^2/∂Io = ∂(target - observed)^2/∂Oo * -∂fo(Io)/∂Io = gradient
	 * (∂E^2/∂Io) => 2(target - observed)  - first step of derivation
	 * ∂(target - observed)/∂Oo => -Oo 
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2(target - observed) * -fo'(Io) = gradient
	 */
	@Test
	void testCalculateNodeGradientOutputNeuron() {
		ACTIVATION_FUNCTION activationFunction = ACTIVATION_FUNCTION.SOFTMAX;
		COST_FUNCTION_TYPE costF = COST_FUNCTION_TYPE.SQUARED_ERROR;
		/* (target - observed) */
		float diffTarPred = 0.5f;
		float[] IoAll = {3f,4f,6f};
		int neuronId = 1;
		/* Derivative of softmax  ((e^4)((e^3)+(e^4)+(e^6)) - (e^4)(e^4))/((e^3)+(e^4)+(e^6))^2 = 0.10115465582 */
		float softmaxDerivative = 0.10115465582f;
		/* 2(target - observed) * -fo'(Io) = gradient */
		float expected = 2 * diffTarPred * -1 * softmaxDerivative;
		float actual = sut.calculateOutputNodeGradient(costF, activationFunction,  diffTarPred,IoAll,neuronId);
		assertEquals(expected,actual);
	}

	/**
	 * Test for function that calculates gradient of input to error in output layer
	 * of the output neuron
	 * Io - neuron input, E = (Required - Predicted)
	 * ∂(E)^2/∂Io = ∂(E)^2/∂Io * ∂(E)/∂Io * ∂fo(Io)/∂Io = gradient
	 * (∂E^2/∂Io) => 2E  - first step of derivation
	 * ∂(Required - Predicted)/∂Io => -Oo => -fo(.)
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2E * -fo'(Io) = gradient
	 * hidden node gradient = f'(Ih)*(Whi* Gradrient_i + Whi+1* Gradrient_i+1 .. + )
	 * Ih - input to node, Whi - weight from node h to node i, Gradient_i - gradient for neuron 
	 * for node i 
	 */
	@Test
	void testCalculationNodeGradientHidden() {
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
		float[] outputNodeGradients = { 2 * error * -softmaxDerivative,  2 * error * -softmaxDerivative};
		/* Gradient = f'(neuronInput)* (Whi* oGradrient_i + Whi+1* oGradrient_i+1 .. )*/
		float expected = sigmoidDerivative * (outputNodeGradients[0] * Who[0] + outputNodeGradients[1] * Who[1]);
		float actual = sut.calculateNodeGradient(activationFunction, outputNodeGradients, Who, Ih);
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
	void testCalculateWeightDelta() {
		float[] Oh = {3f,4f,6f};
		int neuronIdx = 1;
		float error = 0.5f;
		float softmaxDerivative = 0.10115465582f;
		float gradient = 2 * error * softmaxDerivative;
		float expected = gradient * Oh[neuronIdx]; 
		/* Oh[neuronIdx] - output of sending neuron */
		float actual = sut.calculateWeightDelta(gradient, Oh[neuronIdx]);
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
		float momentumDecayFactor = 0;
		/* Gradient = 2* 0.5 * 0.10115465582f */
		float gradient = 2 * error * softmaxDerivative;
		float learningRate = 0.001f;
		/* Delta weight = 0.10115465582f * 4f ~ 0.4044  */
		float calculatedDeltaWeight = sut.calculateWeightDelta(gradient, Oh[neuronIdx]);

		float actual =  sut.calculateWeight(calculatedDeltaWeight,oldDelta, learningRate,momentumDecayFactor,currentWeight);
		float expected = currentWeight + (momentumDecayFactor * oldDelta) - learningRate * calculatedDeltaWeight;
		assertEquals(expected,actual);
	}	

	/**
	 *Test for update weight rule with momentum
	 * Learning rate 
	 */
	@Test
	void testUpdateWeightRuleWithMomentum() {
		float[] Oh = {3f,4f,6f};
		int neuronIdx = 1;
		float error = 0.5f;
		float oldDeltaWeight = 0.004f;
		float softmaxDerivative = 0.10115465582f;
		float currentWeight = 0.03f;
		float momentumDecayFactor = 0.96f;
		sut.setMomentumDecayFactor(momentumDecayFactor);
		/* Gradient = 2* 0.5 * 0.10115465582f */
		float gradient = 2 * error * softmaxDerivative;
		float learningRate = 0.001f;
		/* Delta weight = 0.10115465582f * 4f ~ 0.4044  */
		float calculatedDeltaWeight = sut.calculateWeightDelta(gradient, Oh[neuronIdx]);

		float actual =  sut.calculateWeight(calculatedDeltaWeight,oldDeltaWeight, learningRate,momentumDecayFactor,currentWeight);
		float expected = currentWeight + (momentumDecayFactor * oldDeltaWeight) - learningRate * calculatedDeltaWeight;
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
		ANNMLP mlp = new ANNMLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);
		float[][] weights = mlp.getWeights();
		sut.trainOnSample(td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
		/* Produce error gradient for each output neuron */
		float[][] gradients = sut.getNetworkNodeGradients(); 

		float[][] expectedWeights = new float[weights.length][];

		float momentumDecayFactor = sut.getMomentumDecayFactor();

		float previousDeltaWeight = 0;
		int neuronId = 0;
		int lowerNeuronId = 0;
		float calculatedDedeltaWeight = 0;
		float[] outputs;
		float nodeGradient; 

		/* For each layer, top down*/
		for(int weightLayerIdx = weights.length-1; weightLayerIdx >= 0 ; weightLayerIdx--) {
			/* initiate array for expected weights in layer weightLayerIdx */
			expectedWeights[weightLayerIdx] = new float[weights[weightLayerIdx].length];
			/* get outputs from layer lower */
			outputs = mlp.getLayer(weightLayerIdx).getOutputs();
			/* for each weight in layer */
			for(int weightIdx = 0; weightIdx < weights[weightLayerIdx].length; weightIdx++) {

				/* idx of node gradient for upper layer, weightLayerIdx+1*/
				neuronId = weightIdx%mlp.getLayerSizes()[weightLayerIdx+1];
				/* get node gradient from neuron of upper layer */
				nodeGradient  = gradients[weightLayerIdx][neuronId];
				/* get index of neuron of connecting layer with current weightIdx, weightIdx/upperLayerSize */
				lowerNeuronId = weightIdx/gradients[weightLayerIdx].length;
				/* currently biases are not included, must check for outofbound exception */
				if(lowerNeuronId <outputs.length ) {
					calculatedDedeltaWeight = sut.calculateWeightDelta(nodeGradient, outputs[lowerNeuronId]);
				}else {
					/* Bias */
					calculatedDedeltaWeight = sut.calculateWeightDelta(nodeGradient, 1);
				}

				expectedWeights[weightLayerIdx][weightIdx] = sut.calculateWeight(calculatedDedeltaWeight, 
						previousDeltaWeight, learningRate, momentumDecayFactor, weights[weightLayerIdx][weightIdx]);				
			}
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
	void testGradientDecentWithGainPerNeuronGainsShouldDecreaseWhenGradientSwitchesDirection() {
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		int[] layerSizes = new int[] {3,4,3};
		boolean useSoftmax = true;
		float learningRate = 0.01f;
		int iterations = 10;
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);	
		TerminationCriteria tc = new TerminationCriteria(criteria);	
		tc.setIterations(iterations);
		ANNMLP mlp = new ANNMLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		mlp.setWeights(getTestWeights2());
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);
		sut.setTrainingTerminationCriteria(tc);

		/* Initial weights */
		float[][] expectedNodeGains = getExpectedDecreasedGains();

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
	void testGradientDecentWithGainAdaptationGainHalvedAtGradientDirectionChangeWithDeltaRule() {
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		int[] layerSizes = new int[] {3,4,3};
		boolean useSoftmax = true;
		float learningRate = 0.01f;
		int iterations = 10;
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);	
		TerminationCriteria tc = new TerminationCriteria(criteria);	
		tc.setIterations(iterations);
		ANNMLP mlp = new ANNMLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		mlp.setWeights(getTestWeights2());
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
		sut.setLearningRate(learningRate);
		sut.setTrainingTerminationCriteria(tc);
		/* Initial weights */
		float[][] expectedNodeGains = getExpectedDecreasedGains();
		sut.trainOnSampleWithGainParameterWithoutGainMagnitudeModificationWithDelta(td.getInputRow(td.size()/2), td.getTargetRow(td.size()/2));
		sut.trainOnSampleWithGainParameterWithDeltaRule(td.getInputRow(td.size()/2), td.getTargetRow(td.size()/2));

		float[][] actualNodeGains = sut.getNodeGains();
		for(int layerId = 0; layerId < expectedNodeGains.length;layerId++ ) {
			assertArrayEquals(expectedNodeGains[layerId],actualNodeGains[layerId]);
		}
	}

	private float[][] getExpectedDecreasedGains() {
		float[][] nodeGains = { 
				{1.001f, 1.001f, 1.001f, 1.001f},
				{1.001f, 1.001f, 1.001f}
		};
		return nodeGains;
	}

	private float[][] getTestWeights2() {
		float[][] weights = new float[2][];
		//Initial LR:0.01
		weights[0] = new float[] {-0.11937232f, -0.31125507f, 0.065697454f, -0.09223979f, -0.03957021f, 0.18262728f, -0.2717147f,
				0.084797226f, -0.33899653f, 0.29077682f, 0.21645027f, -0.06085978f, -0.026478386f, 0.102951825f, -0.19819133f, -0.058825392f};
		weights[1] = new float[]{-0.17163269f, 0.009470023f, 0.031649176f, 0.22646435f, 0.083408356f, 0.026134368f, 0.18263562f, 
				-0.30966148f, -0.23505746f, -0.29305205f, 0.3162114f, -0.07728336f, -0.27994624f, -0.00919347f, 0.025387315f};
		//After LR:-0.01
		return weights;
	}

	//	/**
	//	 * Test learning with adaptive learningRate, halve the gain once the neuron gain changes signum
	//	 * test MLP consists of input layer size 3, hidden layer 4 neurons, 3 output neurons
	//	 * hidden layer activation function sigmoid, output activation softmax
	//	 */
	//	@Test
	//	void testIncreaseNodeLearningGainAtStepRepetion() {
	//		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
	//		int[] layerSizes = new int[] {3,4,3};
	//		boolean useSoftmax = true;
	//		float learningRate = 0.01f;
	//		int iterations = 10;
	//		float[][] data = getTrainingDataGD();
	//		TrainingData td = new TrainingData(data, 3);	
	//		TerminationCriteria tc = new TerminationCriteria(criteria);	
	//		tc.setIterations(iterations);
	//		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
	//		mlp.initiate();
	//		mlp.setWeights(getTestWeights());
	//		sut.setMLP(mlp);	
	//		sut.setTrainingData(td);	
	//		sut.setCostFunctionType(COST_FUNCTION_TYPE.SQUARED_ERROR);
	//		sut.setLearningRate(learningRate);
	//		sut.setTrainingTerminationCriteria(tc);
	//		/* Initial weights */
	//		float[][] expectedNodeGains = getExpectedIncreasedGains();
	//
	//		sut.initiateNodeGains();
	//		sut.trainOnSampleWithGainParameterWithoutGainMagnitudeModification(td.getInputRow(td.size()-5), td.getTargetRow(td.size()-5));
	//		/* sign change of some gains */
	//		sut.trainOnSampleWithGainParameter(td.getInputRow(1), td.getTargetRow(1));
	//
	//
	//		/*repeatition increase gain by 0.005*/
	//		float[][] actualNodeGains = sut.getNodeGains();
	//		for(int layerId = 0; layerId < expectedNodeGains.length;layerId++ ) {
	//			assertArrayEquals(expectedNodeGains[layerId],actualNodeGains[layerId]);
	//		}
	//	}

	private float[][] getExpectedIncreasedGains() {
		float[][] nodeGains = {
				{1.0040002f, 1.0040002f, 1.0040002f, -1.0040002f},
				{-1.0050002f, 1.0050002f, -1.0050002f}
		};
		return nodeGains;
	}

	private float[][] getTestWeights() {
		float[][] weights = new float[2][];
		//Initial LR:0.01
		weights[0] = new float[] {-0.1763008f, 0.07451278f, -0.26005787f, -0.1619301f, -0.13430117f, -0.34272027f, -0.31534073f,
				0.15900078f, 0.035133343f, 0.29067487f, -0.23169488f, 0.13239768f, -0.032672074f, -0.10106303f, 0.1268412f, 0.20811728f};
		weights[1] = new float[]{-0.24846007f, 0.13279961f, -0.07516521f, 0.24848956f, 0.23383175f, -0.12932675f, 0.15319659f, 0.17545138f,
				-0.1848386f, 0.20992821f, -0.14724563f, 0.32237983f, -0.059336144f, -0.13039368f, 0.09428874f};
		//After LR:-0.01
		return weights;
	}


	/**
	 * Test for calculating of node gradients within network, with delta rule
	 */
	@Test
	void testCalculateNetworkGradients() {
		int[] layerSizes = new int[] {3,4,3};
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);
		boolean useSoftmax = true;
		int trainingRowId = 0;
		float learningRate = 0.01f;
		COST_FUNCTION_TYPE costFType = COST_FUNCTION_TYPE.SQUARED_ERROR;
		ANNMLP mlp = new ANNMLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setCostFunctionType(costFType);
		sut.setLearningRate(learningRate);

		/* (Target - Predicted) per neuron */
		float[] differenceTarPred = sut.calculateDifferenceTargetSubPredicted(td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
		sut.calculateNetworkNodeGradients(costFType, td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));

		float[][] actualGradients = sut.getNetworkNodeGradients(); 
		/* index of node gradients for output layer, starts with zero because input layer omited */
		int outputLayerGradientsIdx = mlp.getLayerSizes().length -2;
		int outputNeuronLayerIdx = mlp.getLayerSizes().length -1;
		float[] io = mlp.getLayer(outputNeuronLayerIdx).getNetInputs();
		/* expectedGradients is one layer less than the size of mlp because the input layer is omitted */
		float[][] expectedGradients = new float[mlp.getLayerSizes().length-1][];
		float[] inputs;

		ACTIVATION_FUNCTION activationFType = ACTIVATION_FUNCTION.SOFTMAX;
		/* Produce error gradient for each output neuron */
		expectedGradients[outputLayerGradientsIdx] = new float[differenceTarPred.length];
		for(int neuronIdx = 0; neuronIdx < differenceTarPred.length; neuronIdx++ ) {
			/* Using default method which is previously implemented and tested */
			expectedGradients[outputLayerGradientsIdx][neuronIdx] = sut.calculateOutputNodeGradient(costFType, activationFType, differenceTarPred[neuronIdx], io, neuronIdx);
		}
		activationFType = ACTIVATION_FUNCTION.SIGMOID;
		/* Calculate node gradients for each hidden layers*/
		for(int gradientLayerIdx = expectedGradients.length-2; gradientLayerIdx >= 0 ; gradientLayerIdx--) {
			/* Retrieve inputs for the layer */
			inputs = mlp.getLayer(gradientLayerIdx+1).getNetInputs();
			/* initiate gradient array for layerIdx */
			expectedGradients[gradientLayerIdx] = new float[inputs.length];
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < expectedGradients[gradientLayerIdx].length; neuronIdx++ ) {
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				expectedGradients[gradientLayerIdx][neuronIdx] = sut.calculateNodeGradient(activationFType, expectedGradients[gradientLayerIdx+1],
						mlp.getLayer(gradientLayerIdx+1).getNeuron(neuronIdx).getWeightsAsArray(), inputs[neuronIdx]);
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
	void testSetGetMomentumDecayFactor() {
		float momentum = 0.90f;
		sut.setMomentumDecayFactor(momentum);
		float actual = sut.getMomentumDecayFactor();
		float expected = 0.90f;
		assertEquals(expected, actual);
	}

	/**
	 * test of get default learning momentum 
	 */
	@Test
	void testGetDefaultMomentumDecayFactor() {
		float actual = sut.getMomentumDecayFactor();
		float expected = 0.95f;
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

	/**
	 * Produce learning rate according to ADADELTA
	 * 
	 */
	@Test
	void testCalculateLearningRateADADELTA() {
		boolean useAdaptiveLEarningRate = true;
		float mSDeltaWeight = 0.004f;
		float mSGradient = 0.004f;
		int[] layerSizes = new int[] {3,4,3};
		ANNMLP mlp = new ANNMLP( layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);
		sut.setUseAdaptiveLearningRate(useAdaptiveLEarningRate);

		float actual = sut.calculateLearningRateADADELTA(mSDeltaWeight,mSGradient);
		float expected = (float) (Math.sqrt(mSDeltaWeight)/Math.sqrt(mSGradient));
		assertEquals(expected, actual);
	}

	/**
	 * Test implementation of learning on an example with ADADELTA learning rate adaptation. making
	 * 
	 */
	@Test
	void testCalculateWeightDeltaWithADADELTA() {
		int rowId = 1;
		int rowId2 = 2;
		float learningRateCorrector = 0.3f;
		WEIGHT_INITIATION_METHOD weightInitiationMethod = WEIGHT_INITIATION_METHOD.RANDOM;
		boolean useAdaptiveLEarningRate = true;
		boolean useSoftMaxTrue = true;
		float decayFactor = 0;
		int[] layerSizes = new int[] {3,4,3};
		TrainingData td = new TrainingData(getTrainingDataGD(),3);
		ANNMLP mlp = new ANNMLP(weightInitiationMethod, useSoftMaxTrue,layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);
		sut.initiateWeightDeltas();
		decayFactor = sut.getMomentumDecayFactor();
		/* get copy of the weights */
		float[][] expectedWeights = mlp.getWeights();
		/* initiate weightDeltas array */
		float[][] weightDeltas = new float[expectedWeights.length][];
		IntStream.range(0,weightDeltas.length).forEach(i -> 
		weightDeltas[i] = new float[expectedWeights[i].length]);
		/* Initial meanSquaredGradients are set to 0 */
		float[][] meanSquaredGradient = new float[mlp.getLayerSizes().length-1][];
		IntStream.range(0,meanSquaredGradient.length).forEach(i ->
		meanSquaredGradient[i] = new float[mlp.getLayerSizes()[i+1]]);	
		/* Initial meanSquaredWeightDeltas are set to 0 */
		float[][] meanSquaredDeltaWeight = new float[weightDeltas.length][];
		IntStream.range(0,meanSquaredDeltaWeight.length).forEach(i ->
		meanSquaredDeltaWeight[i] = new float[weightDeltas[i].length]);	
		float[][] gradients;
		float[] outputs;
		int neuronId = 0;
		int lowerNeuronId = 0;
		float nodeGradient = 0;
		float calculatedWeightDelta = 0;
		sut.setUseAdaptiveLearningRate(useAdaptiveLEarningRate);
		sut.setLearningRateCorrector(learningRateCorrector);
		sut.trainOnSampleWithADADELTA(td.getInputRow(rowId), td.getTargetRow(rowId));


		gradients = sut.getNetworkNodeGradients();
		if(gradients == null) {
			sut.calculateNetworkNodeGradients(sut.getCostFunctionType(), td.getInputRow(rowId), td.getTargetRow(rowId));
			gradients = sut.getNetworkNodeGradients();
		}
		boolean visited[];
		/* For each layer, top down*/
		for(int weightLayerIdx = expectedWeights.length-1; weightLayerIdx >= 0 ; weightLayerIdx--) {
			/* get outputs from layer lower, weightLayerIdx is one less than largest index of layers*/
			outputs = mlp.getLayer(weightLayerIdx).getOutputs();
			visited = new boolean[gradients[weightLayerIdx].length];
			/* for each weight in layer */
			for(int weightIdx = 0; weightIdx < expectedWeights[weightLayerIdx].length; weightIdx++) {

				/* idx of node gradient for upper layer, weightLayerIdx+1*/
				neuronId = weightIdx%mlp.getLayerSizes()[weightLayerIdx+1];
				/* get node gradient from neuron of upper layer */
				nodeGradient  = gradients[weightLayerIdx][neuronId];
				if(!visited[neuronId]) {
					visited[neuronId] = true;
					meanSquaredGradient[weightLayerIdx][neuronId] = 
							StatisticUtils.calculateMeanSqured(meanSquaredGradient[weightLayerIdx][neuronId], decayFactor, nodeGradient);
				}

				/* get index of neuron of connecting layer with current weightIdx, weightIdx/upperLayerSize */
				lowerNeuronId = weightIdx/mlp.getLayerSizes()[weightLayerIdx+1];
				/* currently biases are not included, must check for outofbound exception */
				if(lowerNeuronId <outputs.length ) {
					calculatedWeightDelta = nodeGradient * outputs[lowerNeuronId];
				}else {
					/* Bias */
					calculatedWeightDelta = nodeGradient;
				}

				/* is zero in the beginning */
				weightDeltas[weightLayerIdx][weightIdx] = decayFactor*weightDeltas[weightLayerIdx][weightIdx] - 
						sut.calculateLearningRateADADELTA(meanSquaredDeltaWeight[weightLayerIdx][weightIdx]
								, meanSquaredGradient[weightLayerIdx][neuronId])*learningRateCorrector * calculatedWeightDelta;
				expectedWeights[weightLayerIdx][weightIdx] = expectedWeights[weightLayerIdx][weightIdx] + weightDeltas[weightLayerIdx][weightIdx];	
				meanSquaredDeltaWeight[weightLayerIdx][weightIdx] =
						StatisticUtils.calculateMeanSqured(meanSquaredDeltaWeight[weightLayerIdx][weightIdx], decayFactor, calculatedWeightDelta);
			}
		}
		sut.trainOnSampleWithADADELTA(td.getInputRow(rowId2), td.getTargetRow(rowId2));
		gradients = sut.getNetworkNodeGradients();
		if(gradients == null) {
			sut.calculateNetworkNodeGradients(sut.getCostFunctionType(), td.getInputRow(rowId2), td.getTargetRow(rowId2));
			gradients = sut.getNetworkNodeGradients();
		}

		/* For each layer, top down*/
		for(int weightLayerIdx = expectedWeights.length-1; weightLayerIdx >= 0 ; weightLayerIdx--) {
			/* get outputs from layer lower, weightLayerIdx is one less than largest index of layers*/
			outputs = mlp.getLayer(weightLayerIdx).getOutputs();
			visited = new boolean[gradients[weightLayerIdx].length];
			/* for each weight in layer */
			for(int weightIdx = 0; weightIdx < expectedWeights[weightLayerIdx].length; weightIdx++) {

				/* idx of node gradient for upper layer, weightLayerIdx+1*/
				neuronId = weightIdx%mlp.getLayerSizes()[weightLayerIdx+1];
				/* get node gradient from neuron of upper layer */
				nodeGradient  = gradients[weightLayerIdx][neuronId];
				if(!visited[neuronId]) {
					visited[neuronId] = true;
					meanSquaredGradient[weightLayerIdx][neuronId] = 
							StatisticUtils.calculateMeanSqured(meanSquaredGradient[weightLayerIdx][neuronId], decayFactor, nodeGradient);
				}

				/* get index of neuron of connecting layer with current weightIdx, weightIdx/upperLayerSize */
				lowerNeuronId = weightIdx/mlp.getLayerSizes()[weightLayerIdx+1];
				/* currently biases are not included, must check for outofbound exception */
				if(lowerNeuronId <outputs.length ) {
					calculatedWeightDelta = nodeGradient * outputs[lowerNeuronId];
				}else {
					/* Bias */
					calculatedWeightDelta = nodeGradient;
				}

				/* is zero in the beginning */
				weightDeltas[weightLayerIdx][weightIdx] = decayFactor*weightDeltas[weightLayerIdx][weightIdx] - 
						sut.calculateLearningRateADADELTA(meanSquaredDeltaWeight[weightLayerIdx][weightIdx]
								, meanSquaredGradient[weightLayerIdx][neuronId])*learningRateCorrector * calculatedWeightDelta;
				expectedWeights[weightLayerIdx][weightIdx] = expectedWeights[weightLayerIdx][weightIdx] + weightDeltas[weightLayerIdx][weightIdx];	
				meanSquaredDeltaWeight[weightLayerIdx][weightIdx] =
						StatisticUtils.calculateMeanSqured(meanSquaredDeltaWeight[weightLayerIdx][weightIdx], decayFactor, calculatedWeightDelta);
			}
		}

		float[][] actualWeight = mlp.getWeights();
		for (int i = 0; i < actualWeight.length; i++) {
			assertArrayEquals(expectedWeights[i],actualWeight[i]);
		}
	}


	/**
	 * Should return two dimensional array, one entity per neuron with 1th as initial value
	 */
	@Test
	void testInitiationNodeGains() {
		int[] layerSizes = new int[] {3,4,3};
		ANNMLP mlp = new ANNMLP( layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);		
		float[][] nodeGains = sut.initiateNodeGains();
		/* one layer less than number of network layer */
		float[][] expected = new float[layerSizes.length-1][];
		for(int layerId = 0; layerId < expected.length; layerId++) {
			/* skipping the input Layer */
			expected[layerId] = new float[layerSizes[layerId+1]];
			for(int i = 0; i < expected[layerId].length;i++) {
				expected[layerId][i] = 1;
			}
		}
		for(int layerId = 0; layerId < expected.length; layerId++) {
			assertArrayEquals(expected[layerId],nodeGains[layerId]);
		}
	}


	/**
	 * Test on update of node gain , decrease function
	 */
	@Test
	void testUpdateDecreaseGainFunction() {
		float[][] values = new float[2][];
		values[0] = new float[] {1f,1f};
		values[1] = new float[] {1f,1f};
		int layerIdx = 0,neuronIdx = 0;
		float gainMagnitudeMultiplier = 0.5f;
		sut.updateNodeGainsDecreaseMagnitude(values,layerIdx, neuronIdx, gainMagnitudeMultiplier);
		float actual = values[layerIdx][neuronIdx];
		float expected = 0.5f;
		assertEquals(expected,actual);
	}

	/**
	 * Test on update of node gain , decrease function negative values
	 */
	@Test
	void testUpdateDecreaseGainFunctionNegativeValue() {
		float[][] values = new float[2][];
		values[0] = new float[] {-1f,1f};
		values[1] = new float[] {1f,1f};
		int layerIdx = 0,neuronIdx = 0;
		float gainMagnitudeMultiplier = 0.5f;
		sut.updateNodeGainsDecreaseMagnitude(values,layerIdx, neuronIdx, gainMagnitudeMultiplier);
		float actual = values[layerIdx][neuronIdx];
		float expected = -0.5f;
		assertEquals(expected,actual);
	}

	/**
	 * Test on update of node gain , Increase function 
	 */
	@Test
	void testUpdateIncreaseGainFunctionNegativeValue() {
		float[][] values = new float[2][];
		values[0] = new float[] {-1f,1f};
		values[1] = new float[] {1f,1f};
		int layerIdx = 0,neuronIdx = 0;
		float gainMagnitudeIncrement = 0.005f;
		sut.updateNodeGainsIncreaseMagnitude(values,layerIdx, neuronIdx, gainMagnitudeIncrement);
		float actual = values[layerIdx][neuronIdx];
		float expected = -1.005f;
		assertEquals(expected,actual);
	}

	/**
	 * Test on update of node gain , Increase function positive value
	 */
	@Test
	void testUpdateIncreaseGainFunctionPositiveValue() {
		float[][] values = new float[2][];
		values[0] = new float[] {1f,1f};
		values[1] = new float[] {1f,1f};
		int layerIdx = 0,neuronIdx = 0;
		float gainMagnitudeIncrement = 0.005f;
		sut.updateNodeGainsIncreaseMagnitude(values,layerIdx, neuronIdx, gainMagnitudeIncrement);
		float actual = values[layerIdx][neuronIdx];
		float expected = 1.005f;
		assertEquals(expected,actual);
	}


	/**
	 * Test train() function with MAX iterations as stopping criteria. Number of iterations 2
	 */ 
	@Test
	void testTrainWithStoppingCriteriaMaxIterations() {
		int inputTargetDemarcation = 3;
		TrainingData td = new TrainingData(getTrainingDataGD(), inputTargetDemarcation);
		int[] layerSizes = new int[] {3,30,3};
		int maxIterations = 100;
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		boolean useSoftmax = true;
		WEIGHT_INITIATION_METHOD weightInitiationMethod = WEIGHT_INITIATION_METHOD.RANDOM;
		TerminationCriteria tc = new TerminationCriteria(criteria,maxIterations);
		/* Auxiliary GradientDescent  */ 
		GradientDescent gd = new GradientDescent();
		ANNMLP mlp = new ANNMLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlp.setTrainingTerminationCriteria(tc);
		mlp.initiate();

		ANNMLP mlpA = new ANNMLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlpA.setTrainingTerminationCriteria(tc);
		mlpA.initiate();
		mlpA.setWeights(mlp.getWeights());

		gd.setTrainingData(td);
		gd.setMLP(mlp);
		gd.setTrainingTerminationCriteria(tc);
		sut.setTrainingData(td);
		sut.setMLP(mlpA);	
		sut.setTrainingTerminationCriteria(tc);

		sut.train();

		int rowId = 0;
		for(int iteration = 0; iteration < maxIterations;iteration++) {
			gd.trainOnSampleWithGainParameterWithoutGainMagnitudeModificationWithDelta(td.getInputRow(rowId),td.getTargetRow(rowId));
			for(int row = 0;row < td.size(); row++) {
				gd.trainOnSampleWithGainParameterWithDeltaRule(td.getInputRow(row),td.getTargetRow(row));
			}
		}
		float[][] expectedWeights = mlp.getWeights();
		float[][] actualWeights = mlpA.getWeights();
		for (int i = 0; i < actualWeights.length; i++) {
			assertArrayEquals(expectedWeights[i],actualWeights[i]);
		}		
	}

	/**
	 * Test train() function with MAX iterations as stopping criteria. Number of iterations 2
	 */ 
	@Test
	void testTrainWithStoppingCriteriaMaxIterationsWithMomentum() {
		int inputTargetDemarcation = 3;
		TrainingData td = new TrainingData(getTrainingDataGD(), inputTargetDemarcation);
		TrainingData td1 = new TrainingData(getTrainingDataGD(), inputTargetDemarcation);
		int[] layerSizes = new int[] {3,30,3};
		int maxIterations = 100;
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		boolean useSoftmax = true;
		WEIGHT_INITIATION_METHOD weightInitiationMethod = WEIGHT_INITIATION_METHOD.RANDOM;
		TerminationCriteria tc = new TerminationCriteria(criteria,maxIterations);
		/* Auxiliary GradientDescent  */ 
		GradientDescent gd = new GradientDescent();
		ANNMLP mlp = new ANNMLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlp.setTrainingTerminationCriteria(tc);
		mlp.initiate();

		ANNMLP mlpA = new ANNMLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlpA.setTrainingTerminationCriteria(tc);
		mlpA.initiate();
		mlpA.setWeights(mlp.getWeights());

		gd.setTrainingData(td);
		gd.setMLP(mlp);
		gd.setTrainingTerminationCriteria(tc);
		sut.setTrainingData(td1);
		sut.setMLP(mlpA);	
		sut.setTrainingTerminationCriteria(tc);

		sut.trainWithMomentum();

		for(int iteration = 0; iteration < maxIterations;iteration++) {
			for(int row = 0;row < td.size(); row++) {
				gd.trainOnSampleWithMomentum(td.getInputRow(row),td.getTargetRow(row));
			}			
		}

		float[][] expectedWeights = mlp.getWeights();
		float[][] actualWeights = mlpA.getWeights();
		for (int i = 0; i < actualWeights.length; i++) {
			assertArrayEquals(expectedWeights[i],actualWeights[i]);
		}		
	}

	/**
	 * Test train() function with MAX iterations as stopping criteria. Number of iterations 2
	 */ 
	@Test
	void testTrainWithOneSampleWithMomentum() {
		int[] layerSizes = new int[] {3,4,3};
		float[][] data = getTrainingDataGD();
		TrainingData td = new TrainingData(data, 3);
		boolean useSoftmax = true;

		float learningRate = 0.01f;
		ANNMLP mlp = new ANNMLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);	
		sut.setTrainingData(td);	
		sut.setLearningRate(learningRate);

		float momentumDecayFactor = sut.getMomentumDecayFactor();

		int neuronId = 0;
		int lowerNeuronId = 0;
		float calculatedWeightDelta = 0;
		float[] outputs;
		float nodeGradient; 

		int rowId = 0;
		float[][] expectedWeights = mlp.getWeights();
		float[][] weightDeltas = new float[expectedWeights.length][];
		for (int i = 0; i < weightDeltas.length; i++) {
			weightDeltas[i] = new float[expectedWeights[i].length];
		}

		sut.trainOnSampleWithMomentum(td.getInputRow(rowId), td.getTargetRow(rowId));
		float[][] gradients = sut.getNetworkNodeGradients(); 		

		/* For each layer, top down*/
		for(int weightLayerIdx = expectedWeights.length-1; weightLayerIdx >= 0 ; weightLayerIdx--) {
			/* get outputs from layer lower, weightLayerIdx is one less than largest index of layers*/
			outputs = mlp.getLayer(weightLayerIdx).getOutputs();
			/* for each weight in layer */
			for(int weightIdx = 0; weightIdx < expectedWeights[weightLayerIdx].length; weightIdx++) {

				/* idx of node gradient for upper layer, weightLayerIdx+1*/
				neuronId = weightIdx%mlp.getLayerSizes()[weightLayerIdx+1];
				/* get node gradient from neuron of upper layer */
				nodeGradient  = gradients[weightLayerIdx][neuronId];
				/* get index of neuron of connecting layer with current weightIdx, weightIdx/upperLayerSize */
				lowerNeuronId = weightIdx/mlp.getLayerSizes()[weightLayerIdx+1];
				/* currently biases are not included, must check for outofbound exception */
				if(lowerNeuronId <outputs.length ) {
					calculatedWeightDelta = nodeGradient * outputs[lowerNeuronId];
				}else {
					/* Bias */
					calculatedWeightDelta = nodeGradient;
				}
				/* is zero in the beginning */
				weightDeltas[weightLayerIdx][weightIdx] = momentumDecayFactor*weightDeltas[weightLayerIdx][weightIdx] -learningRate*calculatedWeightDelta;
				expectedWeights[weightLayerIdx][weightIdx] = expectedWeights[weightLayerIdx][weightIdx] + weightDeltas[weightLayerIdx][weightIdx];	
			}
		}

		sut.trainOnSampleWithMomentum(td.getInputRow(rowId), td.getTargetRow(rowId));

		gradients = sut.getNetworkNodeGradients();  

		/* For each layer, top down*/
		for(int weightLayerIdx = expectedWeights.length-1; weightLayerIdx >= 0 ; weightLayerIdx--) {
			/* get outputs from layer lower */
			outputs = mlp.getLayer(weightLayerIdx).getOutputs();
			/* for each weight in layer */
			for(int weightIdx = 0; weightIdx < expectedWeights[weightLayerIdx].length; weightIdx++) {

				/* idx of node gradient for upper layer, weightLayerIdx+1*/
				neuronId = weightIdx%mlp.getLayerSizes()[weightLayerIdx+1];
				/* get node gradient from neuron of upper layer */
				nodeGradient  = gradients[weightLayerIdx][neuronId];
				/* get index of neuron of connecting layer with current weightIdx, weightIdx/upperLayerSize */
				lowerNeuronId = weightIdx/mlp.getLayerSizes()[weightLayerIdx+1];
				/* currently biases are not included, must check for outofbound exception */
				if(lowerNeuronId <outputs.length ) {
					calculatedWeightDelta = nodeGradient * outputs[lowerNeuronId];
				}else {
					/* Bias */
					calculatedWeightDelta = nodeGradient;
				}
				weightDeltas[weightLayerIdx][weightIdx] = momentumDecayFactor*weightDeltas[weightLayerIdx][weightIdx] - learningRate* calculatedWeightDelta;
				expectedWeights[weightLayerIdx][weightIdx] = expectedWeights[weightLayerIdx][weightIdx] + weightDeltas[weightLayerIdx][weightIdx];	
			}
		}

		float[][] actualWeights = mlp.getWeights();
		for (int i = 0; i < actualWeights.length; i++) {
			assertArrayEquals(expectedWeights[i],actualWeights[i]);
		}		
	}

	/**
	 * Test implementation of learning on an example with ADADELTA learning rate adaptation. making
	 * 
	 */
	@Test
	void testCalculateWeightDeltaWithADADELTAWithStoppingCriteria() {
		int inputTargetDemarcation = 3;
		TrainingData td = new TrainingData(getTrainingDataGD(), inputTargetDemarcation);
		int[] layerSizes = new int[] {3,30,3};
		int maxIterations = 30;
		float learningRateCorrector = 0.3f;
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		boolean useSoftmax = true;
		WEIGHT_INITIATION_METHOD weightInitiationMethod = WEIGHT_INITIATION_METHOD.RANDOM;
		TerminationCriteria tc = new TerminationCriteria(criteria,maxIterations);
		/* Auxiliary GradientDescent  */ 
		GradientDescent gd = new GradientDescent();
		ANNMLP mlp = new ANNMLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlp.setTrainingTerminationCriteria(tc);
		mlp.initiate();

		ANNMLP mlpA = new ANNMLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlpA.setTrainingTerminationCriteria(tc);
		mlpA.initiate();
		mlpA.setWeights(mlp.getWeights());

		gd.setTrainingData(td);
		gd.setMLP(mlp);
		gd.setTrainingTerminationCriteria(tc);
		gd.setLearningRateCorrector(learningRateCorrector);
		
		sut.setTrainingData(td);
		sut.setMLP(mlpA);	
		sut.setTrainingTerminationCriteria(tc);
		
		sut.setLearningRateCorrector(learningRateCorrector);
		sut.trainADADELTA();

		for(int iteration = 0; iteration < maxIterations;iteration++) {
			for(int row = 0;row < td.size(); row++) {
				gd.trainOnSampleWithADADELTA(td.getInputRow(row),td.getTargetRow(row));
			}
		}

		float[][] expectedWeights = mlp.getWeights();
		float[][] actualWeights = mlpA.getWeights();
		for (int i = 0; i < actualWeights.length; i++) {
			assertArrayEquals(expectedWeights[i],actualWeights[i]);
		}	
	}
	
	/**
	 * Test function epochRSSDelta- change of RSS
	 * RSS - residual sum of squares
	 * Delta = epoch_i+1(x) - epoch_i(x)  
	 * epoch_i = MSE(Cost(X))
	 * X - cross validation test set, set from training data
	 */
	@Test
	void testCalculateEpochRSSDelta() {
		TERMINATION_CRITERIA[] criteria = {TERMINATION_CRITERIA.MAX_ITERATIONS};
		WEIGHT_INITIATION_METHOD weightInitiationMethod = WEIGHT_INITIATION_METHOD.RANDOM;
		int inputTargetDemarcation = 3;
		int[] layerSizes = new int[] {3,30,3};
		int maxIterations = 30;
		float learningRateCorrector = 0.3f;
		boolean useSoftmax = true;
	
		TrainingData td = new TrainingData(getTrainingDataGD(), inputTargetDemarcation);		
		
		TerminationCriteria tc = new TerminationCriteria(criteria,maxIterations);
		/* Auxiliary GradientDescent  */ 
		ANNMLP mlp = new ANNMLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlp.setTrainingTerminationCriteria(tc);
		mlp.initiate();
		
		sut.setTrainingData(td);
		sut.setMLP(mlp);	
		sut.setTrainingTerminationCriteria(tc);		
		sut.setLearningRateCorrector(learningRateCorrector);
		sut.trainADADELTA();
		float expected = sut.calculateTotalMSE();
		sut.trainADADELTA();
		expected = Math.abs(expected - sut.calculateTotalMSE());
		float actual = sut.calculateEpochRSSDelta();
		assertEquals(expected,actual);		
	}

}