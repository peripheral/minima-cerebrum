package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
	 * Tests performs test on error calculation from target and observed
	 * Error = (target - observed)
	 */
	@Test
	void testCalculateErrorPerNeuron() {
		int[] layerSizes = {3,4,3};
		WEIGHT_INITIATION_METHOD weightInitiationMethod = WEIGHT_INITIATION_METHOD.RANDOM;
		ANN_MLP mlp = new ANN_MLP(weightInitiationMethod, layerSizes);
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
	 * of the output neuron
	 * Io - neuron input, E = (Predicted - Required)
	 * ∂(E)^2/∂Io = gradient
	 * (∂E^2/∂Io) => 2E  - first step of derivation
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2E * fo'(Io) = gradient
	 */
	@Test
	void testCalculateNodeGradientOutputNeurons() {
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
	void testCalculationNodeGradientOutputNeuronDeltaRule() {
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
		float actual = sut.calculateNodeGradientDeltaRule(costF, activationFunction,  diffTarPred,IoAll,neuronId);
		assertEquals(expected,actual);
	}
	
	/**
	 * Test for function that calculates gradient of input to error in hidden neurons according to delta rule
	 * Io - neuron input, E = (Required - Predicted)^2
	 * Oo - output of output neuron, 
	 * Io - input of output neuron
	 * ∂(E)^2/∂Ih = ∂(target - observed)^2/∂Oo * -fo'(Io)/∂Io * ∂Io/∂Oh * ∂Oh/∂Ih  = outputGradient
	 * ∂(target - observed)^2/∂Oo * -fo'(Io)/∂Io => 2(target - observed) * -fo'(Io) 
	 * ∂Io/∂Oh => ∂(OiWoi+ Oi+1Woi+1 +OhWoh+..)/∂Oh => Woh
	 * ∂Oh/∂Ih => f'(Ih), f(.) - sigmoid
	 * outputGradient * Woh * f'(Ih) = gradient
	 */
	@Test
	void testCalculationNodeGradientHiddenNeuronDeltaRule() {
		ACTIVATION_FUNCTION activationFunction = ACTIVATION_FUNCTION.SIGMOID;
		float Woh = 0.03f;
		float Ih = 6f;
		float outputGradient = -0.10115465582f;
		/* 2(target - observed) * -fo'(Io) = gradient */
		float expected =  outputGradient * Woh * NeuronFunctionModels.derivativeOf(activationFunction, Ih);
		float actual = sut.calculateNodeGradientDeltaRule(activationFunction,  outputGradient,Woh,Ih);
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
	void testCalculationNodeGradientHiddenDeltaRule() {
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
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
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
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM, useSoftmax, layerSizes);
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

	/**
	 * Test learning with adaptive learningRate, halve the gain once the neuron gain changes signum
	 * test MLP consists of input layer size 3, hidden layer 4 neurons, 3 output neurons
	 * hidden layer activation function sigmoid, output activation softmax
	 */
	@Test
	void testIncreaseNodeLearningGainAtStepRepetion() {
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
		sut.setTrainingTerminationCriteria(tc);
		/* Initial weights */
		float[][] expectedNodeGains = getExpectedIncreasedGains();

		sut.initiateNodeGains();
		sut.trainOnSampleWithGainParameterWithoutGainMagnitudeModification(td.getInputRow(td.size()-5), td.getTargetRow(td.size()-5));
		/* sign change of some gains */
		sut.trainOnSampleWithGainParameter(td.getInputRow(1), td.getTargetRow(1));


		/*repeatition increase gain by 0.005*/
		float[][] actualNodeGains = sut.getNodeGains();
		for(int layerId = 0; layerId < expectedNodeGains.length;layerId++ ) {
			assertArrayEquals(expectedNodeGains[layerId],actualNodeGains[layerId]);
		}
	}

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
				expectedGradients[layerIdx][neuronIdx] = sut.calculateNodeGradient(activationFType,
						expectedGradients[layerIdx+1], mlp.getLayer(layerIdx).getNeuron(neuronIdx).getWeightsAsArray(), inputs[neuronIdx]);
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
		sut.setCostFunctionType(costFType);
		sut.setLearningRate(learningRate);

		float[] errorVector = sut.calculateErrorPerNeuron(td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
		sut.calculateNetworkNodeGradientsStoredLocaly(costFType, td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));

		float[][] actualGradients = sut.getLocallyStoredNetworkNodeGradients(); 
		/* index of node gradients for output layer, starts with zero because input layer omited */
		int outputLayerGradientsIdx = mlp.getLayerSizes().length -2;
		int outputNeuronLayerIdx = mlp.getLayerSizes().length -1;
		float[] io = mlp.getLayer(outputNeuronLayerIdx).getNetInputs();
		/* expectedGradients is one layer less than the size of mlp because the input layer is omitted */
		float[][] expectedGradients = new float[mlp.getLayerSizes().length-1][];
		float[] inputs;

		ACTIVATION_FUNCTION activationFType = ACTIVATION_FUNCTION.SOFTMAX;
		/* Produce error gradient for each output neuron */
		expectedGradients[outputLayerGradientsIdx] = new float[errorVector.length];
		for(int errIdx = 0; errIdx < errorVector.length; errIdx++ ) {
			/* Using default method which is previously implemented and tested */
			expectedGradients[outputLayerGradientsIdx][errIdx] = sut.calculateNodeGradient(costFType, activationFType, errorVector[errIdx], io, errIdx);
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
	
	/**
	 * Test for calculating of node gradients within network, with delta rule
	 */
	@Test
	void testCalculateNodeGradientsWithinNetworkLocallyStoredWithDeltaRule() {
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
		sut.setCostFunctionType(costFType);
		sut.setLearningRate(learningRate);

		/* (Target - Predicted) per neuron */
		float[] differenceTarPred = sut.calculateDifferenceTargetSubPredicted(td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));
		sut.calculateNetworkNodeGradientsStoredLocalyWithDeltaRule(costFType, td.getInputRow(trainingRowId),td.getTargetRow(trainingRowId));

		float[][] actualGradients = sut.getLocallyStoredNetworkNodeGradients(); 
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
			expectedGradients[outputLayerGradientsIdx][neuronIdx] = sut.calculateNodeGradientDeltaRule(costFType, activationFType, differenceTarPred[neuronIdx], io, neuronIdx);
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
	 * test of calculate delta weight with momentum
	 * expected = decay * oldDeltaWeight - newDeltaWeight
	 */
	@Test
	void testCalculateDeltaWeightWithMomentum() {
		float decay = 0.95f;
		float oldDeltaWeight = 0.04f;
		float newDeltaWeight = 0.01f;
		float actual = sut.calculateDeltaWeightWithMomentum(decay,oldDeltaWeight,newDeltaWeight);
		float expected = decay * oldDeltaWeight - newDeltaWeight ;
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
				nodeGradients[layerIdx][neuronIdx] = sut.calculateNodeGradient(activationFType, nodeGradients[layerIdx+1], weights[layerIdx],  inputs[neuronIdx]);
			}
		}
		return nodeGradients;		
	}

	/**
	 * Should return two dimensional array, one entity per neuron with 1th as initial value
	 */
	@Test
	void testInitiationNodeGains() {
		int[] layerSizes = new int[] {3,4,3};
		ANN_MLP mlp = new ANN_MLP( layerSizes);
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
		ANN_MLP mlp = new ANN_MLP(weightInitiationMethod, useSoftmax, layerSizes);
		mlp.setTrainingTerminationCriteria(tc);
		mlp.initiate();
		
		ANN_MLP mlpA = new ANN_MLP(weightInitiationMethod, useSoftmax, layerSizes);
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
	
	
}