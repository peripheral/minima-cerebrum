package mlp.trainer.gradient_descent;

import java.util.HashSet;

import math.utils.StatisticUtils;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.NeuronFunctionModels;
import mlp.NeuronLayer;
import mlp.trainer.Backpropagation;
import mlp.trainer.TerminationCriteria;
import mlp.trainer.TerminationCriteria.TERMINATION_CRITERIA;

public class GradientDescent extends Backpropagation {

	/**
	 * Default momentum
	 */
	private float momentum = 0.00001f;
	private float oldGradient = 0f;
	private TerminationCriteria terminationCriteria = new TerminationCriteria();
	private float[][] nodeGradients = null;
	private float[][] nodeGains = null;
	private float gainReductionFactor = 0.5f;
	private float gainMagnitudeIncrement = 0.001f;
	private float[][] weightDeltas = null;
	private float momentDecayFactor = 0.95f;


	/**
	 * Function calculates gradient from error and given input and activation of layer function parameters
	 * @param costFType -type of cost function 
	 * @param activationFType - type of fo(.)
	 * @param error - (predicted - required )
	 * @param io - net input of the output neuron
	 * @param a
	 * @param b
	 * 2E * (- fo'(Io)) = gradient
	 * @return
	 */
	public float calculateNodeGradient(COST_FUNCTION_TYPE costFType, ACTIVATION_FUNCTION activationFType,
			float error, float[] io, int neuronIdx, float a, float b) {
		float partialDerivative = 0;
		switch(activationFType) {
		case SOFTMAX:
			partialDerivative = StatisticUtils.calculateSoftmaxPartialDerivative(io, neuronIdx);
			break;
		default:
			System.err.println("Derivative of "+activationFType+" not implemented");
			break;
		}
		float result = 2 * error * -partialDerivative;
		return result;
	}

	/**
	 * Function calculates gradient from error and given input and activation of layer function parameters.
	 * paramaters a and b = 1; Main function calculation of node gradients of output layer neurons
	 * @param costFType -type of cost function 
	 * @param activationFType - type of fo(.)
	 * @param error - (predicted - required )
	 * @param io - net input of the output neurons
	 * @param neuronIdx - index of the neuron for which the gradient is calculated, used for softmax 
	 * 2E * fo'(Io) = gradient
	 * @return
	 */
	public float calculateNodeGradient(COST_FUNCTION_TYPE costFType, ACTIVATION_FUNCTION activationFType,
			float error, float[] io, int neuronIdx) {
		float partialDerivative = 0;
		float result = 0;
		switch(costFType) {

		case SQUARED_ERROR:
			switch(activationFType) {
			case SOFTMAX:
				partialDerivative = StatisticUtils.calculateSoftmaxPartialDerivative(io, neuronIdx);
				result = 2 * error * -partialDerivative;
				break;
			default:
				System.err.println("Derivative of "+activationFType+" not implemented");
				break;
			}
			break;
		default:
			System.err.println("Cost function "+costFType+" not implemented");
			break;
		}	
		return result;
	}

	/**
	 * Function calculates gradient from error and given input and activation of layer
	 * function parameters parameter a = 1 and b = 1
	 * @param costFType -type of cost function 
	 * @param activationFType - type of fo(.)
	 * @param error - (predicted - required )
	 * @param io - net input of the output neuron
	 * 2E * fo'(Io) = gradient
	 * @return
	 */
	public float calculateGradientInputOverError(COST_FUNCTION_TYPE costFType, ACTIVATION_FUNCTION activationFType,
			float error, float[] io, int neuronIdx) {

		float partialDerivative = 0;
		switch(activationFType) {
		case SOFTMAX:
			partialDerivative = StatisticUtils.calculateSoftmaxPartialDerivative(io, neuronIdx);
			break;
		default:
			System.err.println("Derivative of "+activationFType+" not implemented");
			break;
		}
		float result = 2 * error * partialDerivative;
		return result;
	}



	/**
	 * 	 * Function calculates gradient for following(Hidden) layer gradients by using gradient of output layer.
	 * Gradient_l-1 = Gradient * Who * fsig'(Ih)
	 * @param activFunction - type of activation function
	 * @param gradient - gradient of upper layer
	 * @param Ih - netinput to neuron h
	 * @param Who - weight from neuron H to Neuron O
	 * @param a
	 * @param b
	 * @return
	 */
	public float calculateNodeGradient(ACTIVATION_FUNCTION activFunction, float gradient,float Ih, float Who, float a, float b) {
		float derivativeActivationFunction = 0;
		switch(activFunction) {
		case SIGMOID:
			derivativeActivationFunction = NeuronFunctionModels.derivativeOf(activFunction, a, b, Ih);
			break;		
		default: 
			System.err.println("Activation function:"+activFunction+" not supported.");
			break;
		}
		return gradient * Who * derivativeActivationFunction;
	}

	/**
	 * Function calculates gradient for following(Hidden) layer gradients by using gradient of output layer.
	 * param a = 1, param b = 1
	 * Gradient_l-1 = Gradient * Who * fsig'(Ih)
	 * @param activFunction - type of activation function
	 * @param gradient - gradient of upper layer
	 * @param Ih - netinput to neuron h
	 * @param Who - weight from neuron H to Neuron O
	 * @return
	 */
	public float calculateGradientInputOverError(ACTIVATION_FUNCTION activFunction, float gradient,float Ih, float Who) {
		float derivativeActivationFunction = 0;
		float a = 1,b = 1;
		switch(activFunction) {
		case SIGMOID:
			derivativeActivationFunction = NeuronFunctionModels.derivativeOf(activFunction, a, b, Ih);
			break;		
		default: 
			System.err.println("Activation function:"+activFunction+" not supported.");
			break;
		}
		return gradient * Who * derivativeActivationFunction;
	}

	public float calculateDelta(COST_FUNCTION_TYPE squaredError, float error, float io, float a, float b) {
		// TODO Auto-generated method stub
		return 0;
	}

	public float calculateGradientInputOverError(float gradient, float ih, float a, float b) {
		// TODO Auto-generated method stub
		return 0;
	}


	public float calculateDelta(float gradient, float[] ih, int neuronIdx) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Produces delta weight from gradient and the sending neuron output
	 * ∂(E)^2/∂Ih = (∂E^2/∂Io) * (∂Io/∂Oh) * (∂Oh/∂Ih) = gradient
	 * ∂(E)^2/∂Who = (∂E^2/∂Io) * (∂Io/∂Oh) * (∂Oh/∂Ih) * (∂Ih/∂Who)
	 * (∂Ih/∂Who) => ∂(OpWpo+Op+1Wp+1o+ ..+OhWho) = Oh
	 *  delta = gradient * Oh
	 * @param gradient - gradient of neuron
	 * @param Oh - output of neuron h
	 * @return - gradient * Oh
	 */
	public float calculateDeltaWeight(float gradient, float Oh) {
		return gradient * Oh;
	}

	/**
	 * Calculates new weight
	 * @param nodeGradient -  ∂(E)^2/∂Who
	 * @param oldNodeGradient - gradient previously used weight to calc
	 * @param learningRate - learning rate, factor decreases
	 * @param momentum - to help to progress through low gradient
	 * @param currentWeight - initial weight of incoming connection to the node with nodeGradient
	 * @return currentWeight + learningRate * nodeGradient + momentum * oldNodeGradient
	 */
	public float calculateWeight(float nodeGradient, float oldNodeGRadient, float learningRate, float momentum,
			float currentWeight) {
		return currentWeight + learningRate * nodeGradient + (momentum * oldNodeGRadient);
	}

	/**
	 * Calculates new weight
	 * @param deltaWeight -  ∂(E)^2/∂Who
	 * @param oldDeltaWeight - gradient previously used weight to calc
	 * @param learningRate - learning rate, factor decreases
	 * @param momentum - to help to progress through low gradient
	 * @param currentWeight - initial weight of incoming connection to the node with nodeGradient
	 * @param nodeGain - per node learning rate modifier
	 * @return currentWeight + learningRate * nodeGradient + momentum * oldNodeGradient
	 */
	public float calculateWeight(float deltaWeight, float oldDeltaWeight, float learningRate, float momentum,
			float currentWeight, float nodeGain) {
		return (currentWeight + learningRate * deltaWeight * nodeGain + (momentum * oldDeltaWeight))%7;
	}

	public void trainOnSample(float[] inputRow, float[] targetRow) {

		/* Calculate gradients per weight layer */
		calculateNetworkNodeGradients(costFunctionType, inputRow, targetRow);

		float nodeGradient = 0;
		float newWeight = 0;
		float currentWeight ;
		/* Calculate weights per layer */
		for(int layerIdx = mlp.getLayerSizes().length-1; layerIdx > 0 ; layerIdx--) {
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(layerIdx).size(); neuronIdx++) {
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient = mlp.getNodeGradient(layerIdx,neuronIdx);
				if(nodeGradient > 0 && learningRate > 0) {
					learningRate = learningRate * -1;
				}else if(nodeGradient < 0 && learningRate < 0){
					learningRate = learningRate * -1;
				}
				for(int weightIdx = 0; weightIdx < mlp.getLayer(layerIdx-1).getWeights().length;weightIdx+=mlp.getLayerSizes()[layerIdx] ) {
					currentWeight = mlp.getLayer(layerIdx-1).getWeights()[weightIdx+neuronIdx];

					newWeight = calculateWeight(nodeGradient, oldGradient, learningRate, momentum, currentWeight);

					mlp.getLayer(layerIdx-1).setWeight(weightIdx+neuronIdx,newWeight);
					if(layerIdx == 0 && neuronIdx == 0) {
						System.out.println("Impl Layer id:"+layerIdx+" Neuron id:"+neuronIdx+" New weight:"+mlp.getLayer(layerIdx-1).getWeights()[weightIdx+neuronIdx]+
								" Node grad:"+nodeGradient+" Old grad:"+oldGradient+" LearnR:"+learningRate);
					}
				}				

			}
		}

	}
	/**
	 * Default momentum = 0.00001f;
	 * @return
	 */
	public float getMomentum() {
		return momentum;
	}

	public void setMomentum(float momentum) {
		this.momentum = momentum;

	}

	public float[] calculateNodeGradient(ACTIVATION_FUNCTION activationFunction, float[] outputNodeGradients, float ih,
			float[] who, float a, float b) {
		// TODO Auto-generated method stub
		return null;
	}

	/** Io - neuron input, E = (Required - Predicted)
	 * ∂(E)^2/∂Io = ∂(E)^2/∂Io * ∂(E)/∂Io * ∂fo(Io)/∂Io = gradient
	 * (∂E^2/∂Io) => 2E  - first step of derivation
	 * ∂(Required - Predicted)/∂Io => -Oo => -fo(.)
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2E * -fo'(Io) = gradient
	 * hidden node gradient = f'(Ih)*(Whi* Gradrient_i + Whi+1* Gradrient_i+1 .. + )
	 * Ih - input to node, Whi - weight from node h to node i, Gradient_i - gradient for neuron 
	 * for node i 
	 * 
	 * @param activationFunctionType - node activation method
	 * @param outputNodeGradients - gradients from upper layer
	 * @param Ih - input for the node
	 * @param who - outgoing weights from the node
	 * @return
	 */
	public float calculateNodeGradient(ACTIVATION_FUNCTION activationFunctionType, float[] outputNodeGradients,
			float[] who, float Ih) {
		float a = 1;
		float b = 1;
		float gradient = 0;
		switch(activationFunctionType) {
		case SOFTMAX:
			//	gradient = StatisticUtils.calculateSoftmaxPartialDerivative(data, idx)
			System.err.println("Error SOFTMAX is not implemented");
			break;
		case SIGMOID:
			gradient =  NeuronFunctionModels.derivativeOf(activationFunctionType, a, b,Ih);
			break;
		default:
			System.err.println("Activation function not implemented:"+activationFunctionType);
			break;
		}

		float result = 0;

		for(int neuronId = 0; neuronId <who.length;neuronId++ ) {
			result = result + outputNodeGradients[neuronId] * who[neuronId];
		}
		return gradient * result;
	}

	public void calculateNetworkNodeGradients(COST_FUNCTION_TYPE costFType, float[] input,
			float[] target) {
		int outputLayerId = mlp.getLayerSizes().length - 1;
		float[] inputs;
		float[] io = mlp.getLayer(outputLayerId).getNetInputs();
		float[] errorVector = calculateErrorPerNeuron(costFType, input, target);
		if(mlp.isSoftmaxAppliedOnOutput()) {
			for(int neuronId = 0; neuronId < errorVector.length;neuronId++) {
				mlp.getLayer(outputLayerId).getNeuron(neuronId)
				.setNodeGradient(calculateNodeGradient(costFType, ACTIVATION_FUNCTION.SOFTMAX, errorVector[neuronId], io, neuronId));
			}
		}else {
			for(int neuronId = 0; neuronId < errorVector.length;neuronId++) {
				mlp.getLayer(outputLayerId).getNeuron(neuronId)
				.setNodeGradient(calculateNodeGradient(costFType, ACTIVATION_FUNCTION.SIGMOID, errorVector[neuronId], io, neuronId));
			}
		}
		NeuronLayer layer = null;
		float nodeGradient = 0;
		/* iterate top down gradient computation */
		for(int layerIdx = mlp.getLayerSizes().length - 2; layerIdx > 0 ;layerIdx--) {
			/* Retrieve inputs for the layer */
			inputs = mlp.getLayer(layerIdx).getNetInputs();
			layer = mlp.getLayer(layerIdx);
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(layerIdx).size(); neuronIdx++ ) {
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				nodeGradient = calculateNodeGradient(layer.getNeuron(neuronIdx).getActivationFunctionType(), 
						mlp.getLayer(layerIdx+1).getNodeGradients(), layer.getNeuron(neuronIdx).getWeightsAsArray(),inputs[neuronIdx]);
				layer.getNeuron(neuronIdx).setNodeGradient(	nodeGradient);
			}
		}
	}

	public void calculateNetworkNodeGradientsStoredLocaly(COST_FUNCTION_TYPE costFType, float[] input,
			float[] target) {
		if(nodeGradients == null) {
			initiateNodeGradients();
		}
		/* The outputlayer is one index lower since input layer ommited */
		int outputGradienLayertId = mlp.getLayerSizes().length - 2;
		int outputLayerId = mlp.getLayerSizes().length-1;
		float[] inputs;
		float[] Io = mlp.getLayer(outputLayerId).getNetInputs();
		/* Error per neuron (target - output) */
		float[] errorVector = calculateErrorPerNeuron(input, target);
		/* Gradients for output layers calculated in simple fashion, derivative of Error with respect to input */ 
		if(mlp.isSoftmaxAppliedOnOutput()) {
			for(int neuronId = 0; neuronId < errorVector.length;neuronId++) {
				nodeGradients[outputGradienLayertId][neuronId] = 
						calculateNodeGradient(costFType, ACTIVATION_FUNCTION.SOFTMAX, errorVector[neuronId], Io, neuronId);
			}
		}else {
			for(int neuronId = 0; neuronId < errorVector.length;neuronId++) {
				nodeGradients[outputGradienLayertId][neuronId] = 
						calculateNodeGradient(costFType, ACTIVATION_FUNCTION.SIGMOID, errorVector[neuronId], Io, neuronId);
			}
		}
		NeuronLayer layer = null;
		/* iterate top down gradient computation */
		for(int nodeGradientsLayerIdx = mlp.getLayerSizes().length - 2; nodeGradientsLayerIdx > 0 ;nodeGradientsLayerIdx--) {
			/* Retrieve inputs for the layer */
			inputs = mlp.getLayer(nodeGradientsLayerIdx).getNetInputs();
			layer = mlp.getLayer(nodeGradientsLayerIdx);
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(nodeGradientsLayerIdx).size(); neuronIdx++ ) {
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				nodeGradients[nodeGradientsLayerIdx-1][neuronIdx] = calculateNodeGradient(layer.getNeuron(neuronIdx).getActivationFunctionType(), 
						nodeGradients[nodeGradientsLayerIdx],layer.getNeuron(neuronIdx).getWeightsAsArray(), inputs[neuronIdx]);
			}
		}
	}

	private void initiateNodeGradients() {
		nodeGradients = new float[mlp.getLayerSizes().length-1][];	
		for(int layerIdx = 0; layerIdx < nodeGradients.length;layerIdx++) {
			/* skipping input layer mlp.getLayerSizes()[layerIdx+1] */
			nodeGradients[layerIdx] = new float[mlp.getLayerSizes()[layerIdx+1]];
		}
	}

	public void train() {
		HashSet<TERMINATION_CRITERIA> set = new HashSet<TERMINATION_CRITERIA>();
		for(TERMINATION_CRITERIA tc:terminationCriteria.getTerminationCriterias()) {
			set.add(tc);
		}
		if(set.contains(TERMINATION_CRITERIA.MAX_ITERATIONS)) {		
			int rowId = 0;
			int maxIterations = terminationCriteria.getIterations();
			for(int iteration = 0; iteration < maxIterations;iteration++) {
				trainOnSampleWithGainParameterWithoutGainMagnitudeModificationWithDelta(trainingData.getInputRow(rowId),trainingData.getTargetRow(rowId));
				for(int row = 0;row < trainingData.size(); row++) {
					trainOnSampleWithGainParameterWithDeltaRule(trainingData.getInputRow(row),trainingData.getTargetRow(row));
				}
			}
		}

	}

	public void setTrainingTerminationCriteria(TerminationCriteria tc) {
		terminationCriteria = tc;
	}


	public TerminationCriteria getTerminationCriteria() {
		return terminationCriteria;
	}

	public float[][] getLocallyStoredNetworkNodeGradients() {
		return nodeGradients;
	}

	public void setLocallyStoredNodeGradients(float[][] nodeGradients) {
		this.nodeGradients = nodeGradients;
	}

	public void trainOnSampleWithGainParameterWithoutGainMagnitudeModification(float[] inputRow, float[] targetRow) {
		if(nodeGains == null) {
			initiateNodeGains();
		}
		/* Calculate gradients per weight layer */
		calculateNetworkNodeGradientsStoredLocaly(costFunctionType, inputRow, targetRow);
		float nodeGradient = 0;
		float newWeight = 0;
		float currentWeight ;
		int weightLayerIdx = 0;
		int neuronLayerIdx = 0;
		/* Calculate weights per layer */
		for(int gradientLayerIdx = nodeGradients.length-1; gradientLayerIdx >= 0 ; gradientLayerIdx--) {
			weightLayerIdx = gradientLayerIdx;
			neuronLayerIdx = gradientLayerIdx+1;
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(neuronLayerIdx).size(); neuronIdx++) {
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient = nodeGradients[gradientLayerIdx][neuronIdx];
				if(nodeGradient > 0 && nodeGains[gradientLayerIdx][neuronIdx] > 0) {
					updateNodeGainsDecreaseMagnitude(nodeGains,gradientLayerIdx,neuronIdx,-1);
				}else if(nodeGradient < 0 && nodeGains[gradientLayerIdx][neuronIdx] < 0){
					updateNodeGainsDecreaseMagnitude(nodeGains,gradientLayerIdx,neuronIdx,-1);
				}
				for(int weightIdx = 0; weightIdx < mlp.getWeightLayer(weightLayerIdx).length;weightIdx+=mlp.getLayerSizes()[neuronLayerIdx] ) {
					currentWeight = mlp.getWeightLayer(weightLayerIdx)[weightIdx+neuronIdx];

					newWeight = calculateWeight(nodeGradient, oldGradient, learningRate, momentum, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]);
					//					System.out.println("WeightLayerIdx:"+(weightLayerIdx)+" Size of weights:"+
					//							mlp.getWeightLayer(weightLayerIdx).length+" Weight idx:"+(weightIdx+neuronIdx));
					//					System.out.println("LayerIdx:"+(gradientLayerIdx)+" Size of weights:"+
					//							mlp.getLayer(gradientLayerIdx).getWeights().length+" Weight idx:"+(weightIdx+neuronIdx));
					mlp.getLayer(gradientLayerIdx).setWeight(weightIdx+neuronIdx,newWeight);
				}
			}
		}

	}

	public void trainOnSampleWithGainParameterWithoutGainMagnitudeModificationWithDelta(float[] inputRow, float[] targetRow) {
		if(nodeGains == null) {
			initiateNodeGains();
		}
		if(weightDeltas == null) {
			initiateWeightDeltas();
		}
		/* Calculate gradients per weight layer */
		calculateNetworkNodeGradientsStoredLocalyWithDeltaRule(costFunctionType, inputRow, targetRow);
		float nodeGradient = 0;
		float newWeight = 0;
		float currentWeight ;
		int weightLayerIdx = 0;
		int neuronLayerIdx = 0;
		float deltaWeight = 0;
		/* Calculate weights per layer */
		for(int gradientLayerIdx = nodeGradients.length-1; gradientLayerIdx >= 0 ; gradientLayerIdx--) {
			weightLayerIdx = gradientLayerIdx;
			neuronLayerIdx = gradientLayerIdx+1;
			/* loop through each neuron of neuron layer to which belongs current nodeGradients */
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(neuronLayerIdx).size(); neuronIdx++) {

				nodeGradient = nodeGradients[gradientLayerIdx][neuronIdx];

				for(int weightIdx = 0; weightIdx < mlp.getWeightLayer(weightLayerIdx).length;weightIdx+=mlp.getLayerSizes()[neuronLayerIdx] ) {

					deltaWeight = calculateDeltaWeight(nodeGradient, mlp.getLayer(gradientLayerIdx).getOutputs()[weightIdx% mlp.getLayerSizes()[gradientLayerIdx]]);

					currentWeight = mlp.getWeightLayer(weightLayerIdx)[weightIdx+neuronIdx];

					if(nodeGains[gradientLayerIdx][neuronIdx] < 0) {
						newWeight = calculateWeight(-deltaWeight, weightDeltas[gradientLayerIdx][weightIdx], learningRate, momentum, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]*-1);
					}else {
						newWeight = calculateWeight(-deltaWeight, weightDeltas[gradientLayerIdx][weightIdx], learningRate, momentum, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]);
					}
					if(newWeight == Float.NaN) {
						System.err.println("Nan");
					}


					weightDeltas[gradientLayerIdx][weightIdx] = deltaWeight;
					//					System.out.println("WeightLayerIdx:"+(weightLayerIdx)+" Size of weights:"+
					//							mlp.getWeightLayer(weightLayerIdx).length+" Weight idx:"+(weightIdx+neuronIdx));
					//					System.out.println("LayerIdx:"+(gradientLayerIdx)+" Size of weights:"+
					//							mlp.getLayer(gradientLayerIdx).getWeights().length+" Weight idx:"+(weightIdx+neuronIdx));

					mlp.getLayer(gradientLayerIdx).setWeight(weightIdx+neuronIdx,newWeight);
				}
			}
		}

	}

	private void initiateWeightDeltas() {
		float[][] weights = mlp.getWeights();
		weightDeltas = new float[weights.length][];
		for (int i = 0; i < weightDeltas.length; i++) {
			weightDeltas[i] = new float[weights[i].length];
		}		
	}

	public void trainOnSampleWithGainParameter(float[] inputRow, float[] targetRow) {
		if(nodeGains == null) {
			initiateNodeGains();
		}
		if(weightDeltas == null) {
			initiateWeightDeltas();
		}
		/* Calculate gradients per weight layer */
		calculateNetworkNodeGradientsStoredLocaly(costFunctionType, inputRow, targetRow);

		float nodeGradient = 0;
		float newWeight = 0;
		float currentWeight ;
		int weightLayerIdx = 0,neuronLayerIdx = 0;
		/* Calculate weights per layer */
		for(int gradientLayerIdx = nodeGradients.length-1; gradientLayerIdx >= 0 ; gradientLayerIdx--) {
			weightLayerIdx = gradientLayerIdx;
			neuronLayerIdx = gradientLayerIdx+1;
			for(int neuronIdx = 0; neuronIdx < nodeGradients[gradientLayerIdx].length; neuronIdx++) {
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient = nodeGradients[gradientLayerIdx][neuronIdx];
				for(int weightIdx = 0; weightIdx < mlp.getWeightLayer(weightLayerIdx).length;weightIdx+=mlp.getLayerSizes()[neuronLayerIdx] ) {
					/* If sign changed divide the gain by 2 */
					if(haSsignChanged(nodeGradient,gradientLayerIdx,neuronIdx)) {
						updateNodeGainsDecreaseMagnitude(nodeGains,gradientLayerIdx,neuronIdx,gainReductionFactor);
					}
					/* If repeated increase by small value*/
					else {	
						updateNodeGainsIncreaseMagnitude(nodeGains,gradientLayerIdx,neuronIdx,gainMagnitudeIncrement );
					}
					currentWeight = mlp.getWeightLayer(weightLayerIdx)[weightIdx+neuronIdx];
					/* nodeGains has double function, stores the sign and the factor for the learningRate,
					 * if nodeGain negative, multiply by -1 to make multiplier positive*/
					if(nodeGains[gradientLayerIdx][neuronIdx]<0) {
						newWeight = calculateWeight(nodeGradient, oldGradient, learningRate, momentum
								, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]*-1);
					}else {
						newWeight = calculateWeight(nodeGradient, oldGradient, learningRate, momentum
								, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]);
					}
					newWeight = calculateWeight(nodeGradient, oldGradient, learningRate, momentum, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]);
					//					System.out.println("WeightLayerIdx:"+(weightLayerIdx)+" Size of weights:"+
					//							mlp.getWeightLayer(weightLayerIdx).length+" Weight idx:"+(weightIdx+neuronIdx));
					//					System.out.println("LayerIdx:"+(gradientLayerIdx)+" Size of weights:"+
					//							mlp.getLayer(gradientLayerIdx).getWeights().length+" Weight idx:"+(weightIdx+neuronIdx));
					mlp.getLayer(gradientLayerIdx).setWeight(weightIdx+neuronIdx,newWeight);
				}				

			}
		}

	}

	public void trainOnSampleWithGainParameterWithDeltaRule(float[] inputRow, float[] targetRow) {
		if(nodeGains == null) {
			initiateNodeGains();
		}
		if(weightDeltas == null) {
			initiateWeightDeltas();
		}
		/* Calculate gradients per weight layer */
		calculateNetworkNodeGradientsStoredLocalyWithDeltaRule(costFunctionType, inputRow, targetRow);
		float[] outputs;
		float nodeGradient = 0;
		float newWeight = 0;
		float currentWeight ;
		float deltaWeight = 0;
		int weightLayerIdx = 0,neuronLayerIdx = 0;
		/* Calculate weights per layer */
		for(int gradientLayerIdx = nodeGradients.length-1; gradientLayerIdx >= 0 ; gradientLayerIdx--) {
			weightLayerIdx = gradientLayerIdx;
			neuronLayerIdx = gradientLayerIdx+1;
			for(int neuronIdx = 0; neuronIdx < nodeGradients[gradientLayerIdx].length; neuronIdx++) {
				outputs = mlp.getLayer(neuronLayerIdx-1).getOutputs();
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient = nodeGradients[gradientLayerIdx][neuronIdx];
				/* If sign changed divide the gain by 2 */
				if(haSsignChanged(nodeGradient,gradientLayerIdx,neuronIdx)) {
					updateNodeGainsDecreaseMagnitude(nodeGains,gradientLayerIdx,neuronIdx,gainReductionFactor);
				}
				/* If repeated increase by small value*/
				else {	
					updateNodeGainsIncreaseMagnitude(nodeGains,gradientLayerIdx,neuronIdx,gainMagnitudeIncrement );
				}
				for(int weightIdx = 0; weightIdx < mlp.getWeightLayer(weightLayerIdx).length;weightIdx+=mlp.getLayerSizes()[neuronLayerIdx] ) {
					deltaWeight = calculateDeltaWeight(nodeGradient, outputs[weightIdx%mlp.getLayerSizes()[neuronLayerIdx]]);

					currentWeight = mlp.getWeightLayer(weightLayerIdx)[weightIdx+neuronIdx];
					/* nodeGains has double function, stores the sign and the factor for the learningRate,
					 * if nodeGain negative, multiply by -1 to make multiplier positive*/

					if(nodeGains[gradientLayerIdx][neuronIdx]<0) {
						newWeight = calculateWeight(-deltaWeight, weightDeltas[gradientLayerIdx][weightIdx], learningRate, momentum
								, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]*-1);
					}else {
						newWeight = calculateWeight(-deltaWeight, weightDeltas[gradientLayerIdx][weightIdx], learningRate, momentum
								, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]);
					}
					if(newWeight == Float.NaN) {
						System.err.println("Nan");
					}

					/* Store old deltaWeight */
					weightDeltas[gradientLayerIdx][weightIdx] = deltaWeight;
					//					System.out.println("WeightLayerIdx:"+(weightLayerIdx)+" Size of weights:"+
					//							mlp.getWeightLayer(weightLayerIdx).length+" Weight idx:"+(weightIdx+neuronIdx));
					//					System.out.println("LayerIdx:"+(gradientLayerIdx)+" Size of weights:"+
					//							mlp.getLayer(gradientLayerIdx).getWeights().length+" Weight idx:"+(weightIdx+neuronIdx));
					mlp.getLayer(gradientLayerIdx).setWeight(weightIdx+neuronIdx,newWeight);
				}				
			}
		}

	}

	private boolean haSsignChanged(float nodeGradient, int layerIdx, int neuronIdx) {
		if(nodeGradients[layerIdx][neuronIdx] > 0 && nodeGradient > 0 || nodeGradients[layerIdx][neuronIdx] < 0 && 
				nodeGradient < 0) {
			return false;
		}else {
			return true;
		}
	}

	/**
	 * 
	 * @param values 
	 * @param layerIdx
	 * @param neuronIdx
	 * @param gainMagnitudeIncrement - value by which magnitude will be increased. If gain is negative the
	 * value will be subtracted, if gain is positive the value added
	 */
	public void updateNodeGainsIncreaseMagnitude(float[][] values, int layerIdx, int neuronIdx, float gainMagnitudeIncrement) {

		if(Math.signum(values[layerIdx][neuronIdx]) > 0 ) {
			values[layerIdx][neuronIdx]+=gainMagnitudeIncrement;
		}else {
			values[layerIdx][neuronIdx]-=gainMagnitudeIncrement;
		}
	}

	/** 
	 * @param values
	 * @param layerIdx
	 * @param neuronIdx
	 * @param f
	 */
	public void updateNodeGainsDecreaseMagnitude(float[][] values, int layerIdx, int neuronIdx, float f) {
		values[layerIdx][neuronIdx] = values[layerIdx][neuronIdx]*f;		
	}

	public float[][] initiateNodeGains() {
		/* one node gain layer per weight layer, Node gains for input layer are omitted*/
		int nodeGainLayerCount = mlp.getLayerSizes().length-1;
		nodeGains = new float[nodeGainLayerCount][];		
		/* input layer does not have nodeGains, starting with second layer mlp.getLayerSizes()[layerId+1]*/
		for(int layerId = 0; layerId < nodeGainLayerCount;layerId++) {
			nodeGains[layerId] = new float[mlp.getLayerSizes()[layerId+1]];
			for(int i = 0; i < nodeGains[layerId].length;i++) {
				nodeGains[layerId][i] = 1;
			}
		}
		return nodeGains;
	}

	public float[][] getNodeGains() {
		return nodeGains;
	}

	/**
	 * Error = (tareget - observed)
	 * @param inputRow
	 * @param targetRow
	 * @return
	 */
	public float[] calculateErrorPerNeuron(float[] inputRow, float[] targetRow) {
		float[] result = mlp.predict(inputRow);
		float[] error = new float[result.length];
		for(int i = 0;i < error.length;i++) {
			error[i] = targetRow[i]-result[i];
		}
		return error;		
	}

	/**
	 * For Squared Error
	 * Io - neuron input, E = (Required - Predicted)^2
	 * Oo - output of output neuron, 
	 * Io - input of output neuron
	 * ∂(E)^2/∂Io = ∂(target - observed)^2/∂Oo * -∂fo(Io)/∂Io = gradient
	 * (∂E^2/∂Io) => 2(target - observed)  - first step of derivation
	 * ∂(target - observed)/∂Oo => -Oo 
	 * (∂f(Io)/∂Io) => f'(Io), f(.) - softmax
	 * 2(target - observed) * -fo'(Io) = gradient
	 * 
	 * @param costFunction - cost function type
	 * @param activationFunction - activation function type
	 * @param diffTarPred - difference (target - observed)
	 * @param ioAll - inputs for layer
	 * @param neuronId - index 
	 * @return
	 */
	public float calculateNodeGradientDeltaRule(COST_FUNCTION_TYPE costFunction, ACTIVATION_FUNCTION activationFunction, float diffTarPred,
			float[] ioAll, int neuronId) {
		float result = 0;
		switch(costFunction) {
		case SQUARED_ERROR:
			switch(activationFunction) {
			case SOFTMAX:
				result = 2 * diffTarPred * -StatisticUtils.calculateSoftmaxPartialDerivative(ioAll, neuronId);
				if(Float.isNaN(result)) {
					System.err.println("NaN i result calculateNodeGradientDeltarule");
					throw new RuntimeException("NaN");
				}
				break;
			default:
				System.err.println("Function not implemented:"+activationFunction);
				break;
			}
			break;
		default:
			System.err.println("Function not implemented:"+costFunction);
			break;
		}
		return result;
	}

	public float calculateNodeGradientDeltaRule(ACTIVATION_FUNCTION activationFunction, float outputGradient, float weight,
			float input) {
		float result = -1;
		switch(activationFunction) {
		case SIGMOID:
			result = 	outputGradient * weight * NeuronFunctionModels.derivativeOf(activationFunction, input);
			break;
		default:
			System.err.println("Function not implemented:"+activationFunction);
			break;
		}
		return result;
	}

	public void calculateNetworkNodeGradientsStoredLocalyWithDeltaRule(COST_FUNCTION_TYPE costFType, float[] inputRow,
			float[] targetRow) {
		if(nodeGradients == null) {
			initiateNodeGradients();
		}
		/* The outputlayer is one index lower since input layer ommited */
		int outputGradienLayertId = nodeGradients.length-1;
		int outputLayerId = mlp.getLayerSizes().length-1;
		float[] inputs;
		float[] Io = mlp.getLayer(outputLayerId).getNetInputs();
		float nodeGradient = 0;
		/* Error per neuron (target - output) */
		float[] diffTarPred = calculateDifferenceTargetSubPredicted(inputRow, targetRow);
		/* Gradients for output layers calculated in simple fashion, derivative of Error with respect to input */ 
		if(mlp.isSoftmaxAppliedOnOutput()) {

			for(int neuronIdx = 0; neuronIdx < diffTarPred.length;neuronIdx++) {
				nodeGradient = calculateNodeGradientDeltaRule(costFType, ACTIVATION_FUNCTION.SOFTMAX, diffTarPred[neuronIdx],Io, neuronIdx);
				if(haSsignChanged(nodeGradient, outputGradienLayertId, neuronIdx)) {
					updateNodeGainsDecreaseMagnitude(nodeGradients,outputGradienLayertId,neuronIdx,gainReductionFactor);
				}else{
					updateNodeGainsIncreaseMagnitude(nodeGradients,outputGradienLayertId,neuronIdx,gainMagnitudeIncrement);
				}
				nodeGradients[outputGradienLayertId][neuronIdx] = nodeGradient;						
			}
		}else {
			for(int neuronIdx = 0; neuronIdx < diffTarPred.length;neuronIdx++) {
				nodeGradient = calculateNodeGradientDeltaRule(costFType, ACTIVATION_FUNCTION.SIGMOID, diffTarPred[neuronIdx],Io, neuronIdx);
				if(haSsignChanged(nodeGradient, outputGradienLayertId, neuronIdx)) {
					updateNodeGainsDecreaseMagnitude(nodeGradients,outputGradienLayertId,neuronIdx,gainReductionFactor);
				}else{
					updateNodeGainsIncreaseMagnitude(nodeGradients,outputGradienLayertId,neuronIdx,gainMagnitudeIncrement);
				}
				nodeGradients[outputGradienLayertId][neuronIdx] = nodeGradient;

			}
		}
		NeuronLayer layer = null;
		/* iterate top down gradient computation */
		for(int nodeGradientsLayerIdx = nodeGradients.length -2; nodeGradientsLayerIdx >= 0 ;nodeGradientsLayerIdx--) {
			/* Retrieve inputs for the 2nd layer */
			layer = mlp.getLayer(nodeGradientsLayerIdx+1);
			inputs =layer.getNetInputs();
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < layer.size(); neuronIdx++ ) {
				nodeGradient = calculateNodeGradient(layer.getNeuron(neuronIdx).getActivationFunctionType(),
						nodeGradients[nodeGradientsLayerIdx+1], layer.getNeuron(neuronIdx).getWeightsAsArray(), inputs[neuronIdx]);
				if(haSsignChanged(nodeGradient, nodeGradientsLayerIdx, neuronIdx)) {
					updateNodeGainsDecreaseMagnitude(nodeGradients,nodeGradientsLayerIdx,neuronIdx,gainReductionFactor);
				}else{
					updateNodeGainsIncreaseMagnitude(nodeGradients,nodeGradientsLayerIdx,neuronIdx,gainMagnitudeIncrement);
				}
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				nodeGradients[nodeGradientsLayerIdx][neuronIdx] = nodeGradient;
			}
		}

	}

	public float[] calculateDifferenceTargetSubPredicted(float[] inputRow, float[] targetRow) {
		float[] predicted = mlp.predict(inputRow);
		float[] difference = new float[predicted.length];
		for(int i = 0;i < difference.length;i++) {
			difference[i] = targetRow[i]-predicted[i];
		}
		return difference;		
	}

	public void setMomentumDecayFactor(float decayFactor) {
		this.momentDecayFactor = decayFactor;		
	}

	public float getMomentumDecayFactor() {
		return momentDecayFactor;
	}
}
