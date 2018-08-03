package mlp.trainer.gradient_descent;

import java.util.Arrays;

import math.utils.StatisticUtils;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.NeuronFunctionModels;
import mlp.NeuronLayer;
import mlp.trainer.Backpropagation;
import mlp.trainer.TerminationCriteria;

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
	private float gainMagnitudeIncrement = 0.005f;


	public void getDeltasForLayer(int layerId) {
		// TODO Auto-generated method stub

	}

	/**
	 * Function calculates gradient from error and given input and activation of layer function parameters
	 * @param costFType -type of cost function 
	 * @param activationFType - type of fo(.)
	 * @param error - (predicted - required )
	 * @param io - net input of the output neuron
	 * @param a
	 * @param b
	 * 2E * fo'(Io) = gradient
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
		float result = 2 * error * partialDerivative;
		return result;
	}

	/**
	 * Function calculates gradient from error and given input and activation of layer function parameters.
	 * paramaters a and b = 1; Main function calculation of node gradients of output layer neurons
	 * @param costFType -type of cost function 
	 * @param activationFType - type of fo(.)
	 * @param error - (predicted - required )
	 * @param io - net input of the output neurons
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
				result = 2 * error * partialDerivative;
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
	 * @param nodeGradient -  ∂(E)^2/∂Who
	 * @param oldNodeGradient - gradient previously used weight to calc
	 * @param learningRate - learning rate, factor decreases
	 * @param momentum - to help to progress through low gradient
	 * @param currentWeight - initial weight of incoming connection to the node with nodeGradient
	 * @param nodeGain - per node learning rate modifier
	 * @return currentWeight + learningRate * nodeGradient + momentum * oldNodeGradient
	 */
	public float calculateWeight(float nodeGradient, float oldNodeGRadient, float learningRate, float momentum,
			float currentWeight, float nodeGain) {
		return currentWeight + learningRate * nodeGradient * nodeGain + (momentum * oldNodeGRadient);
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

	public float calculateNodeGradient(ACTIVATION_FUNCTION activationFunctionType, float[] outputNodeGradients, float ih,
			float[] who) {
		float a = 1;
		float b = 1;
		float gradient = 0;
		switch(activationFunctionType) {
		case SOFTMAX:
			//	gradient = StatisticUtils.calculateSoftmaxPartialDerivative(data, idx)
			System.err.println("Error SOFTMAX is not implemented");
			break;
		case SIGMOID:
			gradient =  NeuronFunctionModels.derivativeOf(activationFunctionType, a, b,ih);
			break;
		default:
			System.err.println("Activation function not implemented:"+activationFunctionType);
			break;
		}

		float result = 0;
		int counter = 0;
		for(float outNodGrad:outputNodeGradients) {
			result = result + gradient * outNodGrad * who[counter++];
		}
		return result;
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
						mlp.getLayer(layerIdx+1).getNodeGradients(), inputs[neuronIdx],layer.getWeights());
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
		float[] io = mlp.getLayer(outputLayerId).getNetInputs();
		float[] errorVector = calculateErrorPerNeuron(costFType, input, target);
		if(mlp.isSoftmaxAppliedOnOutput()) {
			for(int neuronId = 0; neuronId < errorVector.length;neuronId++) {
				nodeGradients[outputGradienLayertId][neuronId] = 
				calculateNodeGradient(costFType, ACTIVATION_FUNCTION.SOFTMAX, errorVector[neuronId], io, neuronId);
			}
		}else {
			for(int neuronId = 0; neuronId < errorVector.length;neuronId++) {
				nodeGradients[outputGradienLayertId][neuronId] = calculateNodeGradient(costFType, ACTIVATION_FUNCTION.SIGMOID, errorVector[neuronId], io, neuronId);
			}
		}
		NeuronLayer layer = null;
		/* iterate top down gradient computation */
		for(int layerIdx = mlp.getLayerSizes().length - 2; layerIdx > 0 ;layerIdx--) {
			/* Retrieve inputs for the layer */
			inputs = mlp.getLayer(layerIdx).getNetInputs();
			layer = mlp.getLayer(layerIdx);
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(layerIdx).size(); neuronIdx++ ) {
				/* gradient is product between f'(in) * sum ( upperLayerGradient_h*weight_ho + ..) */
				nodeGradients[layerIdx-1][neuronIdx] = calculateNodeGradient(layer.getNeuron(neuronIdx).getActivationFunctionType(), 
						nodeGradients[layerIdx], inputs[neuronIdx],layer.getWeights());
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
		// TODO Auto-generated method stub
		
	}

	public void setTerminationCriteria(TerminationCriteria tc) {
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

	public void trainOnSampleWithGainParameterWithoutGainReduction(float[] inputRow, float[] targetRow) {
		if(nodeGains == null) {
			initiateNodeGains();
		}
		/* Calculate gradients per weight layer */
		calculateNetworkNodeGradientsStoredLocaly(costFunctionType, inputRow, targetRow);
		System.out.println("Node Gradients length:"+nodeGradients.length);
		System.out.println("Node Gains length:"+nodeGains.length);
		float nodeGradient = 0;
		float newWeight = 0;
		float currentWeight ;
		int weightLayerIdx = 0;
		int neuronLayerIdx = 0;
		/* Calculate weights per layer */
		for(int gradientLayerIdx = nodeGradients.length-1; gradientLayerIdx > 0 ; gradientLayerIdx--) {
			weightLayerIdx = gradientLayerIdx;
			neuronLayerIdx = gradientLayerIdx+1;
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(neuronLayerIdx).size(); neuronIdx++) {
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient = nodeGradients[gradientLayerIdx][neuronIdx];
				if(nodeGradient > 0 && nodeGains[gradientLayerIdx][neuronIdx] > 0) {
					updateNodeGainsDecrease(gradientLayerIdx,neuronIdx,-1);
				}else if(nodeGradient < 0 && nodeGains[gradientLayerIdx][neuronIdx] < 0){
					updateNodeGainsDecrease(gradientLayerIdx,neuronIdx,-1);
				}
				for(int weightIdx = 0; weightIdx < mlp.getWeightLayer(weightLayerIdx).length;weightIdx+=mlp.getLayerSizes()[neuronLayerIdx] ) {
					currentWeight = mlp.getWeightLayer(weightLayerIdx)[weightIdx+neuronIdx];

					newWeight = calculateWeight(nodeGradient, oldGradient, learningRate, momentum, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]);

					mlp.getLayer(gradientLayerIdx-1).setWeight(weightIdx+neuronIdx,newWeight);
				}				

			}
		}
		
	}
	
	public void trainOnSampleWithGainParameter(float[] inputRow, float[] targetRow) {
		if(nodeGains == null) {
			initiateNodeGains();
		}
		/* Calculate gradients per weight layer */
		calculateNetworkNodeGradientsStoredLocaly(costFunctionType, inputRow, targetRow);

		float nodeGradient = 0;
		float newWeight = 0;
		float currentWeight ;
		/* Calculate weights per layer */
		for(int gradientLayerIdx = nodeGradients.length-1; gradientLayerIdx >= 0 ; gradientLayerIdx--) {
			for(int neuronIdx = 0; neuronIdx < nodeGradients[gradientLayerIdx].length; neuronIdx++) {
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient = nodeGradients[gradientLayerIdx][neuronIdx];
				if(nodeGradient > 0 && nodeGains[gradientLayerIdx][neuronIdx] > 0) {
			
					updateNodeGainsDecrease(gradientLayerIdx,neuronIdx,-gainReductionFactor);
				
				}else if(nodeGradient < 0 && nodeGains[gradientLayerIdx][neuronIdx] < 0){
				
					updateNodeGainsDecrease(gradientLayerIdx,neuronIdx,-gainReductionFactor);
				
				}else {
			
					updateNodeGainsIncreaseMagnitude(gradientLayerIdx,neuronIdx,gainMagnitudeIncrement );
				}
				for(int weightIdx = 0; weightIdx < mlp.getLayer(gradientLayerIdx).getWeights().length;weightIdx+=mlp.getLayerSizes()[gradientLayerIdx+1] ) {
					currentWeight = mlp.getLayer(gradientLayerIdx).getWeights()[weightIdx+neuronIdx];

					newWeight = calculateWeight(nodeGradient, oldGradient, learningRate, momentum, currentWeight,nodeGains[gradientLayerIdx][neuronIdx]);

					mlp.getLayer(gradientLayerIdx).setWeight(weightIdx+neuronIdx,newWeight);
				}				

			}
		}
		
	}

	/**
	 * 
	 * @param layerIdx
	 * @param neuronIdx
	 * @param gainMagnitudeIncrement - value by which magnitude will be increased. If gain is negative the
	 * value will be subtracted, if gain is positive the value added
	 */
	private void updateNodeGainsIncreaseMagnitude(int layerIdx, int neuronIdx, float gainMagnitudeIncrement) {
		if(Math.signum(nodeGains[layerIdx][neuronIdx]) > 0 ) {
			nodeGains[layerIdx][neuronIdx]+=gainMagnitudeIncrement;
		}else {
			nodeGains[layerIdx][neuronIdx]-=gainMagnitudeIncrement;
		}
	}

	private void updateNodeGainsDecrease(int layerIdx, int neuronIdx, float f) {
		nodeGains[layerIdx][neuronIdx] = nodeGains[layerIdx][neuronIdx]*f;		
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
}
