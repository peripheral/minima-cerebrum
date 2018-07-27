package mlp.trainer.gradient_descent;

import math.utils.StatisticUtils;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.Neuron;
import mlp.NeuronFunctionModels;
import mlp.trainer.Backpropagation;

public class GradientDescent extends Backpropagation {

	/**
	 * Default momentum
	 */
	private float momentum = 0.00001f;
	private float oldGradient = 0f;

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
		float a =1,b=1;
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
	 * Calcualtes new weight
	 * @param nodeGradient -  ∂(E)^2/∂Who
	 * @param oldNodeGRadient - gradient previously used weight to calc
	 * @param learningRate - learning rate, factor decreases
	 * @param momentum - to help to progress through low gradient
	 * @param currentWeight - initial weight
	 * @return currentWeight + learningRate * deltaWeight + momentum * oldDeltaWeight
	 */
	public float calculateWeight(float nodeGradient, float oldNodeGRadient, float learningRate, float momentum,
			float currentWeight) {
		return currentWeight + learningRate * nodeGradient + (momentum * oldNodeGRadient);
	}

	public void trainOnSample(float[] inputRow, float[] targetRow) {
		float[] errorVector = calculateErrorPerNeuron(costFunctionType, inputRow, targetRow);
		
		int outputLayerIdx = mlp.getLayerSizes().length -1;
		float[] io = mlp.getLayer(outputLayerIdx).getNetInputs();
		//float[][] weights = mlp.getWeights();
		
		float[] inputs;
	
		
		ACTIVATION_FUNCTION activationFType = ACTIVATION_FUNCTION.SOFTMAX;
		/* Produce error gradient for each output neuron */
		
		for(int errIdx = 0; errIdx < errorVector.length; errIdx++ ) {
			mlp.setNodeGradient(outputLayerIdx,errIdx,calculateNodeGradient(costFunctionType,
					activationFType, errorVector[errIdx], io, errIdx));
		}
		activationFType = ACTIVATION_FUNCTION.SIGMOID;
		/* Calculate gradients per weight layer */
		for(int layerIdx = mlp.getLayerSizes().length-2; layerIdx >= 0 ; layerIdx--) {
			/* Retrieve inputs for the layer */
			inputs = mlp.getLayer(layerIdx).getNetInputs();
			/* calculate gradients per neuron of current neuron layer */
			for(int neuronIdx = 0; neuronIdx < errorVector.length; neuronIdx++ ) {
				mlp.setNodeGradient(layerIdx,neuronIdx, calculateNodeGradient(activationFType, mlp.getLayerNodeGradients(layerIdx+1),
						inputs[neuronIdx], mlp.getLayerWeights(layerIdx)));
			}
		}
		int neuronId = 0;
		float nodeGradient = 0;
		/* Calculate weights per layer */
		for(int layerIdx = mlp.getLayerSizes().length-1; layerIdx > 0 ; layerIdx--) {
			for(int neuronIdx = 0; neuronIdx < mlp.getLayer(layerIdx).size(); neuronIdx++) {
				/* Gradient must be negative to reach a valley. set Learning rate to negative to 
				 * make delta negative */
				nodeGradient = mlp.getNodeGradient(layerIdx,neuronId);
				if(nodeGradient > 0 && learningRate > 0) {
					learningRate = learningRate * -1;
				}else if(nodeGradient < 0 && learningRate < 0){
					learningRate = learningRate * -1;
				}
				for(int neuronIdxL2 = 0; neuronIdxL2 < mlp.getLayer(layerIdx-1).getNeurons().size();neuronIdxL2++ ) {
					mlp.getLayer(layerIdx-1).getNeuron(neuronIdxL2).setWeight(neuronIdx,calculateWeight(nodeGradient, 
							oldGradient, learningRate, momentum, mlp.getLayer(layerIdx-1).getNeuron(neuronIdxL2).getWeight(neuronIdx)));
				}				
					
			}
		}

	}

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

	public float calculateNodeGradient(ACTIVATION_FUNCTION activationFunction, float[] outputNodeGradients, float ih,
			float[] who) {
		float a = 1;
		float b = 1;
		float gradient = 0;
		switch(activationFunction) {
		case SOFTMAX:
		//	gradient = StatisticUtils.calculateSoftmaxPartialDerivative(data, idx)
			System.err.println("Error SOFTMAX is not implemented");
			break;
		case SIGMOID:
			gradient =  NeuronFunctionModels.derivativeOf(activationFunction, a, b,ih);
			break;
		}
		
		float result = 0;
		int counter = 0;
		for(float outNodGrad:outputNodeGradients) {
			result = result + gradient * outNodGrad * who[counter++];
		}
		return result;
	}

	public void calculateNetworkNodeGradients(float[] error) {
		// TODO Auto-generated method stub
		
	}




}
