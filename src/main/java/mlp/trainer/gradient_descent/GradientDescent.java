package mlp.trainer.gradient_descent;

import math.utils.StatisticUtils;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.NeuronFunctionModels;
import mlp.trainer.Backpropagation;

public class GradientDescent extends Backpropagation {

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
	public float calculateGradientInputOverError(COST_FUNCTION_TYPE costFType, ACTIVATION_FUNCTION activationFType,
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
	 * 
	 * @param costFType - type of cost function to apply
	 * @param error - (predicted - required)
	 * @param neuronIdx 
	 * @param io - array with net inputs for output layer 
	 * @return
	 */
	public float calculateGradientInputOverError(COST_FUNCTION_TYPE costFType, float error,int neuronIdx, float[] io) {
		// TODO Auto-generated method stub
		return 0;
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
	public float calculateGradientInputOverError(ACTIVATION_FUNCTION activFunction, float gradient,float Ih, float Who, float a, float b) {
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
	 * @param gradient
	 * @param Oh
	 * @param currentWeight - weight
	 * @return
	 */
	public float calculateDeltaWeight(float gradient, float Oh, float currentWeight) {
		return gradient * Oh * currentWeight;
	}

	public float calculateWeight(float deltaWeight, float oldDelta, float learningRate, float momentum,
			float oldWeight) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	
}
