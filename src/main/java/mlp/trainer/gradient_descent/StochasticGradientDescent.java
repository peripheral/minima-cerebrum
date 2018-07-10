package mlp.trainer.gradient_descent;

import java.util.Random;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.NeuronFunctionModels;
import mlp.tests.ActivationFunctionModelTest;
import mlp.trainer.TrainingData;

public class StochasticGradientDescent {

	public enum COST_FUNCTION_TYPE {
		SQUARED_ERROR

	}

	private TrainingData trainingData;
	private float learningRate = 0.1f;

	public void setTrainingData(TrainingData td) {
		trainingData = td;
	}

	/**
	 * Randomly selects data rows from trainingData
	 * @param size - size of the batch
	 * @return
	 */
	public TrainingData generateTrainingBatch(int size) {
		TrainingData td = new TrainingData();
		Random rm = new Random();
		float[][] data = new float[size][];
		for(int i = 0; i < size; i++) {
			data[i] = trainingData.getDataRow(rm.nextInt(trainingData.size()));
		}
		td.setData(data);
		return td;
	}

	public float getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(float learningRate) {
		this.learningRate = learningRate;		
	}

	/**
	 * 
	 * @param momentum - fraction of oldWeight
	 * @param learningRate - learning rate, step size
	 * @param oldWeight - initial weight
	 * @param deltaW - delta to be added
	 * @return - new weight
	 */
	public float generateNewWeight(float momentum, float learningRate, float oldWeight, float deltaW) {
		return oldWeight + momentum*oldWeight + learningRate * deltaW;
	}

	/**
	 * Gradient of Change in Input with respect to error
	 * @param fType - type of cost function
	 * @param error - (Predicted - Required)
	 * @param Io - neuron input
	 * @param a 
	 * @param b
	 * @return 2E * fo'(Io)
	 */
	public float calculateGradientInputOverError(COST_FUNCTION_TYPE fType,ACTIVATION_FUNCTION aType, float error,
			float Io, float a, float b) {
		switch(fType) {
		case SQUARED_ERROR:
			return 2*error * NeuronFunctionModels.derivativeOf(aType, a, b, Io);
		default :
			System.err.println("Cost function not implemented:"+fType);
			break;
		}
		return 0;
	}

	/**
	 * ∂(E)^2/I_h = (∂E^2/∂I_o)(∂I_o/∂O_h)(∂O_h/∂I_h)
	 *  (∂E^2/∂I_o) => 2E * fo'(I_o),  Fo(.) - output function 
	 *  (∂I_o/∂O_h) => Wh, 
	 *  (∂O_h/∂I_h) => fh'(I_h)  , fh(.) - hidden neuron activation function, partial derivative of
	 *  output of hidden neuron over input of hidden neuron
	 * h - hidden neuron, o - output of connecting neuron
	 *  gradient * Wh * fh'(I_h) =  gradient  input of hidden neuron over Error 
	 * Gradient of Change in Input with respect to error, based on previous gradient, weight and input
	 * @param fType
	 * @param gradientInputToError
	 * @param wh - weight on connection
	 * @param Ih - input of neuron
	 * @return
	 */
	public float calculateGradientInputOverError(ACTIVATION_FUNCTION fType, float gradientInputToError, float wh,
			float Ih,float a,float b) {
		switch(fType) {
		case SIGMOID:
			return gradientInputToError * wh*NeuronFunctionModels.derivativeOf(fType, a, b, Ih);
		default:
			System.err.println("Activation function not implemented:"+fType);
			break;
		}
		return 0;
	}

}
