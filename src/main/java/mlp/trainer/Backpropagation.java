package mlp.trainer;

import java.util.Arrays;

import mlp.ANN_MLP;
import mlp.trainer.Backpropagation.COST_FUNCTION_TYPE;

public class Backpropagation {

	private ANN_MLP mlp;
	protected TrainingData trainingData;
	private float defaultErrorMinimimg = 0.1f;
	private float approximateErrorMinimum = defaultErrorMinimimg;

	public enum COST_FUNCTION_TYPE {
		SQUARED_ERROR
	}

	public void setMLP(ANN_MLP mlp) {
		this.mlp = mlp;

	}

	public void setTrainingData(TrainingData td) {
		this.trainingData = td;
	}

	public float[] calCulateMeanSquaredErrorPerNeuron() {
		double[] MSE = new double[mlp.getOutputLayer().size()];
		float[][] inputs = trainingData.getInputs();
		float[] result = null;
		float[] target;

		for(int i = 0; i < inputs.length;i++) {
			result = mlp.predict(inputs[i]);
			target =  trainingData.getTargetRow(i);
			for(int ii = 0; ii < result.length;ii++) {
				MSE[ii] =  MSE[ii]+ Math.pow(result[ii] - target[ii],2)/inputs.length;
			}
		}
		for(int i = 0; i < result.length;i++) {
			result[i] = (float) MSE[i];
		}

		return result;
	}

	public void setApproximateErrorMinimum(float errMinimum) {
		approximateErrorMinimum = errMinimum;		
	}

	/**
	 * Stopping criteria, error minimum
	 * @return
	 */
	public float getApproximateErrorMinimum() {
		return approximateErrorMinimum;
	}

	public float[] getErrorPerNeuron(COST_FUNCTION_TYPE costFunctionType, float[] input, float[] target) {
		switch(costFunctionType) {
		case SQUARED_ERROR:
			float[] result = mlp.predict(input);
			float[] error = new float[result.length];
			for(int i = 0;i < error.length;i++) {
				error[i] = (float) Math.pow(result[i] - target[i],2);
			}
			return error;
		default:
			System.err.println("Not implemented cost function:"+costFunctionType);
			break;
		}
		return null;
	}


}
