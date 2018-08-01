package mlp.trainer;

import mlp.ANN_MLP;

public class Backpropagation {

	protected ANN_MLP mlp;
	protected TrainingData trainingData;
	private float defaultErrorMinimimg = 0.1f;
	private float approximateErrorMinimum = defaultErrorMinimimg;
	protected float learningRate = 0.001f;
	protected COST_FUNCTION_TYPE costFunctionType = COST_FUNCTION_TYPE.SQUARED_ERROR;

	public enum COST_FUNCTION_TYPE {
		SQUARED_ERROR
	}

	public void setMLP(ANN_MLP mlp) {
		this.mlp = mlp;
	}

	public void setTrainingData(TrainingData td) {
		this.trainingData = td;
	}

	/**
	 * Calculates total error as sum by applying the training data on ann mlp
	 * @return
	 */
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
	
	/**
	 * Calculates total error as sum by applying the training data on ann mlp
	 * @return
	 */
	public float calculateTotalMSE() {
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
		float sum = 0;
		for(float f:result) {
			sum = sum + f;
		}
		return sum;
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

	public float[] calculateErrorPerNeuron(COST_FUNCTION_TYPE costFunctionType, float[] input, float[] target) {
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
	
	public void setCostFunctionType(COST_FUNCTION_TYPE costFunctionType) {
		this.costFunctionType = costFunctionType;		
	}
	
	/**
	 * Returns learning rate. Default - 0.1
	 * @return
	 */
	public float getLearningRate() {
		return learningRate;
	}
	

	public void setLearningRate(float learningRate) {
		this.learningRate = learningRate;		
	}

	public COST_FUNCTION_TYPE getCostFunctionType() {
		return costFunctionType;
	}
}
