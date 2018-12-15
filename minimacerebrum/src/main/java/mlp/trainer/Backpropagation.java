package mlp.trainer;

import mlp.ANNMLP;
import mlp.trainer.data.TrainingData;

public class Backpropagation {

	protected ANNMLP mlp;
	protected TrainingData trainingData;
	private float defaultErrorMinimimg = 0.1f;
	private float approximateErrorMinimum = defaultErrorMinimimg;
	protected float learningRate = 0.001f;
	/* costFunctionType - specifies which object function is used, default SQUARED_ERRO */
	protected COST_FUNCTION_TYPE costFunctionType = COST_FUNCTION_TYPE.SQUARED_ERROR;

	public enum COST_FUNCTION_TYPE {
		SQUARED_ERROR
	}

	public void setMLP(ANNMLP mlp) {
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
		double[] mSE = new double[mlp.getOutputLayer().size()];
		float[][] inputs = trainingData.getInputs();
		float[] result = new float[0];
		float[] target;

		for(int i = 0; i < inputs.length;i++) {
			result = mlp.predict(inputs[i]);
			target =  trainingData.getTargetRow(i);
			for(int ii = 0; ii < result.length;ii++) {
				mSE[ii] =  mSE[ii]+ Math.pow(result[ii] - target[ii],2)/inputs.length;
			}
		}
		for(int i = 0; i < result.length;i++) {
			result[i] = (float) mSE[i];
		}

		return result;
	}
	
	/**
	 * Calculates total error as sum by applying the training data on ann mlp
	 * @return
	 */
	public float calculateTotalMSE() {
		double[] mSE = new double[mlp.getOutputLayer().size()];
		float[][] inputs = trainingData.getInputs();
		float[] result = new float[0];
		float[] target;

		for(int i = 0; i < inputs.length;i++) {
			result = mlp.predict(inputs[i]);
			target =  trainingData.getTargetRow(i);
			for(int ii = 0; ii < result.length;ii++) {
				mSE[ii] =  mSE[ii]+ Math.pow(result[ii] - target[ii],2)/inputs.length;
			}
		}
		for(int i = 0; i < result.length;i++) {
			result[i] = (float) mSE[i];
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

	/**
	 * 
	 * @param costFunctionType
	 * @param input
	 * @param target
	 * @return 
	 */
	public float[] calculateErrorPerNeuron(COST_FUNCTION_TYPE costFunctionType, float[] input, float[] target) {
		float[] result = mlp.predict(input);
		float[] error = new float[result.length];
		switch(costFunctionType) {
		case SQUARED_ERROR:			
			for(int i = 0;i < error.length;i++) {
				error[i] = (float) Math.pow(target[i]-result[i],2);
			}
			return error;
		default:
			System.err.println("Not implemented cost function:"+costFunctionType);
			break;
		}
		return new float[0];
	}
	
	/**
	 * Cost/Loss/Object function used to calculate back propagation of error
	 * @param costFunctionType
	 */
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
