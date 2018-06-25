package mlp.trainer;

import java.util.Arrays;

import mlp.ANN_MLP;

public class Backpropagation {

	private ANN_MLP mlp;
	private TrainingData trainingData;

	public void setMLP(ANN_MLP mlp) {
		this.mlp = mlp;
		
	}

	public void setTrainingData(TrainingData td) {
		this.trainingData = td;
	}

	public float[] calCulateSquiedErrorPerNeuron() {
		double[] MSE = new double[mlp.getOutputLayer().size()];
		float[][] data = trainingData.getData();
		float[] result = null;
		float[] target;

		for(int i = 0; i < data.length;i++) {
			result = mlp.predict(Arrays.copyOf(data[i],mlp.getInputLayer().size()));
			target =  trainingData.getTargetData(i,data[i].length-result.length);
			for(int ii = 0; ii < result.length;ii++) {
				MSE[ii] =  MSE[ii]+ Math.pow(result[ii] - target[ii],2)/data.length;
			}
		}
		for(int i = 0; i < result.length;i++) {
			result[i] = (float) MSE[i];
		}
		
		return result;
	}


}
