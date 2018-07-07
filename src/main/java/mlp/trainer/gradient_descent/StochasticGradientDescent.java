package mlp.trainer.gradient_descent;

import java.util.Random;

import mlp.trainer.TrainingData;

public class StochasticGradientDescent {

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
	
}
