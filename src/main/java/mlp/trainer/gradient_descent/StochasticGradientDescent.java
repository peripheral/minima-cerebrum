package mlp.trainer.gradient_descent;

import java.util.Random;

import mlp.trainer.TrainingData;

public class StochasticGradientDescent {

	private TrainingData trainingData;

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
	
}
