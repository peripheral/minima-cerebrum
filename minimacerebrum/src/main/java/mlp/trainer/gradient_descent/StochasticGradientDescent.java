package mlp.trainer.gradient_descent;

import java.util.Random;

import mlp.ANNMLP.ACTIVATION_FUNCTION;
import mlp.NeuronFunctionModels;
import mlp.trainer.Backpropagation;
import mlp.trainer.data.TrainingData;

public class StochasticGradientDescent extends GradientDescent {

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
			data[i] = trainingData.getInputRow(rm.nextInt(trainingData.size()));
		}
		td.setInputs(data);
		return td;
	}
	
}
