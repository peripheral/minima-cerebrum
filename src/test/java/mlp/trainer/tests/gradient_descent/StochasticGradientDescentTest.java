package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANNMLP;
import mlp.NeuronFunctionModels;
import mlp.ANNMLP.ACTIVATION_FUNCTION;
import mlp.ANNMLP.WEIGHT_INITIATION_METHOD;
import mlp.trainer.Backpropagation.COST_FUNCTION_TYPE;
import mlp.trainer.data.TrainingData;
import mlp.trainer.gradient_descent.StochasticGradientDescent;

public class StochasticGradientDescentTest {
	private StochasticGradientDescent sut;

	@BeforeEach
	void init() {
		sut = new StochasticGradientDescent();
	}

	@Test
	void testRandomlyGenerateTrainingBatch() {
		int rows = 100;
		float[][] data = initiateData(rows);
		TrainingData td = new TrainingData();
		td.setInputs(data);
		sut.setTrainingData(td);
		int size = 20;
		TrainingData trainingBatch = sut.generateTrainingBatch(size);
		assertEquals(size,trainingBatch.size());
	}

	

	@Test
	void testSetLearningRate() {
		float learningRate = 0.01f;
		sut.setLearningRate(learningRate);
		float actual = sut.getLearningRate();
		assertEquals(learningRate,actual);
	}

	@Test
	void testGetDefaultLearningRate() {
		float learningRate = 0.001f;
		float actual = sut.getLearningRate();
		assertEquals(learningRate,actual);
	}
	


	/**
	 * Complementary to testRandomGenerateTrainingBatch
	 * @return
	 */
	private float[][] initiateData(int rows) {
		int inputSize = 3, outputSize = 3;
		float[][] data = new float[rows][inputSize + outputSize];
		int counter = 0, target = 0;
		Random rm = new Random();
		for(int row = 0;row < rows; row++) {
			for(int col = 0;col < inputSize;col++) {
				data[row][col] = counter; 
			}
			target = rm.nextInt(outputSize);
			for(int col = inputSize;col < inputSize + outputSize;col++) {
				if(col == inputSize+target) {
					data[row][col] = 1; 
				}else {
					data[row][col] = 0;
				}
			}
			counter++;
		}
		return data;
	}
	
	/**
	 * Load Iris Data set
	 * 
	 */
	

}
