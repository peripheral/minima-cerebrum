package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertEquals;


import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.trainer.TrainingData;
import mlp.trainer.gradient_descent.StochasticGradientDescent;

public class StochasticGradientDescentTest {
		private StochasticGradientDescent sut;

		@BeforeEach
		void init() {
			sut = new StochasticGradientDescent();
		}
		
		@Test
		void testRandomlyGenerateTrainingBatch() {
			float[][] data = initiateData();
			TrainingData td = new TrainingData();
			td.setData(data);
			sut.setTrainingData(td);
			int size = 20;
			TrainingData trainingBatch = sut.generateTrainingBatch(size);
			assertEquals(size,trainingBatch.size());
		}

		/**
		 * Complementary to testRandomGenerateTrainingBatch
		 * @return
		 */
		private float[][] initiateData() {
			int inputSize = 3, outputSize = 3;
			int rows = 100;
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

}
