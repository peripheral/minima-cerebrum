package mlp.trainer.tests;

import static org.junit.Assert.assertArrayEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANN_MLP;
import mlp.trainer.Backpropagation;
import mlp.trainer.TrainingData;

public class BackpropagationTest {
	private Backpropagation sut;

	@BeforeEach
	void init() {
		sut = new Backpropagation();
	}

	void testCalculateSquiredErrorPerNeuron() {
		float[][] data = {{1,4,7},
				{2,5,8},
				{3,6,9}};
		TrainingData td = new TrainingData();
		int[] mlp_topology = {3,4,3};
		ANN_MLP mlp = new ANN_MLP(mlp_topology);
		mlp.initiate();
		mlp.setTrainingData(td);
		
	}	
}
