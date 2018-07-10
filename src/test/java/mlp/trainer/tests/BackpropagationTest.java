package mlp.trainer.tests;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;


import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;


import mlp.ANN_MLP;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.ANN_MLP.WEIGHT_INITIATION_METHOD;
import mlp.NeuronFunctionModels;
import mlp.trainer.Backpropagation;
import mlp.trainer.TrainingData;

public class BackpropagationTest {
	private Backpropagation sut;

	@BeforeEach
	void init() {
		sut = new Backpropagation();
	}

	/**
	 * Arbitrary data used as test calculation of Squared Error
	 */
	@Test
	void testCalculateMeanSquiredErrorPerNeuron() {
		int inputSize = 3,outputSize = 3,hiddenLayerSize = 4;
		float[][] data = {{10,10,10,1,0,0},
				{2,2,2,0,1,0},
				{30,30,30,0,0,1}};
		TrainingData td = new TrainingData(data);
		int[] mlp_topology = {inputSize,hiddenLayerSize,outputSize};
		ANN_MLP mlp = new ANN_MLP(mlp_topology);
		mlp.setActivationFunction(ACTIVATION_FUNCTION.SIGMOID);
		mlp.setWeightInitiationMethod(WEIGHT_INITIATION_METHOD.RANDOM);
		mlp.initiate();
		float[] expected = new float[outputSize];
		int[] layerSizes = mlp.getLayerSizes();
		float[][] weights = mlp.getWeights();
		float[] input;

		float[] output = null;
		for( int sampleIdx = 0; sampleIdx < data.length; sampleIdx++) {
			input = new float[layerSizes[0]];
			for( int col = 0; col < input.length; col++) {
				input[col] = data[sampleIdx][col];
			}

			for(int layerIdx = 0; layerIdx < weights.length; layerIdx++) {

				output = new float[layerSizes[layerIdx+1]];
				/* Each neuron of bottom layer contributes to each neuron in the top layer*/
				for(int colBotLayer = 0; colBotLayer < layerSizes[layerIdx];colBotLayer++) {
					/* Iteration for each neuron of the top layer */
					for(int colTopLayer = 0; colTopLayer < layerSizes[layerIdx+1]; colTopLayer++) {

						/* produce netinput to each neuron */
						output[colTopLayer] = output[colTopLayer] + input[colBotLayer]*
								weights[layerIdx][layerSizes[layerIdx+1]*colBotLayer + colTopLayer];
					}
				}
				/* Apply activation function		 */
				for(int i = 0; i < output.length; i++) {
					output[i] = NeuronFunctionModels.activate(mlp.getActivationFunction(),
							1, 1, output[i]);
				}
				input = output;
			}
	
			/* Mean Squared error */
			for(int i = 0; i < output.length;i++) {
				expected[i] = (float) (expected[i]+Math.pow(output[i] - 
						data[sampleIdx][data[sampleIdx].length-output.length+i],2)/data.length);
			}

		}
		sut.setMLP(mlp);
		sut.setTrainingData(td);
		float[] actual = sut.calCulateSquiedErrorPerNeuron();
		assertArrayEquals(expected,actual,0.0001f);
	}	
	

	/**
	 * Set approximate error minimum
	 */
	@Test
	void testSetGetApproximateErrorMinimum() {
		float errMinimum = 0.15f;
		sut.setApproximateErrorMinimum(errMinimum);
		float actual = sut.getApproximateErrorMinimum();
		assertEquals(errMinimum,actual);
	}
	
	/**
	 * Default approximate error minimum, stopping criteria
	 */
	@Test
	void testDefaultApproximateErrorMinimum() {
		float expected = 0.1f;
		float actual = sut.getApproximateErrorMinimum();
		assertEquals(expected,actual);
	}
}
