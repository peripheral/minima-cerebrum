package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANN_MLP;
import mlp.NeuronFunctionModels;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.ANN_MLP.WEIGHT_INITIATION_METHOD;
import mlp.trainer.Backpropagation.COST_FUNCTION_TYPE;
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
		int rows = 100;
		float[][] data = initiateData(rows);
		TrainingData td = new TrainingData();
		td.setData(data);
		sut.setTrainingData(td);
		int size = 20;
		TrainingData trainingBatch = sut.generateTrainingBatch(size);
		assertEquals(size,trainingBatch.size());
	}

	/**
	 * expected = 0.035 + 0.001 *0.035 + 0.01 * 0.05 = 0.035535
	 */
	@Test
	void testCalculateNewWeightFunction() {
		float momentum = 0.001f;
		float learningRate = 0.01f;
		float oldWeight = 0.035f;
		float deltaW = 0.05f;
		float expected = 0.035535f;
		float actual = sut.generateNewWeight(momentum,learningRate,oldWeight,deltaW);
		assertEquals(expected,actual);
	}

	/**
	 * Test for function that returns error for specified input and expected target
	 */
	@Test
	void testCalculateErrorForInputAndTargetRow() {	
		int rowIdx = 50;
		float[][] data = initiateData(100);
		int outputLayerSize = 3;
		int[] layerSizes = {3,4,outputLayerSize};

		TrainingData td = new TrainingData(data,outputLayerSize);
		sut.setTrainingData(td);
		ANN_MLP mlp = new ANN_MLP(WEIGHT_INITIATION_METHOD.RANDOM,layerSizes);
		mlp.initiate();
		sut.setMLP(mlp);
		sut.calculateError();
		
		float[][] weights = mlp.getWeights();
		float[] input = td.getInputRow(rowIdx);
		float[] target = td.getTargetRow(rowIdx);
		float[] output = null;

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
				output[i] = NeuronFunctionModels.activate(mlp.getActivationFunctionType(),
						1, 1, output[i]);
			}
			input = output;
		}
		float[] expected = new float[output.length];
		for(int i = 0; i < output.length;i++) {
			expected[i] = (float) Math.pow(output[i] -target[i],2);
		}
	
		float[] actual = sut.calculateErrorPerNeuron(COST_FUNCTION_TYPE.SQUARED_ERROR,td.getInputRow(rowIdx),td.getTargetRow(rowIdx));
		assertArrayEquals(expected,actual);
	}
	
	

	/**
	 * Test for function that calculates gradient of input to error output layer
	 * of the output neuron
	 * Io - neuron input, E = (Predicted - Required)
	 * ∂(E)^2/∂I_o = gradient
	 * (∂E^2/∂I_o) => 2E  - first step of derivation
	 * (∂f(I_o)/∂I_o) => f'(Io), f(.) - sigmoid 
	 * 2E * fo'(Io)
	 */
	@Test
	void testCalculateGradientInputOverError() {		
		float a = 1; /* activation function parameter*/
		float b = 1; /* activation function parameter*/
		float Io = 6f;		
		float Error = 0.5f;	
		/* gradient sigmoid 2*(e^-6)/(1+e^-6)^2 = 0.00493301858 */
		float gradientSigmoid = (float) (b*a*2*(Math.pow(Math.E,-a*Io))/Math.pow(1+Math.pow(Math.E,-a*Io),2));	
		/* expected 2E * Fo'(Io) =  2*0.5 * 0.00493301858 = 0.00493301858 */
		float expected = 2*Error * gradientSigmoid;
		float actual = sut.calculateGradientInputOverError(COST_FUNCTION_TYPE.SQUARED_ERROR,ACTIVATION_FUNCTION.SIGMOID,Error,Io,a,b);
		assertEquals(expected,actual);
	}

	/**
	 * ∂(E)^2/I_h = (∂E^2/∂I_o)(∂I_o/∂O_h)(∂O_h/∂I_h)
	 *  (∂E^2/∂I_o) => 2E * fo'(I_o),  Fo(.) - output function 
	 *  (∂I_o/∂O_h) => Wh, 
	 *  (∂O_h/∂I_h) => fh'(I_h)  , fh(.) - hidden neuron activation function, partial derivative of
	 *  output of hidden neuron over input of hidden neuron
	 * h - hidden neuron, o - output of connecting neuron
	 *  2E * f'(I_o) * Wh * fh'(I_h) =  gradient  input of hidden neuron over Error 
	 */
	@Test
	void testCalculateGradientInputToHiddenNeuronOverError() {
		float a = 1;
		float b = 1;
		float Ih = 6f; /* Input of a hidden neuron*/

		/* Weight between lower layer neuron and upper neuron */
		float Wh = -0.04f;

		/* gradient sigmoid 2*(e^-6)/(1+e^-6)^2 = 0.00493301858 */
		float gradientSigmoidH = (float) (b*a*2*(Math.pow(Math.E,-a*Ih))/Math.pow(1+Math.pow(Math.E,-a*Ih),2));
		/* arbitrary value, taken from previous test */
		float gradientInputToError = 0.00493301858f;
		float expected = gradientInputToError * Wh * gradientSigmoidH ;


		float actual = sut.calculateGradientInputOverError(ACTIVATION_FUNCTION.SIGMOID,gradientInputToError,Wh,Ih,a,b);
		assertEquals(expected,actual);
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
		float learningRate = 0.1f;
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

}
