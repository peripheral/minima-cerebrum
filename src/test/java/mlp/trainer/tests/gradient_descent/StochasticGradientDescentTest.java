package mlp.trainer.tests.gradient_descent;

import static org.junit.jupiter.api.Assertions.assertEquals;


import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.NeuronFunctionModels;
import mlp.trainer.TrainingData;
import mlp.trainer.gradient_descent.StochasticGradientDescent;
import mlp.trainer.gradient_descent.StochasticGradientDescent.COST_FUNCTION_TYPE;

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
		 * Test for function that calculates gradient of input to error
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
			float actual = sut.calculateGradientInputOverError(COST_FUNCTION_TYPE.SQUARED_ERROR,Error,Io,a,b);
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
			float Io = 5.4f;
			float Error = 0.5f;
			/* Weight between lower layer neuron and upper neuron */
			float Wh = -0.04f;
			
			/* gradient sigmoid 2*(e^-5.4)/(1+e^-5.4)^2 = 0.00895211337 */
			float gradientSigmoidO = (float) (b*a*2*(Math.pow(Math.E,-a*Io))/Math.pow(1+Math.pow(Math.E,-a*Io),2));
			
			/* gradient sigmoid 2*(e^-6)/(1+e^-6)^2 = 0.00493301858 */
			float gradientSigmoidH = (float) (b*a*2*(Math.pow(Math.E,-a*Ih))/Math.pow(1+Math.pow(Math.E,-a*Ih),2));
		
			float expected = 2*Error * gradientSigmoidO * Wh * gradientSigmoidH ;
			/* arbitrary value, taken from previous test */
			float gradientInputToError = 0.00493301858f;
			
			float actual = sut.calculateGradientInputOverError(ACTIVATION_FUNCTION.SIGMOID,gradientInputToError,Wh,Ih);
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
