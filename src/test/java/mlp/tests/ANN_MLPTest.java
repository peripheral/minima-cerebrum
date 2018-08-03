package mlp.tests;

import static org.junit.Assert.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import math.utils.StatisticUtils;
import mlp.ANN_MLP;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.ANN_MLP.WEIGHT_INITIATION_METHOD;
import mlp.trainer.Backpropagation;
import mlp.trainer.TrainingData;

public class ANN_MLPTest {
	private ANN_MLP sut;
	
	@BeforeEach
	void init() {
		sut = new ANN_MLP();
	}
	
	/**
	 * Functional tests
	 */
	
	/**
	 * Test of ANN_MLP constructor with argument as an array, the test 
	 * compares compares layer sizes provides and measured
	 */
	@Test
	void testConstructANNMLPFromArrayOFLayerSizesAsArgument() {
		int[] actualLayerSizes = {2,3,3};
		sut = new ANN_MLP(actualLayerSizes);
		actualLayerSizes = sut.getLayerSizes();
		int[] expectedLayerSizes = {2,3,3};
		assertArrayEquals(expectedLayerSizes,actualLayerSizes);
	}
	
	/**
	 * Test of prediction function of MLP with 3 input neurons, 4 hidden neurons
	 * and 3 output neurons. Weight Initiation uses default
	 * Under input of 10,10,10
	 * first hidden neuron:
	 * 	Net input:10*0.5 + 10*1 + 10*1 = 30
	 * 	Neuron output:f(30) = 1
	 * first output neuron:
	 * 	Net input:1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 = 2
	 * 	Neuron output:f(2) = 0.76159415595
	 */
	@Test
	void testOfPredictFunction1() {
		int[] layerSizes = {3,4,3};
		sut = new ANN_MLP(layerSizes);
		sut.setWeightInitiationMethod(WEIGHT_INITIATION_METHOD.CONSTANT);
		sut.initiate();

		float[] input = {10,10,10};
		float[] expected = {0.761f,0.761f,0.761f};
		float[] actual = sut.predict(input);
		assertArrayEquals(expected,actual,0.01f);
	}
	
	/**
	 * Test of prediction function of MLP with 3 input neurons, 4 hidden neurons
	 * and 3 output neurons. Weight Initiation uses default
	 * Under input of 0,0,0
	 * first hidden neuron:
	 * 	Net input:0
	 * 	Neuron output:f(0) = 0
	 * first output neuron:
	 * 	Net input:0
	 * 	Neuron output:f(0) = 0
	 */
	@Test
	void testOfPredictFunction2() {
		int[] layerSizes = {3,4,3};
		sut = new ANN_MLP(layerSizes);
		sut.setWeightInitiationMethod(WEIGHT_INITIATION_METHOD.CONSTANT);
		sut.initiate();

		float[] input = new float[]{0,0,0};
		float[] expected = new float[]{0,0,0};
		float[] actual = sut.predict(input);
		assertArrayEquals(expected,actual,0.01f);
	}
	
	/**
	 * Test of prediction function of MLP with 3 input neurons, 4 hidden neurons
	 * and 3 output neurons. Weight Initiation uses default
	 * Sigmoid as activation for hidden and softmax as activation for output layer
	 * Under input of 0,0,0
	 * first hidden neuron:
	 * 	Net input:0
	 * 	Neuron output:f(0) = 0
	 * first output neuron:
	 * 	Net input:0
	 * 	Neuron output with softmax (e^0)/((e^0)*3) = 1/3 = 0.333
	 */
	@Test
	void testOfPredictFunctionWithSoftmax() {
		int[] layerSizes = {3,4,3};
		sut = new ANN_MLP(WEIGHT_INITIATION_METHOD.CONSTANT,layerSizes);
		boolean b = true;
		sut.setUseSoftmaxOnOutput(b);
		sut.initiate();

		float[] input = new float[]{0,0,0};
		float[] expected = new float[]{0.333f,0.333f,0.333f};
		float[] actual = sut.predict(input);
		assertArrayEquals(expected,actual,0.01f);
	}
	
	/**
	 * Test of prediction function of MLP with 3 input neurons, 4 hidden neurons
	 * and 3 output neurons. Weight Initiation uses default
	 * Sigmoid as activation for hidden and softmax as activation for output layer
	 * Under input of 10,10,10
	 * first hidden neuron:
	 * 	Net input:10*0.5 + 10*1 + 10*1 = 30
	 * 	Neuron output:f(30) = 1 , f() - sigmoid
	 * first output neuron:
	 * 	Net input:1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 = 2
	 * 	Neuron output:(e^(2))/((e^(2))*3) = 0.333.. , f() - Softmax
	 */
	@Test
	void testOfPredictFunctionWithSoftmax2() {
		int[] layerSizes = {3,4,3};
		sut = new ANN_MLP(WEIGHT_INITIATION_METHOD.CONSTANT,layerSizes);
		boolean b = true;
		sut.setUseSoftmaxOnOutput(b);
		sut.initiate();

		float[] input = new float[]{10,10,10};
		float[] expected = new float[]{0.333f,0.333f,0.333f};
		float[] actual = sut.predict(input);
		assertArrayEquals(expected,actual,0.01f);
	}
	
	/**
	 * Integration and unit tests
	 */
	
	
	@Test
	void getInputOutputLayerShouldReturnExpectedLayerSizes() {
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		int[] actualInputLayerSizes = new int[3];
		actualInputLayerSizes[0] = sut.getInputLayer().size();
		actualInputLayerSizes[1] = sut.getLayer(1).size();
		actualInputLayerSizes[2] = sut.getOutputLayer().size();
		int[] expected =  {2,3,1};
		assertArrayEquals(expected,actualInputLayerSizes);
	}
	
	/**
	 * Weights initiated according to constant value schema where biases were given 0 weight and input 1 . 
	 */
	@Test
	void weightsMustBeInitiatedAccordingToConstantMethod() {
		int[] layerSizes = {2,3,2};
		sut = new ANN_MLP(layerSizes);
		sut.setWeightInitiationMethod(ANN_MLP.WEIGHT_INITIATION_METHOD.CONSTANT);
		sut.initiate();
		float[] expectedLayerWeights1 = new float[(layerSizes[0]+1)*layerSizes[1]];
		for(int i = 0;i < expectedLayerWeights1.length - layerSizes[1];i++) {
			expectedLayerWeights1[i] = sut.getWeightInitiationConstant();
		}
		for(int i = expectedLayerWeights1.length - layerSizes[1];i < expectedLayerWeights1.length;i++) {
			expectedLayerWeights1[i] = 0;
		}
		float[] expectedLayerWeights2 = new float[(layerSizes[1]+1)*layerSizes[2]];
		for(int i = 0;i < expectedLayerWeights2.length - layerSizes[2];i++) {
			expectedLayerWeights2[i] = sut.getWeightInitiationConstant();
		}
		for(int i =  expectedLayerWeights2.length - layerSizes[2];i < expectedLayerWeights2.length;i++) {
			expectedLayerWeights2[i] = 0;
		}
		float[][] actual = sut.getWeights();
		assertArrayEquals(expectedLayerWeights1,actual[0],0.01f);
		assertArrayEquals(expectedLayerWeights2,actual[1],0.01f);	
	}
	
	@Test
	void setGetInitiationMethod() {
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		ANN_MLP.WEIGHT_INITIATION_METHOD expected = ANN_MLP.WEIGHT_INITIATION_METHOD.CONSTANT;
		sut.setWeightInitiationMethod(ANN_MLP.WEIGHT_INITIATION_METHOD.CONSTANT);
		ANN_MLP.WEIGHT_INITIATION_METHOD actual = sut.getWeightInitiationMethod();
		assertEquals(expected,actual);
	}
	

	@Test
	void setterAndGetterOfWeightConstant() {
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		float expected = 0.7f;
		sut.setWeightInititiationConstant(0.7f);
		float actual = sut.getWeightInitiationConstant();
		assertEquals(expected,actual,0.0f);
	}
	
	/**
	 * Default weight constant 0.5f
	 */
	@Test
	void testGetDefaultWeightConstant() {
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		float expected = 0.5f;
		float actual = sut.getWeightInitiationConstant();
		assertEquals(expected,actual,0.0f);
	}
	
	/**
	 * Test setter and getter for activation function
	 */
	@Test
	void testSetGetActivationFunction(){
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		ACTIVATION_FUNCTION expected = ACTIVATION_FUNCTION.IDENTITY;
		sut.setActivationFunction(ACTIVATION_FUNCTION.IDENTITY);
		ACTIVATION_FUNCTION actual = sut.getActivationFunctionType();
		assertEquals(expected,actual);
	}
	
	/**
	 * Test random weight initiation
	 */
	@Test
	void testRandomWeightInitiation(){
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		sut.setWeightInitiationMethod(WEIGHT_INITIATION_METHOD.RANDOM);
		sut.initiate();
		/* size -Number of all Weights in the network */
		int size = 0;
		/*Weights from the networks */
		float[][] weights = sut.getWeights();
		/* Count number of weights in the network and store 
		 * in variable size */
		for(int layerIdx = 0;layerIdx <weights.length;layerIdx++) {
			size = size + weights[layerIdx].length;
		}
		/* Converting to one dimensional array before calculating variance */
		float[] array = new float[size];
		int counter = 0;
		for(int layerIdx = 0;layerIdx <weights.length;layerIdx++) {
			for(int column = 0;column<weights[layerIdx].length;column++) {
				array[counter++] = weights[layerIdx][column];
			}			
		}
		float variance = StatisticUtils.variance(array);
		assertTrue(variance>0);
	}
	
	/**
	 * Test set get TestData
	 */
	@Test
	void testSetGetTestDataMethods() {
		TrainingData expected = new TrainingData();
		sut.setTrainingData(expected);
		TrainingData actual = sut.getTrainingData();
		assertEquals(expected,actual);
	}
	
	/**
	 * Test for setter and getter of Backpropagation trainer
	 */
	@Test
	void testSetGetBackpropagationTrainer() {
		Backpropagation expected = new Backpropagation();
		sut.setBackpropagationTrainer(expected);
		Backpropagation actual = sut.getBackpropagationTrainer();
		assertEquals(expected,actual);
	}
	
	/**
	 * ANN_MLP should return Backpropagation trainer as default
	 */
	@Test
	void testGetBackpropagationTrainerShouldNotReturnNull() {
		Backpropagation actual = sut.getBackpropagationTrainer();
		assertTrue(actual != null);
	}
	
	/**
	 * Testing set weight function
	 */
	@Test
	void testSetWeightFunction() {
		int[] layerSizes = {3,4,3};
		sut = new ANN_MLP(layerSizes);
		sut.initiate();
		float[][] expected = getTestWeights();
		sut.setWeights(getTestWeights());
		float[][] actualWeights = sut.getWeights();
		assertArrayEquals(expected,actualWeights);
	}
	
	private float[][] getTestWeights() {
		float[][] weights = new float[2][];
		//Initial LR:0.01
		weights[0] = new float[] {0.26172805f, -0.023212755f, 0.3237994f, 0.33701196f, -0.027228683f, 0.09190323f, -0.00764591f, 
				0.2830995f, 0.27917573f, -0.18975836f, -0.3373269f, 0.32033712f, -0.15263462f, 0.17369014f, 0.20265998f, -0.115796775f};
		weights[1] = new float[]{0.04440312f, 0.09132995f, -0.14755417f, 0.28749457f, -0.26251328f, 0.26934755f, 0.34488472f, 0.025896106f, -0.30336866f,
				0.04440658f, 0.1995952f, 0.23860021f, -0.019049816f, 0.13698725f, 0.043328285f};
		//After LR:-0.01
		return weights;
	}
}
