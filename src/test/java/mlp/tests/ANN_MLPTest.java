package mlp.tests;

import static org.junit.Assert.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import math.utils.StatisticUtils;
import mlp.ANN_MLP;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.ANN_MLP.WEIGHT_INITIATION_METHOD;

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
	 * Test of prediction function of MLP with 3 input neurons, 4 hidden neurons
	 * and 3 output neurons. Weight Initiation uses default
	 * Under input of 10,10,10
	 * first hidden neuron:
	 * 	Net input:10*0.5 + 10*0.5 + 10*0.5 = 15
	 * 	Neuron output:f(15) = 0.99999938819
	 * first output neuron:
	 * 	Net input:0.99999938819*0.5 + 0.99999938819*0.5 + 0.99999938819*0.5 + 0.99999938819*0.5 = 3.99999755278
	 * 	Neuron output:f(3.99999755278) = 0.96402749362
	 */
	@Test
	void testOfPredictFunction1() {
		int[] layerSizes = {3,4,3};
		sut = new ANN_MLP(layerSizes);
		sut.setInitiationMethod(WEIGHT_INITIATION_METHOD.CONSTANT);
		sut.initiate();

		float[] input = {10,10,10};
		float[] expected = {0.964f,0.964f,0.964f};
		float[] actual = sut.predict(input);
		assertArrayEquals(actual,expected,0.01f);
		input = new float[]{10,10,10};
		expected = new float[]{0,0,0};
		actual = sut.predict(input);
		assertArrayEquals(actual,expected,0.01f);
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
		sut.setInitiationMethod(WEIGHT_INITIATION_METHOD.CONSTANT);
		sut.initiate();

		float[] input = new float[]{10,10,10};
		float[] expected = new float[]{0,0,0};
		float[] actual = sut.predict(input);
		assertArrayEquals(actual,expected,0.01f);
	}
	
	
	/**
	 * Integrational and unit tests
	 */
	
	@Test
	void constructorShallCreateMLPFromArrayOFLayerSizesAsArgument() {
		int[] actualLayerSizes = {2,3,1};
		sut = new ANN_MLP(actualLayerSizes);
		actualLayerSizes = sut.getLayerSizes();
		int[] expectedLayerSizes = {2,3,1};
		assertArrayEquals(actualLayerSizes,expectedLayerSizes);
	}
	
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
	 * Weights of first layer are 1 by default, the other weights 
	 * initiated according to schema 
	 */
	@Test
	void weightsMustBeInitiatedAccordingToConstantMethod() {
		int[] layerSizes = {2,3,2};
		sut = new ANN_MLP(layerSizes);
		sut.setInitiationMethod(ANN_MLP.WEIGHT_INITIATION_METHOD.CONSTANT);
		sut.initiate();
		float[] expectedLayerWeights1 = new float[layerSizes[0]*layerSizes[1]];
		for(int i = 0;i < expectedLayerWeights1.length;i++) {
			expectedLayerWeights1[i] = 1;
		}
		float[] expectedLayerWeights2 = new float[layerSizes[1]*layerSizes[2]];
		for(int i = 0;i < expectedLayerWeights2.length;i++) {
			expectedLayerWeights2[i] = sut.getWeightConstant();
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
		sut.setInitiationMethod(ANN_MLP.WEIGHT_INITIATION_METHOD.CONSTANT);
		ANN_MLP.WEIGHT_INITIATION_METHOD actual = sut.getInitiationMethod();
		assertEquals(expected,actual);
	}
	

	@Test
	void setterAndGetterOfWeightConstant() {
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		float expected = 0.7f;
		sut.setWeightConstant(0.7f);
		float actual = sut.getWeightConstant();
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
		float actual = sut.getWeightConstant();
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
		ACTIVATION_FUNCTION actual = sut.getActivationFunction();
		assertEquals(expected,actual);
	}
	
	/**
	 * Test random weight initiation
	 */
	void testRandomWeightInitiation(){
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		sut.setInitiationMethod(WEIGHT_INITIATION_METHOD.RANDOM);
		sut.initiate();
		int size = 0;
		float[][] weights = sut.getWeights();
		for(int i = 0;i <weights.length;i++) {
			size = size + weights[i].length;
		}
		float[] array = new float[size];
		int counter = 0;
		for(int layerIdx = 0;layerIdx <weights.length;layerIdx++) {
			for(int i = 0;i<weights[layerIdx].length;i++) {
				array[counter++] = weights[layerIdx][i];
			}
		}
		float variance = StatisticUtils.variance(array);
		assertTrue(variance>0);
	}

}
