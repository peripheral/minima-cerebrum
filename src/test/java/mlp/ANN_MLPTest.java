package mlp;

import static org.junit.Assert.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;

public class ANN_MLPTest {
	private ANN_MLP sut;
	
	@BeforeEach
	void init() {
		sut = new ANN_MLP();
	}
	
	@Test
	void constructorShallCreateMLPWithArrayOFLayerSizesAsArgument() {
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

}
