package mlp.tests;

import static org.junit.Assert.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANN_MLP;
import mlp.Neuron;
import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.Neuron.TRANSFER_FUNCTION;

public class NeuronTest {
	private Neuron sut;
	@BeforeEach
	void init() {
		sut = new Neuron();
	}

	@Test
	void testNetInputGet() {
		sut.setNetInput(5);
		float actualNet = sut.setGetNetInput();
		float expectedNet = 5.0f;
		assertTrue(Float.compare(expectedNet,actualNet) == 0);
	}
	
	@Test
	void testNetInputGetT2() {
		sut.setNetInput(1);
		float actualNet = sut.setGetNetInput();
		int expectedNet = 1;
		assertTrue(Float.compare(expectedNet,actualNet) == 0);
	}

	@Test
	void testActivationThresholdSetGet() {
		sut.setActivationThreshold(0.9);
		float actualActThreshold = sut.getActivationThreshold();    	
		assertTrue(Float.compare(0.9f,actualActThreshold) == 0);
	}

	@Test
	void testWeightSetGet() {
		sut.setWeight(0,0.9f);
		float actualActThreshold = sut.getWeight(0);    	
		assertTrue(Float.compare(0.9f,actualActThreshold) == 0);
	}
	
	@Test
	void testWeightSetGetT2() {
		int weightIdx = 1;
		sut.setWeight(weightIdx,3.9f);
		float actualActThreshold = sut.getWeight(weightIdx);    	
		assertTrue(Float.compare(3.9f,actualActThreshold) == 0);
	}

	@Test
	void testIdSetGet() {
		sut.setId(1);
		int expected = 1;
		int actual = sut.getId();    	
		assertEquals(expected,actual);
	}
	
	@Test
	void testIdSetGetT2() {
		sut.setId(2);
		int expected = 2;
		int actual = sut.getId();    	
		assertEquals(expected,actual);
	}
	
	@Test
	void testOutputSetGet() {
		sut.setOutput(2.0f);
		float expected = 2;
		float actual = sut.getOutput();    	
		assertTrue(Float.compare(actual, expected) == 0);
	}
	
	@Test
	void testSetGetActivationFunctionType() {
		sut.setActivationFunctionType(ANN_MLP.ACTIVATION_FUNCTION.SIGMOID);
		int actual = sut.getActivationFunctionType().ordinal();
		int expected = ANN_MLP.ACTIVATION_FUNCTION.SIGMOID.ordinal();
		assertEquals(expected,actual);
	}
	
	@Test
	void testConstructorWithActivationAsParam() {
		Neuron sut = new Neuron(ACTIVATION_FUNCTION.GAUSSIAN);
		int actual = sut.getActivationFunctionType().ordinal();
		int expected = ACTIVATION_FUNCTION.GAUSSIAN.ordinal();
		assertEquals(expected, actual);
	}
	
	@Test
	void testConstructorWithUpperLayerSizeAsParam() {
		int upperLayerSize = 5;
		Neuron sut = new Neuron(upperLayerSize);
		int actual = sut.getWeights().size();
		int expected = 5;
		assertEquals(expected, actual);
	}
	
	@Test
	void testConstructorWithActivationUpperLayerSizeAsParam() {
		Neuron sut = new Neuron(ACTIVATION_FUNCTION.GAUSSIAN);
		int actual = sut.getActivationFunctionType().ordinal();
		int expected = ACTIVATION_FUNCTION.GAUSSIAN.ordinal();
		assertEquals(expected, actual);
		int upperLayerSize = 5;
		sut = new Neuron(upperLayerSize);
		actual = sut.getWeights().size();
		expected = 5;
		assertEquals(expected, actual);
	}
	
	@Test
	void testNeuronActivationFunctionShouldReturn0() {
		Neuron sut = new Neuron(ACTIVATION_FUNCTION.SIGMOID);
		sut.setNetInput(0);
		//Symmetrical sigmoid activation function f(x) = B *(1-e^-Ax)/(1+e^-Ax)  with B = 1,A = 1, netinput = 0
		float actual = sut.activate();
		float expected = 0.0f;
		assertEquals(expected, actual,0.0f);
		actual = sut.activate();
		
	}
	
	@Test
	void testNeuronActivationFunctionShouldReturn1() {
		Neuron sut = new Neuron(ACTIVATION_FUNCTION.SIGMOID);
		sut.setNetInput(10);
		//Symmetrical sigmoid activation function f(x) = B *(1-e^-Ax)/(1+e^-Ax)  with B = 1,A = 1, netinput = 0
		float actual = sut.activate();
		float expected = 1.0f;
		assertEquals(expected, actual,0.1f);
		actual = sut.activate();
		
	}
	
	@Test
	void testSetterGetterTransferFunctionType() {
		Neuron sut = new Neuron();
		sut.setTransferFunction(TRANSFER_FUNCTION.IDENTITY);
		TRANSFER_FUNCTION actual = sut.getTransferFunction();
		TRANSFER_FUNCTION expected = TRANSFER_FUNCTION.IDENTITY;
		assertEquals(expected, actual);
	}
}
