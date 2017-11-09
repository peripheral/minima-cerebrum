package mlp;

import static org.junit.Assert.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class NeuronTest {
	private Neuron sut;
	@BeforeEach
	void init() {
		sut = new Neuron();
	}

	@Test
	void testingNetInputGet() {
		sut.setNetInput(5);
		int actualNet = sut.setGetNetInput();
		int expectedNet = 5;
		assertEquals(expectedNet,actualNet);
	}
	
	@Test
	void testingNetInputGetT1() {
		sut.setNetInput(1);
		int actualNet = sut.setGetNetInput();
		int expectedNet = 1;
		assertEquals(expectedNet,actualNet);
	}

	@Test
	void testingActivationThresholdSetGet() {
		sut.setNetInput(5);
		sut.setActivationThreshold(0.9);
		float actualActThreshold = sut.getActivationThreshold();    	
		assertTrue(Float.compare(0.9f,actualActThreshold) == 0);
	}

	@Test
	void testingWeightSetGet() {
		sut.setWeight(0.9f,0);
		float actualActThreshold = sut.getWeight(0);    	
		assertTrue(Float.compare(0.9f,actualActThreshold) == 0);
	}

	@Test
	void testingIdSetGet() {
		sut.setId(1);
		int expected = 1;
		int actual = sut.getId();    	
		assertEquals(expected,actual);
	}
	
	@Test
	void testingIdSetGetT2() {
		sut.setId(2);
		int expected = 2;
		int actual = sut.getId();    	
		assertEquals(expected,actual);
	}
	
	@Test
	void testingOutputSetGet() {
		sut.setOutput(2.0f);
		float expected = 2;
		float actual = sut.getOutput();    	
		assertTrue(Float.compare(actual, expected) == 0);
	}
	

}
