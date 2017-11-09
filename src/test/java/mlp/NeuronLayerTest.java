package mlp;

import static org.junit.Assert.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class NeuronLayerTest {
	private NeuronLayer sut;
	@BeforeEach
	void init() {
		sut = new NeuronLayer();
	}
	
	@Test
	void testingLayerSize() {
		Neuron n = new Neuron();
		sut.addNeuron(n);
		int actualNet = sut.size();
		int expectedNet = 5;
		assertEquals(expectedNet,actualNet);
	}
	
	@Test
	void testingLayerSizeT2() {
		Neuron n = new Neuron();
		sut.addNeuron(n);
		int actualNet = sut.size();
		int expectedNet = 5;
		assertEquals(expectedNet,actualNet);
	}
	
	@Test
	void testingPrapagationFunction() {
		Neuron n = new Neuron();
		sut.addNeuron(n);
		int actualNet = sut.size();
		int expectedNet = 5;
		assertEquals(expectedNet,actualNet);
	}
	
}
