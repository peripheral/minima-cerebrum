package mlp.tests;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.ANNMLP.ACTIVATION_FUNCTION;
import mlp.NeuronFunctionModels;

public class ActivationFunctionModelTest {
	
	@BeforeEach
	void init() {
	
	}
	
	@Test
	void testDerivativeOfSigmoidMethod() {
		float expected = 0.5f;
		float a = 1;
		float b = 1;
		float netInput = 0;
		float actual = (float) NeuronFunctionModels.derivativeOf(ACTIVATION_FUNCTION.SIGMOID, a, b, netInput); 
		assertEquals(expected,actual,0.1f);
	}

}
