package mlp;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;
import mlp.Neuron.TRANSFER_FUNCTION;

public class NeuronFunctionModels {

	public static float activate(ACTIVATION_FUNCTION type,float a,float b,float x) {
		switch(type) {
		case SIGMOID:
			return (float) (b*(1-Math.pow(Math.E,-a*x))/(1+Math.pow(Math.E,-a*x)));
		default:
			System.err.println("Unnknown type");
			return -1;

		}
	}
	
	public static float derivativeOf(ACTIVATION_FUNCTION type,float a,float b,float x) {
		switch(type) {
		case SIGMOID:
			return (float) (b*a*2*(Math.pow(Math.E,-a*x))/Math.pow(1+Math.pow(Math.E,-a*x),2));
		default:
			System.err.println("Unnknown type");
			return -1;
		}
	}

	public static float transfer(TRANSFER_FUNCTION type,float value) {
		switch(type) {
		case IDENTITY:
			return value;
		default:
			System.err.println("Transferfunction:"+type+" not implented.");
			return 0;
		}
	}

}