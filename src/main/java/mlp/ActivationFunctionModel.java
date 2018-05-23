package mlp;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;

public class ActivationFunctionModel {

	public static float activate(ACTIVATION_FUNCTION type,float a,float b,float x) {
		switch(type) {
		case SIGMOID:
			return (float) (b*(1-Math.pow(Math.E,-a*x))/(1+Math.pow(Math.E,-a*x)));
		default:
			System.err.println("Unnknown type");
			return -1;

		}
	}

}
