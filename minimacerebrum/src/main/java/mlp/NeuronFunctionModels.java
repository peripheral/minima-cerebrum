package mlp;

import com.sun.corba.se.impl.orbutil.closure.Constant;

import mlp.ANNMLP.ACTIVATION_FUNCTION;
import mlp.Neuron.TRANSFER_FUNCTION;

public class NeuronFunctionModels {

	private NeuronFunctionModels() {}
	
	public static float activate(ACTIVATION_FUNCTION type,float a,float b,float x) {
		switch(type) {
		case SIGMOID:
			return (float) (b*(1-Math.pow(Math.E,-a*x))/(1+Math.pow(Math.E,-a*x)));
		default:
			System.err.println("Unimplemented activation function type:"+type);
			return -1;

		}
	}
	
	/**
	 * 
	 * @param type - function type
	 * @param a constant
	 * @param b constant
	 * @param x input
	 * @return
	 */
	public static float derivativeOf(ACTIVATION_FUNCTION type,float a,float b,float x) {
		switch(type) {
		case SIGMOID:
			return (float) (b*a*2*(Math.pow(Math.E,-a*x))/Math.pow(1+Math.pow(Math.E,-a*x),2));
		default:
			System.err.println("Unimplemented activation function type:"+type);
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

	/**
	 * params a = 1 and b = 1
	 * @param type
	 * @param x
	 * @return
	 */
	public static float derivativeOf(ACTIVATION_FUNCTION type, float x) {
		float a = 1,b = 1;
		switch(type) {
		case SIGMOID:
			return (float) (b*a*2*(Math.pow(Math.E,-a*x))/Math.pow(1+Math.pow(Math.E,-a*x),2));
		default:
			System.err.println("Unimplemented activation function type:"+type);
			return -1;
		}
	}
}
