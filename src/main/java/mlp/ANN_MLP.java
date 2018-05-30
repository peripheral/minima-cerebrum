package mlp;

import java.util.ArrayList;
import java.util.Arrays;

public class ANN_MLP {

	public static enum WEIGHT_INITIATION_METHOD{CONSTANT, RANDOM};
	public static enum ACTIVATION_FUNCTION{SIGMOID, GAUSSIAN, IDENTITY};
	public static enum LAYER_TYPE{INPUT,HIDDEN,OUTPUT}
	private NeuronLayer[] layers = null;
	public WEIGHT_INITIATION_METHOD DEFAULT_WEIGHT_INITIATION_METHOD = WEIGHT_INITIATION_METHOD.CONSTANT;
	public float DEFAULT_WEIGHT_CONSTANT = 0.5f;
	private float WEIGHT_CONSTANT = DEFAULT_WEIGHT_CONSTANT;
	private WEIGHT_INITIATION_METHOD INITIATION_METHOD;
	private ACTIVATION_FUNCTION activationFunction;
	public ANN_MLP() {}

	public ANN_MLP(int[] layerSizes) {
		layers = new NeuronLayer[layerSizes.length];
		layers[0] = new NeuronLayer(layerSizes[0]);
		for(int i = 1;i < layerSizes.length-1;i++) {
			layers[i] = new NeuronLayer(layerSizes[i]);
		}
		layers[0].setLayerType(LAYER_TYPE.INPUT);
		layers[layers.length-1] = new NeuronLayer(layerSizes[layers.length-1]);
		layers[layers.length-1].setLayerType(LAYER_TYPE.OUTPUT);
	}



	public int[] getLayerSizes() {
		int[] layerSizes = new int[layers.length];
		for(int i = 0; i < layerSizes.length;i++) {
			layerSizes[i] = layers[i].size();
		}
		return layerSizes;
	}

	public NeuronLayer getInputLayer() {
		return layers[0];
	}

	public NeuronLayer getLayer(int i) {
		return layers[i];
	}

	public NeuronLayer getOutputLayer() {
		return layers[layers.length-1];
	}

	public void setInitiationMethod(WEIGHT_INITIATION_METHOD constant) {
		INITIATION_METHOD = constant;		
	}

	public WEIGHT_INITIATION_METHOD getInitiationMethod() {
		return INITIATION_METHOD;
	}

	public float[][] getWeights() {
		float[][] weights;
		if(layers.length >0) {
			weights = new float[layers.length-1][];
			for(int i = 0 ; i < weights.length;i++) {
				weights[i] = layers[i].getWeights();
			}
			return weights;
		}else {
			return null;
		}
	}

	/**
	 * Initiates weights
	 */
	public void initiate() {
		switch(INITIATION_METHOD) {
		case CONSTANT:
			initiateMethodConstant();
			break;
		}
	}

	/**
	 * 
	 */
	private void initiateMethodConstant() {
		ArrayList<Neuron> neurons;
		Neuron neuron;
		for(int layerIdx = 0;layerIdx < layers.length-1;layerIdx++ ) {
			neurons = layers[layerIdx].getNeurons();
			if(layerIdx >0) {
				for(int neuronIdx = 0; neuronIdx < neurons.size();neuronIdx++ ) {
					neuron = neurons.get(neuronIdx);
					for(int i = 0;i < layers[layerIdx+1].size();i++) {
						neuron.setWeight(i,WEIGHT_CONSTANT);
					}
				}
			}else {
				for(int neuronIdx = 0; neuronIdx < neurons.size();neuronIdx++ ) {
					neuron = neurons.get(neuronIdx);
					for(int i = 0;i < layers[layerIdx+1].size();i++) {
						neuron.setWeight(i,1);
					}
				} 
			}
		}
	}

	public void setWeightConstant(float f) {
		WEIGHT_CONSTANT = f;		
	}

	/**
	 * The default weight constant is 0.5f
	 * @return
	 */
	public float getWeightConstant() {
		return WEIGHT_CONSTANT;		
	}

	public void setActivationFunction(ACTIVATION_FUNCTION type) {
		activationFunction = type;		
	}

	public ACTIVATION_FUNCTION getActivationFunction() {
		return activationFunction;
	}

	public float[] predict(float[] input) {
		ArrayList<Neuron> neuronList = layers[0].getNeurons();
		if(input.length !=neuronList.size()) {
			System.err.println("Incorrect number of inputs! Expected:"+neuronList.size());
			return null;
		}
		for(int i = 0;i < input.length;i++) {
			neuronList.get(i).setNetInput(input[i]);
		}
		NeuronLayer previous = null;
		for(NeuronLayer l:layers) {
			if(previous != null) {
				previous.propagate(l);
			}
			l.execute();
			System.out.println(Arrays.toString(l.getNetInputs()));
			System.out.println(Arrays.toString(l.getOutputs()));
			previous = l;
		}
		return layers[layers.length-1].getOutputs();
	}

}
