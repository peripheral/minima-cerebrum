package mlp;

import java.util.ArrayList;
import math.utils.StatisticUtils;
import mlp.trainer.Backpropagation;
import mlp.trainer.TrainingData;

public class ANN_MLP {

	public static enum WEIGHT_INITIATION_METHOD{CONSTANT, RANDOM};
	public static enum ACTIVATION_FUNCTION{SIGMOID, GAUSSIAN, IDENTITY};
	public static enum LAYER_TYPE{INPUT,HIDDEN,OUTPUT}
	private NeuronLayer[] layers = null;
	public WEIGHT_INITIATION_METHOD DEFAULT_WEIGHT_INITIATION_METHOD = WEIGHT_INITIATION_METHOD.CONSTANT;
	public float DEFAULT_WEIGHT_CONSTANT = 0.5f;
	private float WEIGHT_CONSTANT = DEFAULT_WEIGHT_CONSTANT;
	private WEIGHT_INITIATION_METHOD initiationMethod;
	private ACTIVATION_FUNCTION activationFunction = ACTIVATION_FUNCTION.SIGMOID;
	private TrainingData trainingData;
	private Backpropagation trainer = new Backpropagation();
	private boolean applySoftmaxOnOutput = false;
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



	public ANN_MLP(WEIGHT_INITIATION_METHOD weightInitiationMethod, int[] layerSizes) {
		this(layerSizes);
		initiationMethod = weightInitiationMethod;
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

	/**
	 * Specifies which 
	 * @param type
	 */
	public void setWeightInitiationMethod(WEIGHT_INITIATION_METHOD type) {
		initiationMethod = type;		
	}

	public WEIGHT_INITIATION_METHOD getWeightInitiationMethod() {
		return initiationMethod;
	}

	/**
	 * Methods returns weights per layer.
	 * weights arranged in sequence per neuron
	 * @return
	 */
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
	 * Initiates weights, default is the constant value
	 */
	public void initiate() {
		switch(initiationMethod) {
		case CONSTANT:
			initiateMethodConstant();
			break;
		case RANDOM:
			initiateMethodRandom();
			break;
		default:
			break;
		}
	}

	private void initiateMethodRandom() {
		ArrayList<Neuron> neurons;
		Neuron neuron;
		/* For every layer */
		for(int layerIdx = 0;layerIdx < layers.length-1;layerIdx++ ) {
			layers[layerIdx].getBiasNeuron().setNetInput(1);
			/*Get neurons from layer */
			neurons = layers[layerIdx].getNeurons();
			/* Layer is not input and not output layer*/
			//if(layerIdx >0 && layerIdx < layers.length - 1) {
			/* For each neuron in layer make a weight for each neuron it is connected to */
			for(int neuronIdx = 0; neuronIdx < neurons.size();neuronIdx++ ) {
				neuron = neurons.get(neuronIdx);
				for(int i = 0;i < layers[layerIdx+1].size();i++) {
					neuron.setWeight(i,StatisticUtils.getXavierRandomWeight(layers[layerIdx].size(),
							(layerIdx < layers.length ? layers[layerIdx+1].size():0)));
				}
				for(int i = 0;i < layers[layerIdx+1].size();i++) {
					layers[layerIdx].getBiasNeuron().setWeight(i,StatisticUtils.getXavierRandomWeight(layers[layerIdx].size(),
							(layerIdx < layers.length ? layers[layerIdx+1].size():0)));
				}
			}
			//}

			/*If layer is input layer set weight to one */
			/*	else if(layerIdx == 0){
				for(int neuronIdx = 0; neuronIdx < neurons.size();neuronIdx++ ) {
					neuron = neurons.get(neuronIdx);
					for(int i = 0;i < layers[layerIdx+1].size();i++) {
						neuron.setWeight(i,1);
					}
				}
				neuron = layers[layerIdx].getBiasNeuron();
				for(int i = 0;i < layers[layerIdx+1].size();i++) {
					neuron.setWeight(i, 0);
				}				
			}*/
		}

	}

	/**
	 * 
	 */
	private void initiateMethodConstant() {
		ArrayList<Neuron> neurons;
		Neuron neuron;
		/* For every layer */
		for(int layerIdx = 0;layerIdx < layers.length-1;layerIdx++ ) {
			/*Get neurons from layer */
			neurons = layers[layerIdx].getNeurons();
			layers[layerIdx].getBiasNeuron().setNetInput(1);
			/* For each neuron in layer make a weight for each neuron it is connected to */
			for(int neuronIdx = 0; neuronIdx < neurons.size();neuronIdx++ ) {
				neuron = neurons.get(neuronIdx);
				for(int i = 0;i < layers[layerIdx+1].size();i++) {
					neuron.setWeight(i,WEIGHT_CONSTANT);
				}
			}
			for(int i = 0;i < layers[layerIdx+1].size();i++) {
				layers[layerIdx].getBiasNeuron().setWeight(i,0);
			}
		}
	}

	/**
	 * Specifies a value to be used as weight at initiation procedure which sets identical 
	 * value for all weights
	 * @param f - value used as weight 
	 */
	public void setWeightInititiationConstant(float f) {
		WEIGHT_CONSTANT = f;		
	}

	/**
	 * The default weight constant is 0.5f
	 * @return
	 */
	public float getWeightInitiationConstant() {
		return WEIGHT_CONSTANT;		
	}

	public void setActivationFunction(ACTIVATION_FUNCTION type) {
		activationFunction = type;		
	}

	public ACTIVATION_FUNCTION getActivationFunctionType() {
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
			previous = l;
		}
		return layers[layers.length-1].getOutputs();
	}

	public void setTrainingData(TrainingData td) {
		trainingData = td;

	}

	public TrainingData getTrainingData() {
		return trainingData;
	}

	public void setBackpropagationTrainer(Backpropagation tr) {
		trainer = tr;
		
	}

	public Backpropagation getBackpropagationTrainer() {
		return trainer;
	}

	public void setUseSoftmaxOnOutput(boolean b) {
		layers[layers.length-1].setUseSoftmaxOnOutput(b);
		applySoftmaxOnOutput = b;
	}
	
	public boolean isSoftmaxAppliedOnOutput() {
		return applySoftmaxOnOutput;
	}

}
