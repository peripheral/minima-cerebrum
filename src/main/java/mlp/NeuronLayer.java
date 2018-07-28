package mlp;

import java.util.ArrayList;

import math.utils.StatisticUtils;
import mlp.ANN_MLP.LAYER_TYPE;

public class NeuronLayer {
	private ArrayList<Neuron> neurons = new ArrayList<>();
	private LAYER_TYPE layerType = LAYER_TYPE.HIDDEN;
	private Neuron bias = new Neuron();
	private boolean useSoftmaxFuction= false;

	public NeuronLayer() {}

	/**
	 * Produces number of neurons according to size and +1 bias with default net input of 1
	 * @param size
	 */
	public NeuronLayer(int size) {
		for(int i = 0; i < size;i++) {
			neurons.add(new Neuron());
		}
		bias.setNetInput(1);
	}

	public void addNeuron(Neuron n) {
		neurons.add(n);		
	}

	public Neuron getNeuron(int i) {
		if(neurons.size() == i) {
			return bias;
		}
		return 	neurons.get(i);
	}

	/**
	 * Returns size of layer, bias included
	 * @return
	 */
	public int size() {
		return neurons.size();
	}

	public float propagate(int neuronIdx) {
		float sum = 0;
		for(Neuron ne:neurons) {
			sum =sum + ne.getOutput()*ne.getWeight(neuronIdx);
		}
		return sum;
	}

	/**
	 * Returns all weights from the neurons. Weights are arranged in sequence per
	 * neuron. bias included last. Each neuron has connect to each neuron in consequent layer
	 */
	public float[] getWeights() {
		ArrayList<Float> weights = new ArrayList<>();
		for(Neuron n:neurons) {
			weights.addAll(n.getWeights());
		}
		weights.addAll(bias.getWeights());
		float[] weightsArr = new float[weights.size()];
		for(int i = 0; i < weights.size();i++) {
			weightsArr[i] = weights.get(i);
		}
		return weightsArr;				
	}

	/**
	 * Returns neuron list
	 * @return
	 */
	public ArrayList<Neuron> getNeurons() {
		return neurons;
	}

	public void execute() {
		for(Neuron n:neurons) {
			if(layerType != LAYER_TYPE.INPUT) {
				n.activate();
				n.transfer();
			}else {
				n.transferNoActivation();
			}
		}		
	}

	public float[] getOutputs() {
		float[] outputs = new float[neurons.size()];

		if(useSoftmaxFuction) {
			float[] inputs = new float[neurons.size()];
			for(int i = 0;i < inputs.length;i++) {
				inputs[i] = neurons.get(i).getNetInput();
			}
			outputs = StatisticUtils.calculateSoftmax(inputs);
		}else{
			for(int i = 0 ; i < outputs.length;i++) {
				outputs[i] = neurons.get(i).getOutput();
			}
		}
		return outputs;
	}

	public float[] getNetInputs() {
		float[] inputs = new float[neurons.size()];
		for(int i = 0 ; i < inputs.length;i++) {
			inputs[i] = neurons.get(i).getNetInput();
		}
		return inputs;
	}

	public void setLayerType(LAYER_TYPE input) {
		layerType = input;		
	}

	public void propagate(NeuronLayer l) {
		int neuronIdx = 0;
		for(Neuron n:l.getNeurons()) {
			n.setNetInput(propagate(neuronIdx));
			neuronIdx++;
		}
	}

	public Neuron getBiasNeuron() {
		return bias;
	}

	public void setUseSoftmaxOnOutput(boolean b) {
		useSoftmaxFuction = b;		
	}
	public boolean isSoftmaxUsedOnOutput() {
		return useSoftmaxFuction;		
	}

	public void setNeuronGradient(int neuronIdx, float nodeGradient) {
		neurons.get(neuronIdx).setNodeGradient(nodeGradient);
	}

	/** 
	 * Node gradients, bias excluded
	 * @return
	 */
	public float[] getNodeGradients() {
		float[] gradients = new float[neurons.size()];	
		for(int i = 0; i < gradients.length;i++) {
			gradients[i] = neurons.get(i).getNodeGradient(); 
		}
		return gradients;
	}

	public void setWeight(int weightIdx, float weight) {
		int neuronIdx = weightIdx/neurons.get(0).getWeights().size();
		int offset = weightIdx%neurons.get(0).getWeights().size();
		if(neuronIdx == neurons.size()) {
			bias.setWeight(offset, weight);
		}else {
			neurons.get(neuronIdx).setWeight(offset, weight);
		}

	}

}
