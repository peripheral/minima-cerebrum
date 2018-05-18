package mlp;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class NeuronLayer {
	private ArrayList<Neuron> neurons = new ArrayList<>();
	
	public NeuronLayer() {}
	
	public NeuronLayer(int size,ANN_MLP.ACTIVATION_FUNCTION f) {
	}

	public NeuronLayer(int size) {
		for(int i = 0; i < size;i++) {
			neurons.add(new Neuron());
		}
	}

	public void addNeuron(Neuron n) {
		neurons.add(n);		
	}

	public Neuron getNeuron(int i) {
		return 	neurons.get(i);
	}

	/**
	 * Returns size of layer
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
	 * Returns all weights from the neurons
	 */
	public float[] getWeights() {
		ArrayList<Float> weights = new ArrayList<>();
		for(Neuron n:neurons) {
			weights.addAll(n.getWeights());
		}
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
}
