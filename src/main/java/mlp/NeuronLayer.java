package mlp;

import java.util.ArrayList;

public class NeuronLayer {
	private ArrayList<Neuron> neurons = new ArrayList<>();

	public void addNeuron(Neuron n) {
		neurons.add(n);		
	}

	public Neuron getNeuron(int i) {
		return 	neurons.get(i);
	}

	public int size() {
		return neurons.size();
	}

}
