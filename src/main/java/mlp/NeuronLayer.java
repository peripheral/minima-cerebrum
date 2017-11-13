package mlp;

import java.util.ArrayList;

public class NeuronLayer {
	private ArrayList<Neuron> neurons = new ArrayList<>();
	
	
	public NeuronLayer(int size,ANN_MLP.ACTIVATION_FUNCTION f) {
	}

	public NeuronLayer(int size) {
	}

	public NeuronLayer() {
	}

	public void addNeuron(Neuron n) {
		neurons.add(n);		
	}

	public Neuron getNeuron(int i) {
		return 	neurons.get(i);
	}

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

}
