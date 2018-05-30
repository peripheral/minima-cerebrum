package mlp;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import mlp.ANN_MLP.LAYER_TYPE;

public class NeuronLayer {
	private ArrayList<Neuron> neurons = new ArrayList<>();
	private LAYER_TYPE layerType = LAYER_TYPE.HIDDEN;

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
			System.out.println("Weight index:"+neuronIdx+" weight:"+ne.getWeight(neuronIdx));
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

	public void execute() {
		for(Neuron n:neurons) {
			if(layerType != LAYER_TYPE.INPUT) {
				n.activate();
				n.transfer();
				System.out.println("LAYER type:"+layerType);
			}else {
				System.out.println("INPUT");
				n.transferNoActivation();
			}
		}		
	}

	public float[] getOutputs() {
		float[] outputs = new float[neurons.size()];
		for(int i = 0 ; i < outputs.length;i++) {
			outputs[i] = neurons.get(i).getOutput();
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

}
