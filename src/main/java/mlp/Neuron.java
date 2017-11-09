package mlp;

import java.util.ArrayList;

public class Neuron {
	
	private int netInput = 0;
	private float activationThreshold = 0;
	private float weight = 0;
	private int id;
	private float output;
	private ArrayList<Float> weights = new ArrayList<Float>();

	public void setNetInput(int i) {
		netInput = i;
		
	}

	public int setGetNetInput() {
		return netInput;
	}

	public void setActivationThreshold(double threshold) {
		activationThreshold = (float)threshold;
	}

	public float getActivationThreshold() {		
		return activationThreshold;
	}

	public void setWeight(double weight, int i) {
		//this.weights.set(i,(float)weight);
		
	}

	public float getWeight(int n) {
	//	return weight;
		return 0.9f;
	}

	public void setId(int id) {
		this.id = id;		
	}

	public int getId() {
		return id;
	}

	public void setOutput(float output) {
		this.output = output;		
	}

	public float getOutput() {		
		return output;
	}
	
}
