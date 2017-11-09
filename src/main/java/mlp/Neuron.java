package mlp;

public class Neuron {
	
	private int netInput = 0;
	private float activationThreshold = 0;
	private float weight = 0;
	private int id;
	private float output;

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

	public void setWeight(double weight) {
		this.weight = (float)weight;
		
	}

	public float getWeight() {
		return weight;
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
