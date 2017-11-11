package mlp;

import java.util.ArrayList;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;

public class Neuron {
	
	private float netInput = 0;
	private float activationThreshold = 0;
	private float weight = 0;
	private int id;
	private float output;
	private ArrayList<Float> weights = new ArrayList<Float>();

	public void setNetInput(float netinput) {
		netInput = netinput;
		
	}

	public float setGetNetInput() {
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

	public void setActivationFunctionType(ACTIVATION_FUNCTION sigmoid) {
		// TODO Auto-generated method stub
		
	}

	public ACTIVATION_FUNCTION getActivationFunctionType() {
		// TODO Auto-generated method stub
		return ACTIVATION_FUNCTION.SIGMOID;
	}
	
}
