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

	//TODO refactor, change places on weight and index
	public void setWeight(float weight, int index) {
		if(weights.size() <=index) {
			for(int idx = weights.size()-1; idx <index+1;idx++ ) {
				weights.add(0.0f);
			}
		}
		this.weights.set(index,weight);
		
	}

	public float getWeight(int index) {
		return weights.get(index);
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
