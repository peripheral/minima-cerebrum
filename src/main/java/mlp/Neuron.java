package mlp;

import java.util.ArrayList;

import mlp.ANN_MLP.ACTIVATION_FUNCTION;

public class Neuron {
	
	public enum TRANSFER_FUNCTION {IDENTITY}


	private float netInput = 0;
	private float activationThreshold = 0;
	private int id = -1;
	private float output;
	private ArrayList<Float> weights = new ArrayList<Float>();
	private ACTIVATION_FUNCTION aFunction = ACTIVATION_FUNCTION.SIGMOID;
	private TRANSFER_FUNCTION transferFunction = TRANSFER_FUNCTION.IDENTITY;
	private float a = 1;
	private float b = 1;
	private float activationResult;

	public Neuron() {}
	
	
	public Neuron(ACTIVATION_FUNCTION aF) {
		aFunction = aF;
	}
	
	public Neuron(ACTIVATION_FUNCTION f,int upperLayerSize) {

	}

	

	public Neuron(int upperLayerSize) {
		for(int i = 0;i < upperLayerSize;i ++) {
			weights.add(1.0f);
		}
	}


	public void setNetInput(float netinput) {
		netInput = netinput;
		
	}

	public float getNetInput() {
		return netInput;
	}

	public void setActivationThreshold(double threshold) {
		activationThreshold = (float)threshold;
	}

	public float getActivationThreshold() {		
		return activationThreshold;
	}

	/**
	 * Set weight under index. By default weights are zero
	 * @param index
	 * @param weight
	 */
	public void setWeight(int index, float weight) {
		if(weights.size() <=index) {
			for(int idx = weights.size(); idx <index+1;idx++ ) {
				weights.add(0.0f);
			}
			this.weights.set(index,weight);
		}else {
			this.weights.set(index,weight);
		}
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

	public void setActivationFunctionType(ACTIVATION_FUNCTION function) {
		aFunction = function;		
	}
	
	public ACTIVATION_FUNCTION getActivationFunctionType() {
		return aFunction;
	}


	public ArrayList<Float> getWeights() {
		return weights;
	}


	/**
	 * Uses activation function model
	 * @return result of activation function
	 */
	public float activate() {
		activationResult = NeuronFunctionModels.activate(aFunction,a,b ,netInput);
		return activationResult;
	}


	public void setTransferFunction(TRANSFER_FUNCTION type) {
		transferFunction = type;		
	}


	public TRANSFER_FUNCTION getTransferFunction() {
		return transferFunction;
	}


	public void transfer() {
		output = NeuronFunctionModels.transfer(transferFunction,activationResult);		
	}
	
	public void transferNoActivation() {
		output = NeuronFunctionModels.transfer(transferFunction,netInput);		
	}
	
}
