package mlp.trainer.data.factory;

import java.util.List;

import mlp.trainer.data.Data;
import mlp.trainer.data.TrainingData;
import mlp.trainer.data.ValidationData;

public class DataSetFactory{


	private List<float[]> inputs;
	private List<float[]> targets;
	private int classCount = 1;
	/**
	 * Portion of data to be used to train
	 */
	private double trainingDataPortion = 0.5;
	/**
	 * Portion of data to be used to validate
	 */
	private double validationDataPortion = 0.25;
	/**
	 * Portion of data to be used to test
	 */
	private double testDataPortion = 0.25;

	public DataSetFactory() {
		//TODO
	}

	/**
	 * Creates TrainingData from provided inputs and targets rows. Requires even distribution
	 * of classes and sorted data provided in inputs
	 * @param portion
	 * @return
	 */
	public TrainingData getTrainingData() {
		/* Size of examples in data set*/
		int examplesPerClass = inputs.size()/classCount;
		/* Size of set to be created */
		int setSize = (int) (inputs.size()*trainingDataPortion);
		/* Size of examples per class in new set */
		int portionPerClass = setSize/classCount;
		float[][] trainingInputs = new float[setSize][];
		float[][] trainingTargets = new float[setSize][];
		for (int i = 0; i < classCount; i++) {
			for (int offset = 0; offset < portionPerClass; offset++) {
				trainingInputs[i*portionPerClass+offset] = this.inputs.get(i*examplesPerClass+offset);
				trainingTargets[i*portionPerClass+offset] = this.targets.get(i*examplesPerClass+offset);
			}
		}
		return new TrainingData(trainingInputs, trainingTargets);
	}

	public void setInput(List<float[]> inputs) {
		this.inputs = inputs;
		
	}

	public void setTarget(List<float[]> targets) {
		this.targets = targets;		
	}

	public int getClassCount() {
		return classCount;
	}

	public void setClassCount(int classCount) {
		this.classCount = classCount;
	}

	public ValidationData getValidationData() {
		/* Size of examples in data set*/
		int examplesPerClass = inputs.size()/classCount;
		/* Size of set to be created */
		int setSize = (int) (inputs.size()*validationDataPortion);
		/* Size of examples per class in new set */
		int portionPerClass = setSize/classCount;
		float[][] validationInputs = new float[setSize][];
		float[][] validationTargets = new float[setSize][];
		for (int i = 0; i < classCount; i++) {
			for (int offset = 0; offset < portionPerClass; offset++) {
				validationInputs[i*portionPerClass+offset] = this.inputs.get(i*examplesPerClass+offset);
				validationTargets[i*portionPerClass+offset] = this.targets.get(i*examplesPerClass+offset);
			}
		}
		return new ValidationData(validationInputs, validationTargets);
	}
	
	
	
}
