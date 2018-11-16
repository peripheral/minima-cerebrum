package mlp.trainer.data;

import java.util.Arrays;

public class TrainingData extends Data{

	private float[] means = null;
	private float[] variances;
	private boolean subtractMean = false;
	private boolean varianceNormalized = false;
	private boolean meanNormalized = false;
	private float averageMean;
	private float averageVariance;
	private boolean preparationExecuted = false;

	/**
	 * @param inputs - represents x vector
	 * @param targets - represents y vector
	 */
	public TrainingData(float[][] inputs,float[][] targets) {
		this.input = inputs;
		this.target= targets;
	}

	/**
	 * data can be passed in one array. {x1,x2,x3,..,y1,y2,..}
	 * @param data
	 * @param offset - index that defines start of Y values
	 */
	public TrainingData(float[][] data,int offset) {
		input = new float[data.length][offset];
		target = new float[data.length][data[0].length - offset];
		for(int row = 0; row < data.length;row++) {
			for(int col = 0; col <offset;col++) {
				this.input[row][col] = data[row][col];
			}
			for(int col = offset; col <data[0].length;col++) {
				this.target[row][col-offset] = data[row][col];
			}
		}
	}

	public TrainingData() {	}

	public void setInputs(float[][] data) {
		this.input = data;
	}

	public float[][] getInputs() {
		return input;
	}

	public float[] getVariances() {
		return variances;
	}

	public float[] getMeans() {
		if(input.length == 0) {
			return new float[0];
		}else {
			return means;
		}
	}

	public void calculateMeans() {
		if(means == null && input !=null) {
			means = new float[input[0].length];
		}else if(input == null) {
			System.out.println("Couldn't claculate means, no data");
			return;
		}
		for(int col = 0 ; col < input[0].length;col++) {
			means[col] = 0;
			for(int row = 0; row < input[0].length;row++) {
				means[col] = means[col]+input[row][col]/input.length;
			}
		}
	}

	public void calculateVariances() {
		if(variances == null && input !=null) {
			variances = new float[input[0].length];
		}
		if(means == null) {
			calculateMeans();
		}
		for(int col = 0 ; col < input[0].length;col++) {
			variances[col] = 0;
			for(int row = 0; row < input.length;row++) {
				variances[col] = (float) (variances[col]+Math.pow(input[row][col]-means[col],2)/input.length);
			}
		}
	}

	public void setSubtractMean(boolean b) {
		subtractMean = b;		
	}

	public boolean isMeanSubstracted() {
		return subtractMean;
	}

	public float[] getInputRow(int row) {
		float[] dataRow = Arrays.copyOf(input[row], input[0].length);
		if(!preparationExecuted) {
			prepareForVarianceNormalization();
			prepareForMeanNoralization();
			preparationExecuted = true;
		}
		if(varianceNormalized) {
			for(int i = 0; i < dataRow.length;i++) {
				dataRow[i] = (float) ((dataRow[i]-means[i]+averageMean)*Math.sqrt(averageVariance)
						/Math.sqrt(variances[i]));
			}
		}else if(meanNormalized) {
			for(int i = 0; i < dataRow.length;i++) {
				dataRow[i] = (float) (dataRow[i]-means[i]+averageMean);
			}
		}
		if(subtractMean && !varianceNormalized) {
			if(meanNormalized) {
				for(int col = 0; col < dataRow.length; col++) {
					dataRow[col] = dataRow[col] - averageMean;
				}
			}else {
				for(int col = 0; col < dataRow.length; col++) {
					dataRow[col] = dataRow[col] - means[col];
				}
			}
		}
		return dataRow;
	}

	private void prepareForMeanNoralization() {
		calculateMeans();
		calculateAverageMean();		
	}

	private void prepareForVarianceNormalization() {
		calculateVariances();
		calculateAverageVariance();
		calculateAverageMean();		
	}

	public void calculateAverageMean() {
		averageMean = 0.0f;
		for(int i = 0; i < means.length;i++) {
			averageMean = averageMean +means[i];
		}
		averageMean = averageMean/means.length;
	}

	public float getAverageMean() {
		return averageMean;
	}

	public void calculateAverageVariance() {
		averageVariance = 0f;
		for(int i = 0; i < variances.length;i++) {
			averageVariance = averageVariance + variances[i];
		}
		averageVariance = averageVariance/variances.length;
	}

	public float getAverageVaraince() {
		return averageVariance;
	}

	public void setNormalizedMeanTransformInput(boolean b) {
		meanNormalized = b;		
	}

	public boolean isInputTransformedWithNormalizedMean() {
		return meanNormalized;		
	}

	public void setNormalizedVarianceTransformInput(boolean b) {
		varianceNormalized = b;		
	}

	public boolean isInputTransformedWithNormalizedVariance() {
		return varianceNormalized;
	}

	/**
	 * 
	 * @param idx - data row
	 * @param offset - start of new array
	 * @return target array
	 */
	public float[] getTargetRow(int idx) {
		return target[idx];
	}

	public void setTargets(float[][] out) {
		target = out;		
	}

	public TrainingData produceTrainingSet() {
		
		return null;
	}
}
