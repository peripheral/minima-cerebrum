package mlp.trainer;

import java.util.Arrays;

public class TrainingData {

	private float[][] data = null;
	private float[] means = null;
	private float[] variances;
	private boolean subtractMean = false;
	private boolean varianceNormalized = false;
	private boolean meanNormalized = false;
	private float averageMean;
	private float averageVariance;
	private boolean preparationExecuted = false;

	public void setData(float[][] data) {
		this.data = data;
	}

	public float[][] getData() {
		return data;
	}

	public float[] getVariances() {
		return variances;
	}

	public float[] getMeans() {
		if(data.length == 0) {
			return null;
		}else {
			return means;
		}
	}

	public void calculateMeans() {
		if(means == null && data !=null) {
			means = new float[data[0].length];
		}
		for(int col = 0 ; col < data.length;col++) {
			means[col] = 0;
			for(int row = 0; row < data[0].length;row++) {
				means[col] = means[col]+data[row][col]/data.length;
			}
		}
	}

	public void calculateVariances() {
		if(variances == null && data !=null) {
			variances = new float[data[0].length];
		}
		if(means == null) {
			calculateMeans();
		}
		for(int col = 0 ; col < data[0].length;col++) {
			variances[col] = 0;
			for(int row = 0; row < data.length;row++) {
				variances[col] = (float) (variances[col]+Math.pow(data[row][col]-means[col],2)/data.length);
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
		float[] dataRow = Arrays.copyOf(data[row], data[0].length);
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

}
