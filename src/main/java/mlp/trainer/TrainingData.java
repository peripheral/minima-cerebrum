package mlp.trainer;

public class TrainingData {

	private float[][] data = null;
	private float[] means = null;
	private float[] variances;

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

}
