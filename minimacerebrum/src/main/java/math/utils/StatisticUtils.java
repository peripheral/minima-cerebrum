package math.utils;

import java.util.Random;

public class StatisticUtils {

	public static float variance(float[] values) {
		double mean = mean(values);
		double variance = 0;
		for(int i = 0;i < values.length;i++) {
			variance = variance + Math.pow((values[i]-mean),2)/values.length;
		}
		return (float) variance;
	}

	public static double mean(float[] values) {
		double mean = 0;
		for(int i = 0;i < values.length;i++) {
			mean = mean + values[i]/values.length;
		}
		return mean;
	}
	
	/**function that determines if inputs are correlated. 
	 * Function takes as input array of float arrays.
	 * The row presents an input example, the column presents attribute
	 * Correlation with Person's moment product  Pxy = Cov(X,Y)/(sY*sX);
	 * @param data - m by n matrix, m - rows, n - columns
	 * @param threshold - limit that evaluates correlation to true. 1 is positive correlation, -1 negative
	 * @return array with booleans in order A1,A2
	 */
	public boolean[] areInputsCorrelated(float[][] data, float threshold) {
		if(isDataEmpty(data)) {
			return new boolean[]{false,false};
		}
		
		int resultSize = getPermuations(data[0].length);
		boolean[] result = new boolean[resultSize];
		
		double[] mean = calculateMeans(data);
		double[] covariance = calculateCovariance(data,mean);
		
		
		for(int covIdx = 0;covIdx < covariance.length;covIdx++) {
			for (int col1 = 0; col1 < mean.length-1; col1++) {
				for (int col2 = col1+1; col2 < mean.length; col2++) {
					if(threshold < Math.abs(covariance[covIdx]/(mean[col1]*mean[col2]))) {
						result[covIdx] = true;
					}else {
						result[covIdx] = false;
					}
				}
			}

		}		
		return result;
	}
	
	/**
	 * Returns possible permutations, for given array size
	 * @param n number of elements
	 * @return - number of possible permutations among n elements
	 */
	private int getPermuations(int n) {
		int permuations = 0;
		for (int i = 0; i < n; i++) {
			for (int j = i+1; j < n; j++) {
				permuations++;
			}
		}
		return permuations;
	}

	/**
	 * Calculates covariance between columns
	 * @param data miltidimiension array containing rows with data
	 * @param mean mean per column
	 * @return 
	 */
	private double[] calculateCovariance(float[][] data,double[] mean) {
		int permutations = getPermuations(data[0].length);
		int rowLength = data[0].length;
		double[] covariance = new double[permutations];
		int idx = 0;
		for(int col1 = 0; col1 < rowLength-1; col1++) {
			for (int col2 = col1+1; col2 < rowLength; col2++) {
				covariance[idx] = calculateCovarianceFor(col1,col2,mean,data);	
				idx++;
			}					
		}
		return covariance;
	}

	private double calculateCovarianceFor(int col1,int col2, double[] mean, float[][] data) {
		double covariance = 0;
		for(int row = 0;row < data.length;row++) {
			for(int row1 = 0;row1 < data.length;row1++) {
				covariance  = (float) (covariance +
						(data[row][col1]-mean[col1])*(data[row1][col2]-mean[col2])/(2*data.length));
			}
		}	
		return covariance;
	}

	/**
	 * Calculate mean per column 
	 * @param data
	 * @return - returns means per row 
	 */
	private double[] calculateMeans(float[][] data) {
		double[] mean = new double[data[0].length];
		for(int col = 0; col < data[0].length; col++) {			
			mean[col] = 0;
			for(int row = 0;row < data.length;row++) {
				mean[col]  = mean[col] + data[row][col]/data.length;
			}
		}
		return mean;
	}

	private boolean isDataEmpty(float[][] data) {
		return data.length == 0;
	}

	/**
	 * Calculate variances per attribute, all values under same column belongs to same
	 * attributes 
	 * @param data - m by n matrix, m - rows, n - columns
	 * @return array with booleans in order A1,A2
	 */
	public boolean[] isLargeVariance(float[][] data, float threshold) {
		if(data.length == 0) {
			return new boolean[0];
		}
		boolean[] result = new boolean[data[0].length];
		double[] mean = new double[data[0].length];
		double[] variance = new double[data[0].length];
		for(int col = 0; col < data[0].length; col++) {			
			mean[col] = 0;
			for(int row = 0;row < data.length;row++) {
				mean[col]  = mean[col] + data[row][col]/data.length;
			}
		}
		for(int col = 0; col < data[0].length; col++) {
			variance[col] = 0;
			for(int row = 0;row < data.length;row++) {
				variance[col]  = (float) (variance[col] + Math.pow(data[row][col]-mean[col],2)/data.length);
			}
		}
		for(int col = 0; col < result.length;col++) {
			if(threshold >= mean[col]/Math.sqrt(variance[col])) {
				result[col] = true;
			}
		}
		return result;
	}
	
	/**
	 * Tests if the attributes in data are zero mean
	 * @param data - m by n matrix, m - rows, n - columns
	 * @return array with booleans in order A1,A2
	 */
	public boolean[] isNoneZeroMean(float[][] data) {
		if(data.length == 0) {
			return new boolean[0];
		}
		boolean[] result = new boolean[data[0].length];
		float mean = 0;
		for(int col = 0; col < data[0].length; col++) {
			for(int row = 0;row < data.length;row++) {
				mean = mean + data[row][col]/data.length;
			}
			if(mean > Float.MIN_VALUE) {
				result[col] = true;
			}
			mean = 0;
		}
		return result;
	}

	/**
	 * * Produces next random weight according to Xavier Glorort et al distribution
	 *  U[- sqrt(6)/(n + n1),sqrt(6)/(n + n1)] n -size of lower, n1 - size of upper distribution
	 * @param n - lower layer size
	 * @param n1 - top layer size
	 * @return
	 */
	public static float getXavierRandomWeight(int n,int n1) {
		double amplitude = (Math.sqrt(6)/(n+n1))*2;
		double lowerLimit = - Math.sqrt(6)/(n+n1);
		Random rm = new Random();
		double value = rm.nextDouble()*amplitude;
		return (float) (value+lowerLimit);
	}

	/**
	 * The function uses alternative function which doesn't use natural base E
	 * Sj = Aj/sum
	 * @param data
	 * @return
	 */
	public static float[] calculateSoftmaxWithoutE(float[] data) {
		float sum = 0;
		float[] result = new float[data.length];
		for(float f:data) {
			sum = sum +f;
		}
		if(sum == 0) {
			return new float[0];
		}
		for(int i = 0;i < data.length; i++) {
			result[i] = data[i]/sum;
		}
		return result;
	}
	
	/**
	 * The function implement traditional softmax
	 * Sj = e^(Aj)/(e^(A1) + e^(A2) ..e^(An))
	 * @param data
	 * @return
	 */
	public static float[] calculateSoftmax(float[] data) {
		double sum = 0;
		float[] result = new float[data.length];
		float[] ePowsAj = new  float[data.length];
		for(int i = 0 ;i < data.length;i++) {
			ePowsAj[i] = (float) Math.pow(Math.E,data[i]);
			sum = sum + ePowsAj[i];
		}
		if(sum == 0) {
			return new float[0];
		}
		for(int i = 0;i < data.length; i++) {
			result[i] = (float) (ePowsAj[i]/sum);
		}
		return result;
	}
	
	/**
	 * The function implement traditional softmax partial derivative
	 * Sj = e^(Aj)/(e^(A1) + e^(A2) ..e^(An)) , 
	 * partial derivative - 
	 * ((e^Aidx)(e^(Aidx) + e^(A2) ..e^(An)) - (e^Aidx)(e^(Aidx))/(e^(A1) + e^(A2) ..e^(An)^2
	 * @param inputs
	 * @return
	 */
	public static float calculateSoftmaxPartialDerivative(float[] inputs,int idx) {
		double denominator = 0;
		double ePowIdx = (float) Math.pow(Math.E, inputs[idx]);
		float result ;
		for(int i = 0 ;i < inputs.length;i++) {	
			denominator = denominator + Math.pow(Math.E,inputs[i]);
		}
		if(Double.isNaN(denominator)) {
			System.err.println("NaN for denominator");
			ePowIdx = Math.abs(inputs[idx]);
			denominator = 0;
			for(int i = 0 ;i < inputs.length;i++) {	
				denominator = denominator + Math.abs(inputs[i]);
			}
			if(denominator == 0) {
				return Float.NaN;
			}
			result = (float) ((ePowIdx*denominator - ePowIdx*ePowIdx)/(denominator*denominator));
			if(Float.isNaN(result)){
				System.err.println("Second operation yelded NaN");
			}
			return result;
		}
		if(denominator == 0) {
			return Float.NaN;
		}
		result = (float) ((ePowIdx*denominator - ePowIdx*ePowIdx)/(denominator*denominator));
		return result;
	}

	/**
	 * Method used from Mathew D.Zeiler ,"ADADELTA: AN ADAPTIVE LEARNING RATE METHOD",2012
	 * E[v^2] = decayFactor * E[v^2]_t-1 + (1 - decayFactor) v_t^2
	 *  
	 * @param oldMeanSquared - E[v^2]_t-1
	 * @param decayFactor
	 * @param value - v_t
	 * @return
	 */
	public static float calculateMeanSqured(float oldMeanSquared, float decayFactor, float value) {
		return (float) (decayFactor * oldMeanSquared +(1 - decayFactor)*Math.pow(value,2));
	}
}
