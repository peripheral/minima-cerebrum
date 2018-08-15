package math.utils.tests;



import static org.junit.Assert.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import math.utils.StatisticUtils;


public class StatisticUtilsTest {

private StatisticUtils sut;
	
	@BeforeEach
	void init() {
		sut = new StatisticUtils();
	}
	
	@Test
	void testMeanMethod() {
		float[] values = {1,2,3,4,5,6};
		float expectedMean = 0;
		float sum = 0;
		for(int i = 0;i< values.length;i++) {
			sum = sum + values[i];
		}
		expectedMean = sum/values.length;
		float actualAverage = (float) StatisticUtils.mean(values); 
		assertEquals(expectedMean,actualAverage,0.1f);
	}
	
	@Test
	void testVarianceMethod() {
		float[] values = {1,2,3,4,5,6};
		float mean = 0;
		double expectedVariance = 0;
		double sum = 0;
		for(int i = 0;i< values.length;i++) {
			sum = sum + values[i];
		}
		mean = (float) (sum/values.length);
		sum = 0;
		for(int i = 0;i< values.length;i++) {
			sum = sum + Math.pow(values[i]-mean,2);
		}
		expectedVariance = sum/values.length;
		
		float actualVariance = (float) StatisticUtils.variance(values); 
		assertEquals(expectedVariance,actualVariance,0.01f);
	}
	/**
	 * Test for function of determining if the data has NoneZeroMean,
	 * that takes as input array with input arrays.
	 * The row presents an input example, the column presents attribute
	 * isNoneZeroMean should return true if the mean != 0 else false
	 */
	@Test
	void testNoneZeroMean() {
		float[][] data = {{1,2,-2},
				{1,5,6},
				{1,5,-4}};
		boolean[] isNoneZeroMeanActual = sut.isNoneZeroMean(data);
		boolean[] expected = {true,true,false};
		assertArrayEquals(expected,isNoneZeroMeanActual);
	}

	/**
	 * Test tests function that determines if Variance is too large.
	 *  Function takes as input array with input arrays.
	 * The row presents an input example, the column presents attribute.
	 * The function isLargeVariance bases on ratio of expected(mean) / standard deviation and user defined threshold.
	 * if expected(mean) / standard deviation > threshold - false, else true
	 * A1 = {1,2,3}, A2 = {4,5,6}, A3 = {7,100,-50}
	 * Mean A1 = 2, mean A2 = 5, mean A3 = 28.5; 
	 * Variance A1 = (1-2)^2+(2-2)^2+(3-2)^2 = 1+0+1 = 2
	 * Variance A2 = (4-5)^2+(5-5)^2+(6-5)^2 = 1+0+1 = 2
	 * Variance A3 = ((7-28.5)^2+(100-28.5)^2+(-50-28.5)^2)/3 = 3912.25
	 * standard dev A3 = 62.5479815821
	 */
	@Test
	void testIsVarianceLarge() {
		float[][] data = {{1,4,7},
				{2,5,100},
				{3,6,-50}};
		float threshold = 0.50f;
		boolean[] expected = {false,false,true};
		boolean[] isVarianceLargeActual = sut.isLargeVariance(data,threshold);
		assertArrayEquals(expected,isVarianceLargeActual);
	}

	/**
	 * Test tests function that determines if inputs are correlated. 
	 * Function takes as input array with input arrays.
	 * The row presents an input example, the column presents attribute
	 * Correlation with Person's moment product  Pxy = Cov(X,Y)/(sY*sX);
	 * mean A1 = 2,mean A2 = 5, mean A3 = 8;
	 * Variance A1 = (1-2)^2+(2-2)^2+(3-2)^2 = 1+0+1 = 2
	 * Variance A2 = (4-5)^2+(5-5)^2+(6-5)^2 = 1+0+1 = 2
	 * Variance A3 = (7-8)^2+(8-8)^2+(9-8)^2 = 1+0+1 = 2
	 * Cov(A1,A2)  = E[(X - E[X])*(Y-E[Y])]
	    = ((1-2)*(4-5)+(1-2)*(5-5)+(1-2)*(6-5)+
		(2-2)*(4-5)+(2-2)*(5-5)+(2-2)*(6-5)+
		(3-2)*(4-5)+(3-2)*(5-5)+(3-2)*(6-5)) = 0
	 * Pa1a2 = Cov(A1,A2)/(sA1*sA2) = 0
	 */
	@Test
	void testOnCorrelationAmongVariables () {
		float[][] data = {{1,4,7},
				{2,5,8},
				{3,6,9}};
		boolean[] isCorrelatedExpected = {false,false,false};
		float threshold = 0.3f;
		boolean[] isCorrelated = sut.areInputsCorrelated(data,threshold);
		assertArrayEquals(isCorrelatedExpected,isCorrelated);
	}
	
	/**
	 * Alternative function to softmax
	 */
	@Test 
	void testOfAlternativeToSoftmaxFunction() {
		float[] data = {1,10,5,3,100};
		float[] actual = StatisticUtils.calculateSoftmaxWithoutE(data);
		float[] expected = new float[data.length];
		float sum = 0;
		for(float f:data) {
			sum = sum +f;
		}
		for(int i = 0;i < data.length; i++) {
			expected[i] = data[i]/sum;
		}
		assertArrayEquals(expected,actual,0.00f);		
	}
	
	/**
	 * Test of softmax function.
	 */
	@Test 
	void testOfSoftmaxFunction() {
		float[] data = {1,10,5,3,30};
		float[] actual = StatisticUtils.calculateSoftmax(data);
		float[] expected = new float[data.length];
		float sum = 0;
		for(float f:data) {
			sum = sum + (float) (Math.pow(Math.E,f));
		}
		for(int i = 0;i < data.length; i++) {
			expected[i] = (float) (Math.pow(Math.E,data[i])/sum);
		}
		assertArrayEquals(expected,actual,0.00001f);		
	}
	
	/**
	 * Test of softmax function.
	 * sum = e^(0) + e^(10) + e^(5) + e^(3) +e^(15) = 3291213.33696
	 * partial derivate with respect to x1 
	 *( e^(x1)((e^x1) + (e^x2) +(e^x3) +(e^x4)... ) -  (e^x1)(e^x1) )/((e^x1) + (e^x2) +(e^x3) +(e^x4)... )^2
	 */
	@Test 
	void testOfSoftmaxFunctionPartialDerivative() {
		float[] data = {0,10,5,3,15};
		int variableIdx = 0;
		float actual = StatisticUtils.calculateSoftmaxPartialDerivative(data, variableIdx);
	
		float sum = 3291213.33696f;
		float denominator = (float) (3291213.33696 * 3291213.33696);
		float expected = (1 * sum - 1)/denominator;
		assertEquals(expected,actual,0.00001f);		
	}
	
	/**
	 * Test of softmax function.
	 * sum = e^(0) + e^(10) + e^(5) + e^(3) +e^(15) = 3291213.33696
	 * partial derivate with respect to x2- 
	 *( e^(x2)((e^x1) + (e^x2) +(e^x3) +(e^x4)... ) -  (e^x2)(e^x2) )/((e^x1) + (e^x2) +(e^x3) +(e^x4)... )^2
	 */
	@Test 
	void testOfSoftmaxFunctionPartialDerivative2() {
		float[] inputs = {1,10,5,3,15};
		int variableIdx = 1;
		float actual = StatisticUtils.calculateSoftmaxPartialDerivative(inputs, variableIdx);
	
		float sum = 3291215.05524f;
		float denominator = (float) (3291215.05524f * 3291215.05524f);
		float expected = (22026.4657948f * sum - 22026.4657948f * 22026.4657948f)/denominator;
		assertEquals(expected,actual,0.00001f);		
	}
	
}
