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

}
