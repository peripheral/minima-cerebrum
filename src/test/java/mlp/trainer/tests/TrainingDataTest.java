package mlp.trainer.tests;



import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;


import mlp.trainer.TrainingData;

public class TrainingDataTest {
	private TrainingData sut;

	@BeforeEach
	void init() {
		sut = new TrainingData();
	}

	/**
	 * Test for set get data
	 */
	@Test
	void testSetGetData() {
		float[][] data = {{1,2,-2},
				{1,5,6},
				{1,5,-4}};
		float[][] expected = {{1,2,-2},
				{1,5,6},
				{1,5,-4}};
		sut.setData(data);
		float[][] actuals = sut.getData();
		assertArrayEquals(expected,actuals);
	}

	/**
	 *Test function get variances
	 * A1 = {1,2,3}, A2 = {4,5,6}, A3 = {7,100,-50}
	 * Mean A1 = 2, mean A2 = 5, mean A3 = 19; 
	 * Variance A1 = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A2 = ((4-5)^2+(5-5)^2+(6-5)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A3 = ((7-19)^2+(100-19)^2+(-50-19)^2)/3 = 3822
	 */
	@Test
	void testGetVariances() {
		float[][] data = {{1,4,7},
				{2,5,100},
				{3,6,-50}};
		sut.setData(data);
		sut.calculateVariances();
		float[] expected = {2/3f,2/3f, 3822f};
		float[] isVarianceLargeActual = sut.getVariances();
		assertArrayEquals(expected,isVarianceLargeActual);
	}

	/**
	 *Test function get means
	 * A1 = {1,2,3}, A2 = {4,5,6}, A3 = {7,100,-50}
	 * Mean A1 = 2, mean A2 = 5, mean A3 = 19; 
	 */
	@Test
	void testGetMeans() {
		float[][] data = {{1,4,7},
				{2,5,100},
				{3,6,-50}};
		sut.setData(data);
		float[] expected = {2,5, 19f};
		sut.calculateMeans();
		float[] isVarianceLargeActual = sut.getMeans();
		assertArrayEquals(expected,isVarianceLargeActual,0.00001f);
	}

	@Test
	void meanSubtractionVariableMustBeFalseInitially() {
		boolean expected = false;
		boolean actual = sut.isMeanSubstracted();
		assertEquals(expected,actual);
	}

	@Test
	void meanSubtractionVariableMustBeTrue() {
		boolean expected = true;
		sut.setSubtractMean(true);
		boolean actual = sut.isMeanSubstracted();
		assertEquals(expected,actual);
	}

	@Test
	void testSubtractMeanFromInput() {
		float[][] data = {{1,4,7},
				{2,5,8},
				{3,6,9}};
		sut.setData(data);
		sut.calculateMeans();
		float[] expected = {-1,-1,-1};
		float[] expected1 = {0,0,0};
		float[] expected2 = {1,1,1};
		sut.setSubtractMean(true);
		float[] actual = sut.getInputRow(0);
		float[] actual1 =  sut.getInputRow(1);
		float[] actual2 =  sut.getInputRow(2);
		assertArrayEquals(expected,actual);
		assertArrayEquals(expected1,actual1);
		assertArrayEquals(expected2,actual2);
	}

	/**
	 * Test on function that calculates average mean among attributes
	 * Mean A1 = (1+2+3)/3 = 2
	 * Mean A2 = (4+5+6)/3 = 5
	 * Mean A3 = (7+8+9)/3 = 8
	 * Mean = (2+5+8)/3 = 5;
	 *  
	 */
	@Test
	void testCalculateAverageMeans() {
		float[][] data = {{1,4,7},
				{2,5,8},
				{3,6,9}};
		sut.setData(data);
		sut.calculateMeans();
		sut.calculateAverageMean();
		float expected = 5;
		float actual = sut.getAverageMean();
		assertEquals(expected,actual);
	}
	
	/**
	 * Test on function that calculates average variance
	 * Mean A1 = (1+2+3)/3 = 2
	 * Mean A2 = (4+5+6)/3 = 5
	 * Mean A3 = (7+8+9)/3 = 8
	 * Variance A1 = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A2 = ((4-5)^2+(5-5)^2+(6-5)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A3 = ((7-8)^2+(8-8)^2+(9-8)^2)/3 =2/3	 
	 */
	@Test
	void testCalculateAverageVariance() {
		float[][] data = {{1,4,7},
				{2,5,8},
				{3,6,9}};
		sut.setData(data);
		sut.calculateVariances();
		sut.calculateAverageVariance();
		float expected = 2/3f;
		float actual = sut.getAverageVaraince();
		assertEquals(expected,actual);
	}

	/**
	 * Test on function that calculates a Variance among Varaices of each
	 *  attribute
	 * Variance A1 = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A2 = ((4-5)^2+(5-5)^2+(6-5)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A3 = ((7-8)^2+(8-8)^2+(9-8)^2)/3 =2/3
	 */
	@Test
	void shouldReturnMeanNormalizedInputRow() {
		float[][] data = {{1,4,7},
				{2,5,8},
				{3,6,9}};
		sut.setData(data);
		sut.setNormalizedMeanTransformInput(true);
		/*
		 * x1 = (x1 - meanA1 + mean)sqrt(Variance)/sqrt(VarianceA1)
		 */
		float[] expected = {(float) ((1 - 2 + 5)*(Math.sqrt(2/3.0)/Math.sqrt(2/3.0))),
				(float) ((4 - 5 + 5)*(Math.sqrt(2/3.0)/Math.sqrt(2/3.0))),
				(float) ((7 - 8 + 5)*(Math.sqrt(2/3.0)/Math.sqrt(2/3.0)))};
		float[] actual = sut.getInputRow(0);
		assertArrayEquals(expected,actual);
	}

	/**
	 * Test on function that calculates a Variance among Variances of each
	 * attribute
	 * Mean A1 = (1+2+3)/3 = 2
	 * Mean A2 = (4+5+6)/3 = 5
	 * Mean A3 = (7+8+9)/3 = 8
	 * Mean = (2+5+8)/3 = 5;
	 * Variance A1 = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A2 = ((4-5)^2+(5-5)^2+(6-5)^2)/3 = (1+0+1)/3 = 2/3
	 * Variance A3 = ((7-8)^2+(8-8)^2+(9-8)^2)/3 = 2/3
	 * Mean Variance = 2/3
	 * 
	 */
	@Test
	void shouldReturnVarianceNormalizedInput() {
		float[][] data = {{1,4,7},
				{2,5,8},
				{3,6,9}};
		sut.setData(data);
		sut.setNormalizedVarianceTransformInput(true);
		/*
		 * x1 = (x1 - meanA1 + mean)sqrt(Variance)/sqrt(VarianceA1)
		 */
		float[] expected = {(float) ((1 - 2 + 5)*(Math.sqrt(2/3.0)/Math.sqrt(2/3.0))),
				(float) ((4 - 5 + 5)*(Math.sqrt(2/3.0)/Math.sqrt(2/3.0))),
				(float) ((7 - 8 + 5)*(Math.sqrt(2/3.0)/Math.sqrt(2/3.0)))};
		float[] actual = sut.getInputRow(0);
		assertArrayEquals(expected,actual);
	}
	
	@Test
	void testIsInputTransformedWithNormalizedMeanShouldReturnTrue() {
		boolean expected = true;
		sut.setNormalizedMeanTransformInput(true);
		boolean actual = sut.isInputTransformedWithNormalizedMean();
		assertEquals(expected,actual);
	}
	
	@Test
	void testIsInputTransformedWithNormalizedMeanShouldReturnFalse() {
		boolean expected = false;
		boolean actual = sut.isInputTransformedWithNormalizedMean();
		assertEquals(expected,actual);
	}
	
	@Test
	void testIsInputTransformedWithNormalizedVarainceShouldReturnTrue() {
		boolean expected = true;
		sut.setNormalizedVarianceTransformInput(true);
		boolean actual = sut.isInputTransformedWithNormalizedVariance();
		assertEquals(expected,actual);
	}
	
	@Test
	void testIsInputTransformedWithNormalizedVarianceShouldReturnFalse() {
		boolean expected = false;
		boolean actual = sut.isInputTransformedWithNormalizedVariance();
		assertEquals(expected,actual);
	}

}
