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

}
