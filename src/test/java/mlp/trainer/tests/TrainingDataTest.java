package mlp.trainer.tests;



import static org.junit.jupiter.api.Assertions.assertArrayEquals;

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
	 * Mean A1 = 2, mean A2 = 5, mean A3 = 28.5; 
	 * Variance A1 = (1-2)^2+(2-2)^2+(3-2)^2 = 1+0+1 = 2
	 * Variance A2 = (4-5)^2+(5-5)^2+(6-5)^2 = 1+0+1 = 2
	 * Variance A3 = ((7-28.5)^2+(100-28.5)^2+(-50-28.5)^2)/3 = 3912.25
	 * standard dev A3 = 62.5479815821
	 */
	@Test
	void testGetVariances() {
		float[][] data = {{1,4,7},
				{2,5,100},
				{3,6,-50}};
		sut.setData(data);
		float[] expected = {2,2, 3912.25f};
		float[] isVarianceLargeActual = sut.getVariances();
		assertArrayEquals(expected,isVarianceLargeActual);
	}
	
	/**
	 *Test function get means
	 * A1 = {1,2,3}, A2 = {4,5,6}, A3 = {7,100,-50}
	 * Mean A1 = 2, mean A2 = 5, mean A3 = 28.5; 
	 * Variance A1 = (1-2)^2+(2-2)^2+(3-2)^2 = 1+0+1 = 2
	 * Variance A2 = (4-5)^2+(5-5)^2+(6-5)^2 = 1+0+1 = 2
	 * Variance A3 = ((7-28.5)^2+(100-28.5)^2+(-50-28.5)^2)/3 = 3912.25
	 * standard dev A3 = 62.5479815821
	 */
	@Test
	void testGetMeans() {
		float[][] data = {{1,4,7},
				{2,5,100},
				{3,6,-50}};
		sut.setData(data);
		float[] expected = {2,2, 3912.25f};
		sut.calculateMeans();
		float[] isVarianceLargeActual = sut.getMeans();
		assertArrayEquals(expected,isVarianceLargeActual);
	}

}
