package mlp.trainer.tests;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.trainer.TerminationCriteria;
import mlp.trainer.TerminationCriteria.TERMINATION_CRITERIA;

public class TerminationCriteriaTest {
	private TerminationCriteria sut;
	
	@BeforeEach
	void init() {
		sut = new TerminationCriteria();
	}
	
	@Test
	void testTermiantionCriteriaConstructor() {
		TERMINATION_CRITERIA[] expected = {TERMINATION_CRITERIA.MAX_ITERATIONS,TERMINATION_CRITERIA.EPSILON};
		int expectedIterations = 5;
		float expectedEpsilon  = 0.001f;
		sut = new TerminationCriteria(expected,expectedIterations, expectedEpsilon);
		TERMINATION_CRITERIA[] actualCriteria = sut.getTerminationCriterias();
		int actualIterations = sut.getIterations();
		float actualEpsilon = sut.getEpsilon();
		assertArrayEquals(expected,actualCriteria);
		assertEquals(expectedIterations,actualIterations);
		assertEquals(expectedEpsilon,actualEpsilon);
	}
	
	@Test
	void testSetGetTerminationCriteriaParams() {
		TERMINATION_CRITERIA[] actualCriteria = sut.getTerminationCriterias();
		TERMINATION_CRITERIA[] expected = {TERMINATION_CRITERIA.MAX_ITERATIONS,TERMINATION_CRITERIA.EPSILON};
		int expectedIterations = 5;
		float expectedEpsilon  = 0.001f;
		sut.setCriteria(expected);
		sut.setIterations(expectedIterations);
		sut.setEpsilon(expectedEpsilon);
		int actualIterations = sut.getIterations();
		float actualEpsilon = sut.getEpsilon();
		assertArrayEquals(expected,actualCriteria);
		assertEquals(expectedIterations,actualIterations);
		assertEquals(expectedEpsilon,actualEpsilon);
	}
	
	@Test
	void testDefaultTerminationCriteriaParams() {
		TERMINATION_CRITERIA[] actualCriteria = sut.getTerminationCriterias();
		TERMINATION_CRITERIA[] expected = {TERMINATION_CRITERIA.MAX_ITERATIONS,TERMINATION_CRITERIA.EPSILON};
		int expectedIterations = 10;
		float expectedEpsilon  = 0.01f;
		int actualIterations = sut.getIterations();
		float actualEpsilon = sut.getEpsilon();
		assertArrayEquals(expected,actualCriteria);
		assertEquals(expectedIterations,actualIterations);
		assertEquals(expectedEpsilon,actualEpsilon);
	}
}
