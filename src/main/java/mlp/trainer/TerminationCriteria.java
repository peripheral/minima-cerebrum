package mlp.trainer;

public class TerminationCriteria {
	public enum TERMINATION_CRITERIA {
		/**
		 * MIN_ERROR_DELTA, FUNCTIONAL_TOLERANCE - means the same thing
		 */
		EPSILON, MAX_ITERATIONS, MIN_ERROR_DELTA, FUNCTIONAL_TOLERANCE
	}

	private TERMINATION_CRITERIA[] terminationCriteria = {TERMINATION_CRITERIA.MAX_ITERATIONS,TERMINATION_CRITERIA.EPSILON};
	private int maxIterations = 5;
	private float epsilon  = 0.01f;
	float functTolerance;

	public TerminationCriteria(TERMINATION_CRITERIA[] criteria, int iterations, float epsilon) {
		terminationCriteria = criteria;
		this.maxIterations = iterations;
		this.epsilon = epsilon;
	}

	public TerminationCriteria() {}

	public TerminationCriteria(TERMINATION_CRITERIA[] criteria) {
		terminationCriteria = criteria;
	}

	public TerminationCriteria(TERMINATION_CRITERIA[] criteria, int maxIterations) {
		terminationCriteria = criteria;
		this.maxIterations = maxIterations;
	}

	public TERMINATION_CRITERIA[] getTerminationCriterias() {
		return terminationCriteria;
	}

	public int getIterations() {
		return maxIterations;
	}

	public float getEpsilon() {
		return epsilon;
	}

	public void setCriteria(TERMINATION_CRITERIA[] criteria) {
		terminationCriteria = criteria;		
	}

	public void setIterations(int it) {
		maxIterations = it;
	}

	public void setEpsilon(float eps) {
		this.epsilon = eps;		
	}

	public void setFunctTolConst(float minFunctTolerance) {
		this.functTolerance = minFunctTolerance;	
	}
	
	public float getFunctTolConst() {
		return functTolerance;
	}
}
