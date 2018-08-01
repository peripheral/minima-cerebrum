package mlp.trainer;

public class TerminationCriteria {
	public enum TERMINATION_CRITERIA {
		EPSILON, MAX_ITERATIONS, MIN_ERROR_DELTA
	}

	private TERMINATION_CRITERIA[] terminationCriteria = {TERMINATION_CRITERIA.MAX_ITERATIONS,TERMINATION_CRITERIA.EPSILON};
	private int iterations = 5;
	private float epsilon  = 0.01f;

	public TerminationCriteria(TERMINATION_CRITERIA[] criteria, int iterations, float epsilon) {
		terminationCriteria = criteria;
		this.iterations = iterations;
		this.epsilon = epsilon;
	}

	public TerminationCriteria() {}

	public TerminationCriteria(TERMINATION_CRITERIA[] criteria) {
		terminationCriteria = criteria;
	}

	public TERMINATION_CRITERIA[] getTerminationCriterias() {
		return terminationCriteria;
	}

	public int getIterations() {
		return iterations;
	}

	public float getEpsilon() {
		return epsilon;
	}

	public void setCriteria(TERMINATION_CRITERIA[] criteria) {
		terminationCriteria = criteria;		
	}

	public void setIterations(int it) {
		iterations = it;
	}

	public void setEpsilon(float eps) {
		this.epsilon = eps;		
	}
}
