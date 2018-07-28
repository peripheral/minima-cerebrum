package mlp.trainer;

public class TerminationCriteria {
	public enum TERMINATION_CRITERIA {
		EPSILON, MAX_ITERATIONS, MIN_ERROR_DELTA
	}

	public TerminationCriteria(TERMINATION_CRITERIA[] criteria, int i, double d) {
		// TODO Auto-generated constructor stub
	}

	public TerminationCriteria() {
		// TODO Auto-generated constructor stub
	}

	public TERMINATION_CRITERIA[] getTerminationCriterias() {
		// TODO Auto-generated method stub
		return null;
	}

	public int getIterations() {
		// TODO Auto-generated method stub
		return 0;
	}

	public float getEpsilon() {
		// TODO Auto-generated method stub
		return 0;
	}

	public void setCriteria(TERMINATION_CRITERIA[] expected) {
		// TODO Auto-generated method stub
		
	}

	public void setIterations(int expectedIterations) {
		// TODO Auto-generated method stub
		
	}

	public void setEpsilon(float expectedEpsilon) {
		// TODO Auto-generated method stub
		
	}
}
