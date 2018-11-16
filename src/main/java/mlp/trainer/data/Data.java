package mlp.trainer.data;

/**
 * General data holder for input and target values
 * @author 
 *
 */
public class Data {
	protected float[][] input;
	protected float[][] target;
	
	public Data() {}
	
	/**
	 * @param inputs - represents x vector
	 * @param targets - represents y vector
	 */
	public Data(float[][] inputs,float[][] targets) {
		this.input = inputs;
		this.target= targets;
	}
	
	public float[][] getInput() {
		return input;
	}
	public void setInput(float[][] input) {
		this.input = input;
	}
	public float[][] getTarget() {
		return target;
	}
	public void setTarget(float[][] target) {
		this.target = target;
	}
	
	public int size() {
		return input.length;
	}
}
