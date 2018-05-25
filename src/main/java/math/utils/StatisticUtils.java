package math.utils;

public class StatisticUtils {

	public static float variance(float[] values) {
		double mean = mean(values);
		double variance = 0;
		for(int i = 0;i < values.length;i++) {
			variance = variance + Math.pow((values[i]-mean),2)/values.length;
		}
		return (float) variance;
	}

	public static double mean(float[] values) {
		double mean = 0;
		for(int i = 0;i < values.length;i++) {
			mean = mean + values[i]/values.length;
		}
		return mean;
	}
}
