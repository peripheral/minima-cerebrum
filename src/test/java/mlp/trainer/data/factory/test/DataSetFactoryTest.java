package mlp.trainer.data.factory.test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.TreeMap;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlp.trainer.data.Data;
import mlp.trainer.data.TrainingData;
import mlp.trainer.data.ValidationData;
import mlp.trainer.data.factory.DataSetFactory;

public class DataSetFactoryTest extends Data{
	private DataSetFactory sut;
	
	@BeforeEach
	void init() {
		sut = new DataSetFactory();
	}
	
	@Test
	void testCreateTrainingSet() {
		Scanner inScanner = new Scanner("");
		int outputLayerSize = 3;
		double portion = 0.5;
		List<float[]> inputs = new LinkedList<>();
		List<float[]> targets = new LinkedList<>();
	
		TreeMap<String,Integer> labelIdxMap = new TreeMap<>();
		try {
			inScanner = new Scanner(new File("testData\\iris.data.txt"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			inScanner.close();
		}
		String[] line;
		while(inScanner.hasNext()) {
			line = inScanner.nextLine().split(",");
			inputs.add(convertToInputs(line));
			targets.add(createOutputRow(line[line.length-1],labelIdxMap,outputLayerSize));	
		}
		int expected = (int) (inputs.size() *portion);
		
		sut.setInput(inputs);
		sut.setTarget(targets);
		sut.setClassCount(3);
		TrainingData td = sut.getTrainingData();	
		int actual = td.size();
		assertEquals(expected,actual);		
	}
	
	@Test
	void testCreateValidationSet() {
		Scanner inScanner = new Scanner("");
		int outputLayerSize = 3;
		double portion = 0.25;
		List<float[]> inputs = new LinkedList<>();
		List<float[]> targets = new LinkedList<>();
	
		TreeMap<String,Integer> labelIdxMap = new TreeMap<>();
		try {
			inScanner = new Scanner(new File("testData\\iris.data.txt"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			inScanner.close();
		}
		String[] line;
		while(inScanner.hasNext()) {
			line = inScanner.nextLine().split(",");
			inputs.add(convertToInputs(line));
			targets.add(createOutputRow(line[line.length-1],labelIdxMap,outputLayerSize));	
		}
		int expected = (int) (inputs.size() *portion);
		
		sut.setInput(inputs);
		sut.setTarget(targets);
		sut.setClassCount(3);
		ValidationData vd = sut.getValidationData();	
		int actual = vd.size();
		assertEquals(expected,actual);		
	}
	
	
	
	private float[] createOutputRow(String value, TreeMap<String, Integer> lableToIndxMap,
			int outputLayerSize) {
		float[] outputVector = new float[outputLayerSize];
		if(lableToIndxMap.containsKey(value)) {
			outputVector[lableToIndxMap.get(value)] = 1f;
		}else {
			lableToIndxMap.put(value, lableToIndxMap.size());
		}
		return outputVector;
	}


	private float[] convertToInputs(String[] line) {
		float[] data = new float[line.length-1];
		for (int i = 0; i < data.length; i++) {
			data[i] = Float.valueOf(line[i]);
		}
		return data;
	}
}
