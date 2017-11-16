package mlp;

import static org.junit.Assert.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ANN_MLPTest {
	private ANN_MLP sut;
	
	@BeforeEach
	void init() {
		sut = new ANN_MLP();
	}
	
	@Test
	void testOnConstructorWithArray() {
		int[] actualLayerSizes = {2,3,1};
		sut = new ANN_MLP(actualLayerSizes);
		actualLayerSizes = sut.getLayerSizes();
		int[] expectedLayerSizes = {2,3,1};
		assertArrayEquals(actualLayerSizes,expectedLayerSizes);
	}
	
	@Test
	void testLengthOfCreatedLayers() {
		int[] layerSizes = {2,3,1};
		sut = new ANN_MLP(layerSizes);
		int[] actualInputLayerSizes = new int[3];
		actualInputLayerSizes[0] = sut.getInputLayer().size();
		actualInputLayerSizes[1] = sut.getLayer(1).size();
		actualInputLayerSizes[2] = sut.getOutputLayer().size();
		int[] expected =  {2,3,1};
		assertArrayEquals(expected,actualInputLayerSizes);
	}
	
}
