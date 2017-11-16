package mlp;

public class ANN_MLP {
	
	private NeuronLayer[] layers = null;
	public ANN_MLP() {}

	public ANN_MLP(int[] layerSizes) {
		layers = new NeuronLayer[layerSizes.length];
		layers[0] = new NeuronLayer(layerSizes[0]);
		for(int i = 1;i < layerSizes.length-1;i++) {
			layers[i] = new NeuronLayer(layerSizes[i]);
		}
		layers[layers.length-1] = new NeuronLayer(layerSizes[layers.length-1]);
	}

	public static enum ACTIVATION_FUNCTION{SIGMOID, GAUSSIAN}

	public int[] getLayerSizes() {
		int[] layerSizes = new int[layers.length];
		for(int i = 0; i < layerSizes.length;i++) {
			layerSizes[i] = layers[i].size();
		}
		return layerSizes;
	}

	public NeuronLayer getInputLayer() {
		return layers[0];
	}

	public NeuronLayer getLayer(int i) {
		return layers[i];
	}

	public NeuronLayer getOutputLayer() {
		return layers[layers.length-1];
	}

}
