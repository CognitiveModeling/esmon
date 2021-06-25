package esn;

import java.util.Random;

import Jama.Matrix;

/**
 * This class represents the back-projection layer of the esn's.
 * @author Johannes Lohmann, Danil Koryakin
 *
 */
public class BackLayer extends Layer
{
	/**
	 * Constructs a new back-projection layer. It is not parameterized with a double array
	 * for the value-bounds of the weights of this layer, but they can be set if necessary.
	 * @param rows: number of rows of the weight matrix (number of reservoir neurons)
	 * @param cols: number of columns of the weight matrix (number of output neurons)
	 * @param seed, value for seeding generators of random numbers
	 */
	public BackLayer(int rows, int cols, int seed)
	{
		super(rows, cols, 0, null);
		
		_weights      = new Matrix(rows, cols);
		_weights_init = new Matrix(rows, cols);
		_seed         = seed;
	}
	
	/**
	 * Depending on the values stored in the bound array, the weights are initialized with
	 * random, equal distributed values.
	 */
	@Override
	public void initializeWeights()
	{
		int    i, j;
		int    num_row, num_col;//number of rows and columns is a weight matrix
		double rnd_val;//generated random value
		
		Random rand = new Random(_seed);

		num_row = _weights.getRowDimension();
		num_col = _weights.getColumnDimension();
		for(i=0; i < num_row; i++)
		{
			for(j=0; j < num_col; j++)
			{
				rnd_val = _bounds[0] + Math.abs(_bounds[1] - _bounds[0]) * rand.nextDouble();
				_weights.set(i, j, rnd_val);
			}
		}

		//store reservoir weights for the activation later
		storeWeightsInit();
	}
	
	/**
	 * The function assigns weights to OFB connections of active modules. OFB weights are set to "0".
	 * Active modules are indicated in a provided Boolean array.
	 * 
	 * The procedure has different realizations in all classes the "InputLayer", "OutputLayer", "InternalLayer" and
	 * "BackLayer".
	 * 
	 * @param sub_neurons: indices of reservoir neurons which are arranged by sub-reservoir
	 * @param active_modules: indicator array of active modules
	 */
	public void setActiveWeights()
	{
		//assign scaled weights of current module
		_weights.setMatrix(0, _weights_init.getRowDimension()-1, 0, _weights_init.getColumnDimension()-1, _weights_init);
	}
	
	/**
	 * This is a dummy function to match the interface of the class Layer.
	 * The current class BackLayer does not have own nodes.
	 */
	public void generateInitState()
	{}
	
	/**
	 * This is a dummy function to match the interface of the class Layer.
	 * The current class BackLayer does not have own nodes.
	 */
	public void assignInitState()
	{}
}
