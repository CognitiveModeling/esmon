package esn;

import java.util.Random;

import Jama.Matrix;

/**
 * This class represents the input layer of an esn.
 * @author Johannes Lohmann, Danil Koryakin
 *
 */
public class InputLayer extends Layer
{
	/**
	 * Constructs a new input layer. It is not parameterized with a double array
	 * for the value-bounds of the weights of this layer, but they can be set if necessary.
	 * @param rows: number of rows of the weight matrix (number of reservoir neurons)
	 * @param cols: number of columns of the weight matrix (number of input neurons)
	 * @param seed: value for seeding generators of random numbers
	 */
	public InputLayer(int rows, int cols, int seed)
	{
		super(rows, cols, cols, null);

		_seed = seed;
	    
		_weights = new Matrix(new double[_rows][_cols]);
		_weights_init = new Matrix(new double[_rows][_cols]);
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
	 * The function assigns weights to all input connections.
	 * Since input connections are connected to all modules, there is no need for indicator array of active modules.
	 * The indicator array is used as a dummy parameter as well.
	 * 
	 * The procedure has different realizations in all classes the "InputLayer", "OutputLayer", "InternalLayer" and
	 * "BackLayer".
	 */
	public void setActiveWeights()
	{
		_weights.setMatrix(0, _weights_init.getRowDimension()-1, 0, _weights_init.getColumnDimension()-1,
				           _weights_init);
	}
	
	/**
	 * This function generates a set of zero initial states for nodes of the input layer.
	 * Since, at each time step, input nodes are assigned an external vector from the outside, the initial states are
	 * actually useless for the input layer.
	 */
	public void generateInitState()
	{
		int i,j;
		
		for(i=0; i<_init_state.getRowDimension(); i++)
		{
			for(j=0; j<_init_state.getColumnDimension(); j++)
			{
				_init_state.set(i, j, 0);
			}
		}
	}
	
	/**
	 * The function assigns preliminary prepared initial states to nodes of the input layer.
	 */
	public void assignInitState()
	{
		_nodes.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1,
				         _init_state);
	}
}
