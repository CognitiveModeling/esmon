package esn;

import Jama.Matrix;

/**
 * This class represents the output layer of the esn's. A flag indicates if the nodes of
 * this layer are interconnected and self-recurrent.
 * @author Johannes Lohmann, Danil Koryakin
 */
public class OutputLayer extends Layer
{
	/**
	 * This flag indicates if the nodes of the output layer should be connected.
	 */
	private boolean  _selfrecurrence = false;
	private boolean  _train_ok;//indicator of whether output weights are trained
	
	/**
	 * Constructs a new output layer. It is not parameterized with a double array
	 * for the value-bounds of the weights of this layer, as output layers are initialized
	 * with weights that are all zero, as these weights will be trained.
	 * @param rows: number of rows of the weight matrix (number of output neurons)
	 * @param cols: number of columns of the weight matrix (number of neurons where the output connections come from)
	 * @param seed: value for seeding generator of random numbers
	 */
	public OutputLayer(int rows, int cols, int seed)
	{		
		super(rows, cols, rows, null);
		
		_weights    = new Matrix(_rows, _cols);
		_weights_init = new Matrix(_rows, _cols);
		_train_ok     = false;
		_seed         = seed;
	}
	
	/**
	 * This function generates a set of zero initial states for nodes of the output layer.
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
	 * The function sets the states of all output neurons to their predefined initial values. The set of the initial
	 * values was defined in the constructor of the class or was provided from the outside.
	 */
	public void assignInitState()
	{
		_nodes.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1,
				         _init_state);
	}
	
	/**
	 * This is a dummy function to match a generic interface of the class Layer.
	 * There is no need to initialize these weights. They will be trained.
	 * 
	 * @param f_generate: unused parameter which is defined for compatibility with the Layer class
	 */
	@Override
	public void initializeWeights()
	{}
	
	/**
	 * The function assigns provided weights to connections of specified neurons.
	 * The neurons are specified by their indices
	 * 
	 * @param weights_module: provided weights
	 * @param sub_neuron: indices of neurons
	 */
	public void setWeightNeurons(Matrix weights_module, int[] sub_neuron)
	{
		int   i;
		int[] idx_row;//indices of rows for copying the matrices
		
		idx_row = new int[_weights.getRowDimension()];
		for(i=0; i<idx_row.length; i++)
		{
			idx_row[i] = i;
		}
		_weights.setMatrix(idx_row, sub_neuron, weights_module);
	}

	/**
	 * Getter for the flag indicating the recurrence of this layer.
	 * @return: indicator of the self-recurrence
	 */
	public boolean isSelfReccurent()
	{
		return _selfrecurrence;
	}
	
	/**
	 * Setter for the flag indicating the recurrence of this layer.
	 * @param the selfrecurrence to be set.
	 */
	public void setSelfRecurrent(boolean selfrecurrence) {
		this._selfrecurrence = selfrecurrence;
	}
	
	/**
	 * The function assigns the trained output weights of specified active modules.
	 * Output weights of inactive modules are set to "0".
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
	 * The function assigns previously stored states of all output neurons.
	 */
	public void restoreOutput()
	{
		_nodes.setMatrix(0, _rows-1, 0, _cols-1, _nodes_init);
	}
	
	/**
	 * The function indicates whether output weights are trained.
	 *
	 * @return: "true" if weights are trained, "false" otherwise
	 */
	public boolean isTrained()
	{
		return _train_ok;
	}
	
	/**
	 * The function sets an indicator that the output weights are trained.
	 */
	public void setTrained()
	{
		_train_ok = true;
	}
	
	/**
	 * The function clears an indicator that the output weights are trained.
	 */
	public void resetTrained()
	{
		_train_ok = false;
	}
}
