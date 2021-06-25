package esn;

import java.util.Vector;

import types.interval_C;

import esn.Activation;
import esn.Activation.activation_E;
import esn.Module.storage_type_E;
import experiment.ExpParam.exp_param_E;

import Jama.Matrix;

/**
 * The abstract layer class. It is the parent of all other classes representing a layer.
 * Most of the relevant attributes and methods are defined here. The only method remaining
 * abstract is called initializeWeights as the different layer instances might have different
 * rules applying for the generation of their weight matrices.
 * @author Johannes Lohmann, Danil Koryakin
 *
 */
public abstract class Layer
{	
	public enum leakage_assign_E
	{
		LA_SAME,  //the same leakage rate is assigned to all reservoir neurons
		LA_RANDOM,//assigned leakage rates are random values from a given interval
		LA_NONE;  //enumeration value which corresponds to none of available rule of leakage rate assignment
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "LA_NONE", if the provided string was not found in any of the enumeration values
		 */
		public static leakage_assign_E fromString(String str)
		{
			leakage_assign_E type;//output variable
			
			type = LA_NONE;
			for(leakage_assign_E value : leakage_assign_E.values())
			{
				if(value.name().contains(str)==true)
				{
					type = value;
				}
			}
			if(type==LA_NONE)
			{
				System.err.println("leakage_assign_E.fromString: invalid string of leakage assignment rule");
				System.exit(1);
			}
			
			return type;
		}
	};
	
	protected Activation[] _func;//activation functions of layer's neurons
	protected Activation[] _func_inverse;//inverse activation functions of layer's neurons
	protected Matrix   _nodes;//matrix (in fact a vector) representing the nodes
	protected Matrix   _nodes_init;//stored initial states of all neurons
	protected Matrix   _nodes_tmp;//stored temporary states of all neurons
	protected Matrix   _nodes_zero;//zero matrix to clear states of neurons
	protected Matrix[] _nodes_history;//array of several node states
	protected Matrix   _init_state;//array of random values for initialization of the neuron states
	protected Matrix   _weights;//matrix representing the weights of a layer
	protected Matrix   _weights_init;//weights which are used for activation of all sub-reservoirs
	protected Matrix   _weights_zero;//zero matrix to clear weights
	protected Matrix   _bias;//input bias of the layer's neurons
	protected double[] _bounds;//bound values indicating the maximal and minimal values for weights
	protected double   _leakage_rate[];
	protected int      _rows;//number of rows of the weight matrix
	protected int      _cols;//number of columns of the weight matrix
	protected int      _seed;//seeding value of an ESN layer
	protected int      _idx_history;//index of an element in the history array where the next value can be stored
	protected int      _num_neurons;//number of neurons
	protected interval_C[][] _valid_range;//min and max values on negative and positive intervals for each node's output
	
	protected final int _num_valid_intervals_TANH = 2;//number of intervals to compose a valid range of node with TANH
	protected final int _num_valid_intervals_LOG= 1;//number of intervals to compose a valid range of node with LOGISTIC
	protected final int _num_valid_intervals_ID   = 1;//number of intervals to compose a valid range of node with ID
	
	/**
	 * The default constructor for any layer.
	 * @param rows: the number of rows of the weight matrix.
	 * @param cols: the number of columns of the weight matrix.
	 * @param num_node: number of neurons in the layer
	 * @param bounds: the range for the values of the weights.
	 */
	protected Layer(int rows, int cols, int num_node, double [] bounds)
	{
		int i;
		
		_rows          = rows;
		_cols          = cols;
		_num_neurons   = num_node;
		_nodes_zero    = new Matrix(_rows, 1);
		_weights       = null;
		_weights_zero  = new Matrix(_rows, _cols);
		_nodes         = new Matrix(num_node, 1);
		_nodes_history = null;
		_idx_history   = 0;
		_nodes_init    = new Matrix(num_node, 1);
		_nodes_tmp     = new Matrix(num_node, 1);
		_init_state    = new Matrix(num_node, 1);
		
		_bias          = new Matrix(num_node, 1);
		_func          = new Activation[num_node];
		_func_inverse  = new Activation[num_node];
		_valid_range   = new interval_C[num_node][];
		_leakage_rate  = new double[num_node];
		for(i=0; i<num_node; i++)
		{
			_func[i]         = new Activation();
			_func_inverse[i] = new Activation();
			_leakage_rate[i] = 1.0;
			_valid_range[i]  = null;//it cannot be assigned because size depends on activation nodes' function
			                        //that are unknown by now
		}
		
		generateInitState();
		assignInitState();
	}
	
	/**
	 * Abstract method that governs initialization of a weight matrix of a specific layer.
	 */
	public abstract void initializeWeights();
	
	/**
	 * The function should implement generation of initial states of nodes.
	 */
	public abstract void generateInitState();
	
	/**
	 * The function assigns previously prepared initial states to nodes of a layer.
	 */
	public abstract void assignInitState();
	
	/**
	 * Abstract method to store the layer's weights whose non-zero values will be used to activate the required
	 * sub-reservoirs.
	 * The function must have the scope "public" because the trained output weights are stored after a call of
	 * "OutputLayer.storeWeightActivation" from the external class "ESN".
	 */
	//public abstract void storeWeightActivation();
	/**
	 * Abstract method to restore the layer's weights whose non-zero values are used in a configured module.
	 * The function must have the scope "public" because the output weights, the OFB weights and the input weights
	 * are restored after a call from the external class "ExpRun". 
	 */
	public abstract void setActiveWeights(boolean do_scale);
	
	/**
	 * The function allocates a valid range of a specified node depending on its activation function
	 * 
	 * @param idx_node: index of a node
	 */
	private void allocateValidRange(int idx_node)
	{
		int i;
		int num_valid_intervals;
		
		switch(_func[idx_node].getActivation())
		{
			case TANH:
				num_valid_intervals = _num_valid_intervals_TANH;
				break;
			case LOGISTIC:
				num_valid_intervals = _num_valid_intervals_LOG;
				break;
			case ID:
				num_valid_intervals = _num_valid_intervals_ID;
				break;
			default:
				num_valid_intervals = -1;
				System.err.println("Layer.allocateValidRange: undefined number of intervals for specified activation");
				break;
		}
		
		_valid_range[idx_node] = new interval_C[num_valid_intervals];
		for(i=0; i<num_valid_intervals; i++)
		{
			_valid_range[idx_node][i] = new interval_C();
			_valid_range[idx_node][i].setLeftBorder(Double.POSITIVE_INFINITY);
			_valid_range[idx_node][i].setRightBorder(Double.NEGATIVE_INFINITY);
		}
	}
	
	/**
	 * Overrides the toString() method to obtain a string representation of the
	 * weight matrix, this is used to store layers of an esn in files.
	 */
	public String toString() {
		StringBuffer matrix = new StringBuffer("null");
		
		if (this._weights != null) {
			matrix.setLength(0);
			for (double[] row : this._weights.getArray()) {
				for (double val : row) {
					matrix.append(val);
					matrix.append("\t");
				}
				matrix.replace(matrix.length() - 1, matrix.length(), "\n");
			}
			matrix.replace(matrix.length() - 1, matrix.length(), "");
			
		}
		
		return matrix.toString();
	}
	
	/**
	 * Returns a string representation of the current activation of the nodes of a
	 * layer.
	 * @return String, a string holding the activation values of the nodes of this layer.
	 */
	public String getNodeString() {
		StringBuffer sb = new StringBuffer("null");
		
		if (this._nodes != null) {
			sb.setLength(0);
			for (double[] row : this._nodes.getArray()) {
				for (double val : row) {
					sb.append(val);
					sb.append("\t");
				}
				sb.replace(sb.length() - 1, sb.length(), "\n");
			}
			sb.replace(sb.length() - 1, sb.length(), "");
		}
		
		return sb.toString();
	}
	
	/**
	 * The function gets a seeding value of random number generators of an ESN layer.
	 * 
	 * @return: seeding value of random number generators
	 */
	public int getSeed()
	{
		return _seed;
	}
	
	/**
	 * The function assigns a provided seeding value to an ESN layer.
	 * 
	 * @param seed: provided seeding value
	 */
	public void setSeed(int seed)
	{
		_seed = seed;
	}
	
	/**
	 * This function stores current neuron states for their possible activation later.
	 * The neuron states are stored in a specified storage. It is specified by its type.
	 * 
	 * @param type: storage type
	 */
	public void storeNodes(storage_type_E type)
	{
		switch(type)
		{
			case ST_INIT:
				_nodes_init.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1, _nodes);
				break;
			case ST_TMP:
				_nodes_tmp.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1, _nodes);
				break;
			default:
				System.err.println("Layer.storeNodes: unknown storage type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function restores previously stored states of layer's nodes. The states are retried from a specified storage.
	 * The storage is specified by its type.
	 * 
	 * @param type: storage type
	 */
	public void restoreNodes(storage_type_E type)
	{
		switch(type)
		{
			case ST_INIT:
				_nodes.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1, _nodes_init);
				break;
			case ST_TMP:
				_nodes.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1, _nodes_tmp);
				break;
			default:
				System.err.println("Layer.restoreNodes: unknown storage type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function returns an array with the currently used weights.
	 * @return: array of the currently used weights
	 */
	public Matrix getWeights()
	{
		return _weights;
	}
	
	/**
	 * The function returns a matrix of initially generated weights of this layer.
	 * @return: matrix of initially generated weights
	 */
	public Matrix getWeightsInit()
	{
		return _weights_init;
	}
	
	/**
	 * The function returns an array with biases of all neurons.
	 * @return: 
	 */
	public Matrix getBias()
	{
		return _bias;
	}
	
	/**
	 * The function assigns new values of biases to all layer's neurons. 
	 * @param bias: array of new values of biases
	 */
	public void setBias(Matrix bias)
	{
		_bias.setMatrix(0, bias.getRowDimension()-1, 0, 0, bias);
	}
	
	/**
	 * The function assigns a provided value as a bias of a specified node.
	 * The node is specified by its index.
	 * If "-1" is provided as a provided node index then the bias is assigned to all nodes of a layer.
	 *  
	 * @param bias: provided bias value
	 * @param node_idx: provided node index
	 */
	public void setBias(double bias, int node_idx)
	{
		int i;
		int num_nodes;

		if(node_idx!=-1)
		{
			_bias.set(node_idx, 0, bias);
		}
		else
		{
			num_nodes = _nodes.getRowDimension();
			for(i=0; i<num_nodes; i++)
			{
				_bias.set(i, 0, bias);
			}
		}
	}
	
	/**
	 * The function indicates activation functions for all nodes of a layer.
	 * 
	 * @return: array with activation functions
	 */
	public Activation[] getActivation()
	{
		return _func;
	}
	
	/**
	 * The function assigns a provided activation function to the node which is specified by a provided index.
	 * If "-1" is provided as a provided node index then the activation is assigned to all nodes of a layer.
	 * 
	 * @param activation: activation to be assigned
	 * @param node_idx: provided node index
	 */
	public void setActivation(activation_E activation, int node_idx)
	{
		int i;
		
		if(node_idx!=-1)
		{
			_func[node_idx].setActivation(activation);
			_func_inverse[node_idx].setActivation( _func[node_idx].getInverseActivation() );
		}
		else
		{
			for(i=0; i<_num_neurons; i++)
			{
				_func[i].setActivation(activation);
				_func_inverse[i].setActivation( _func[i].getInverseActivation() );
			}
		}
	}
	
	/**
	 * The function assigns activation functions from a provided array to all nodes of a layer.
	 * The function issues an error message if a number of provided activation functions does not equal
	 * to a number of nodes in the layer.
	 * 
	 * @param activation: array with activation functions
	 */
	public void setActivation(Activation[] activation)
	{
		int i;
		
		//check the input
		if(activation.length!=_num_neurons)
		{
			System.err.println("Layer.setActivation: mismatch between provided activations and available nodes");
			System.exit(1);
		}
		
		for(i=0; i<_num_neurons; i++)
		{
			_func[i].setActivation(activation[i].getActivation());
			if(_func[i].equals(activation_E.LOGISTIC))
			{
				_func[i].setLogisticParam(activation[i].getLogisticParam());
			}
			_func_inverse[i].setActivation(_func[i].getInverseActivation());
		}
	}
	
	/**
	 * The function assigns activation functions from a provided array to all nodes of a layer.
	 * The function issues an error message if a number of provided activation functions does not equal
	 * to a number of nodes in the layer.
	 * 
	 * @param activation: array with activation functions
	 */
	public void setActivation(activation_E[] activation)
	{
		int i;
		
		//check the input
		if(activation.length!=_num_neurons)
		{
			System.err.println("Layer.setActivation: mismatch between provided activations and available nodes");
			System.exit(1);
		}
		
		for(i=0; i<_num_neurons; i++)
		{
			_func[i].setActivation(activation[i]);
			_func_inverse[i].setActivation(activation_E.getInverseFunction(activation[i]));
		}
	}
	
	/**
	 * The function returns a number of neurons in a layer.
	 * 
	 * @return: number of layer's neurons
	 */
	public int getNumNodes()
	{
		return _nodes.getRowDimension();
	}

	/**
	 * Getter for the number of rows of the weight matrix of this layer.
	 * @return the rows.
	 */
	public int getRows() {
		return _rows;
	}

	/**
	 * Setter for the number of rows of the weight matrix of this layer.
	 * @param rows the rows to set.
	 */
	public void setRows(int rows) {
		this._rows = rows;
	}

	/**
	 * Getter for the number of columns of the weight matrix of this layer.
	 * @return the cols.
	 */
	public int getCols() {
		return _cols;
	}

	/**
	 * Setter for the number of columns of the weight matrix of this layer.
	 * @param cols the cols to set.
	 */
	public void setCols(int cols) {
		this._cols = cols;
	}
	
	/**
	 * The function sets states of all weights of a layer to "0".
	 */
	public void clearWeights()
	{
		_weights.setMatrix(0, _rows-1, 0, _cols-1, _weights_zero);
	}

	/**
	 * Setter for the weight matrix of this layer.
	 * @param weights: array of new values of weights
	 */
	public void setWeights(Matrix weights)
	{
		int i, j;
		
		if(_weights.getRowDimension()   !=weights.getRowDimension() ||
		   _weights.getColumnDimension()!=weights.getColumnDimension())
		{
			System.err.println("setWeights: number of new values does not match number of weights");
			System.exit(1);
		}
		
		for(i=0; i<weights.getRowDimension(); i++)
		{
			for(j=0; j<weights.getColumnDimension(); j++)
			{
				_weights.set(i, j, weights.get(i, j));
			}
		}
	}
	
	/**
	 * The function assigns initial weights of this layer a provided matrix.
	 * @param weights: array of new values of initial weights
	 */
	public void setWeightsInit(Matrix weights)
	{
		_weights_init.setMatrix(0, _weights_init.getRowDimension()-1,
				                0, _weights_init.getColumnDimension()-1, weights);
	}
	
	/**
	 * Another setter for the weights of this layer. The second parameter represents
	 * the connectivity of this layer: If a value is false the weight is set to zero,
	 * to its value otherwise.
	 * @param weights, the matrix containing the values for the weights.
	 * @param connections, a boolean array representing the existing connections.
	 */
	public void setWeigths(Matrix weights, boolean[] connections) {
		if (connections.length == weights.getColumnDimension() * weights.getRowDimension()) {
			int count = 0;
			double[][] newWeights = new double[weights.getArray().length]
			                                   [weights.getArray()[0].length];
			for (int i = 0; i < weights.getArray().length; i++) {
				for (int j = 0; j < weights.getArray()[i].length; j++) {
					if (connections[count]) {
						newWeights[i][j] = weights.getArray()[i][j];
					} else {
						newWeights[i][j] = 0.0D;
					}
					count++;
				}
			}
			
			this._weights = new Matrix(newWeights);
		} else {
			this._weights = weights;
		}
	}
	
	/**
	 * This function stores current weights for their possible activation later.
	 */
	public void storeWeightsInit()
	{
		_weights_init.setMatrix(0, _weights.getRowDimension()-1, 0, _weights.getColumnDimension()-1, _weights);
	}
	
	/**
	 * The function restores previously stored layer's weights.
	 */
	public void restoreWeightsInit()
	{
		_weights.setMatrix(0, _weights.getRowDimension()-1, 0, _weights.getColumnDimension()-1, _weights_init);
	}

	/**
	 * Getter for the value bounds of this layer's weight matrix.
	 * @return the bounds.
	 */
	public double[] getWeightBounds()
	{
		return _bounds;
	}

	/**
	 * Setter for the value bounds of this layer's weight matrix.
	 * @param bounds the bounds to set.
	 */
	public void setWeightBounds(double[] bounds)
	{
		this._bounds = bounds;
	}
	
	/**
	 * The function sets states of all layer's nodes to "0".
	 */
	public void clearNodes()
	{
		_nodes.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1, _nodes_zero);
	}

	/**
	 * Getter for the nodes of this layer.
	 * @return the nodes.
	 */
	public Matrix getNodes() {
		return _nodes;
	}
	
	/**
	 * The function retrieves values of the nodes whose indices are specified in the provided index array.
	 * The retrieved values are stored in the provided storage array.
	 * Sizes of the provided arrays must be equal.
	 * 
	 * @param node_idx: provided index array
	 * @param node_val: provided storage
	 */
	public void getNodes(int[] node_idx, double[] node_val)
	{
		int i;
		int len;//length of the provided arrays
		
		//check the provided input
		if(node_idx.length != node_val.length)
		{
			System.err.println("Layer.getNodes: provided arrays have difefrent numbers of elements");
			System.exit(1);
			len = -1;
		}
		else
		{
			len = node_idx.length;
		}
		
		//nodes must always be arranged in a column
		if(_nodes.getColumnDimension() > 1)
		{
			System.err.println("Layer.setNodes: array to get values is not a column");
			System.exit(1);
		}
		
		for(i=0; i<len; i++)
		{
			node_val[i] = _nodes.get(node_idx[i], 0);
		}
	}
	
	/**
	 * The function stores states of nodes in a storage array which is pointed by a provided identifier.
	 *
	 * @param storage: identifier of the storage array
	 */
	public void getNodes(double[] storage)
	{
		int i;
		int len;//length of the provided arrays
		
		len = _nodes.getRowDimension();
		for(i=0; i<len; i++)
		{
			storage[i] = _nodes.get(i, 0);
		}
	}

	/**
	 * The function assigns values from a provided matrix to states of a layer's nodes.
	 * If it is required, an activation function is applied to an input value before its assignment.
	 * 
	 * @param nodes: matrix with provided values
	 * @param apply_activation: request to apply an activation function
	 */
	public void setNodes(Matrix nodes, boolean apply_activation)
	{
		int i;
		int num_nodes;
		double value_to_assign;
		
		num_nodes = _nodes.getRowDimension();
		if(apply_activation==true)
		{
			for(i=0; i<num_nodes; i++)
			{
				value_to_assign = _func[i].calculateValue( nodes.get(i, 0) );
				_nodes.set(i, 0, value_to_assign);
			}
		}
		else
		{
			_nodes.setMatrix(0, num_nodes-1, 0, 0, nodes);
		}
	}
	
	/**
	 * The function assigns values from a provided array to states of layer's nodes.
	 * If it is required, an activation function is applied to an input value before its assignment.
	 * 
	 * @param node: values to be assigned
	 * @param apply_activation: request to apply an activation function
	 */
	public void setNodes(double[] node, boolean apply_activation)
	{
		int i;
		int num_nodes;//number of nodes
		double value_to_assign;
		
		num_nodes = node.length;
		if(apply_activation==true)
		{
			for(i=0; i<num_nodes; i++)
			{
				value_to_assign = _func[i].calculateValue( node[i] );
				_nodes.set(i, 0, value_to_assign);
			}
		}
		else
		{
			for(i=0; i<num_nodes; i++)
			{
				_nodes.set(i, 0, node[i]);
			}
		}
	}
	
	/**
	 * The function creates a node history of a required length.
	 * 
	 * @param len: required length of a history
	 */
	public void createNodesHistory(int len)
	{
		_nodes_history = new Matrix[len];
		_idx_history = 0;
	}
	
	/**
	 * The function sets an array with a node history to NULL.
	 */
	public void deleteNodesHistory()
	{
		_nodes_history = null;
		_idx_history = 0;
	}
	
	/**
	 * The function stores all elements of a history of the layer's nodes in a provided array.
	 * The function allocates each element in the provided array.
	 */
	public void getNodesHistory(Matrix[] storage)
	{
		int i;
		
		for(i=0; i<_nodes_history.length; i++)
		{
			storage[i] = _nodes_history[i].copy();
		}
	}
	
	/**
	 * The function stores all elements of a history of the layer's nodes in a provided array.
	 */
	public void getNodesHistory(double[][] storage)
	{
		int i, j;
		double[][] tmp_history;//temporary storage to keep a storage as a array of doubles
		
		for(i=0; i<_nodes_history.length; i++)
		{
			tmp_history = _nodes_history[i].getArray();
			for(j=0; j<_num_neurons; j++)
			{
				storage[i][j] = tmp_history[j][0];
			}
		}
	}
	
	/**
	 * The function indicates states of the layer's nodes that were stored in a specified element of the history array.
	 * The element is specified by its index. 
	 * 
	 * @param index: index of required element
	 */
	public Matrix getNodesHistory(int index)
	{
		return _nodes_history[index];
	}
	
	/**
	 * The function stores an element of the history array in a provided storage.
	 * The element is specified by its index.
	 * 
	 * @param index: index of a required element
	 * @param storage: provided storage
	 */
	public void getNodesHistory(int index, double[] storage)
	{
		int i;
		
		for(i=0; i<_num_neurons; i++)
		{
			storage[i] = _nodes_history[index].get(i, 0);
		}
	}
	
	/**
	 * The function resets a counter of elements in a history array for storing the next elements. 
	 */
	public void startNodesHistory()
	{
		_idx_history = 0;
	}
	
	/**
	 * The function stores current states of the layer's nodes as the last element in the history array.
	 * If the history array is full, the history array is shifted by one element first. Then the layer's nodes
	 * are stored as the last element.
	 * Under the shift, the first element of the array is discarded; and the 2nd element becomes the 1st one.
	 */
	public void storeNodesHistory()
	{
		//store as the last element
		if(_nodes_history[_idx_history]==null)
		{
			_nodes_history[_idx_history] = _nodes.copy();
		}
		else
		{
			_nodes_history[_idx_history].setMatrix(0, _nodes.getRowDimension()-1,
					                               0, _nodes.getColumnDimension()-1, _nodes);
		}
		
		//update a counter of stored elements if necessary
		_idx_history++;
		if(_idx_history==_nodes_history.length)
		{
			_idx_history = 0;
		}
	}
	
	/**
	 * The function assigns values of a specified history element to nodes of a layer.
	 * The history element is specified by its index.
	 */
	public void restoreNodesHistory(int element_idx)
	{
		_nodes.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1,
                         _nodes_history[element_idx]);
	}

	/**
	 * The function converts a matrix of the initial weights of the host layer to an array of strings.
	 * 
	 * @return: layer's initial weights as an array of strings
	 */
	public Vector<String> getInitWeightsAsStr()
	{
		int i, j;
		String str;
		Vector<String> w;
		
		w = new Vector<String>(0,1);
		
		for(i=0; i<_weights_init.getRowDimension(); i++)
		{
			str = "";
			for(j=0; j<_weights_init.getColumnDimension(); j++)
			{
				str += _weights_init.get(i, j);
				str += " ";
			}
			w.add(str);
		}
		
		return w;
	}
	
	/**
	 * The function converts values of the specified parameter to an array of strings.
	 * The function terminates a program with an error, if no conversion is implemented for the specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @return: values of the specified parameter as an array of strings
	 */
	public Vector<String> getLayerParamAsStr(exp_param_E param)
	{
		int i;
		String str;
		Vector<String> param_val;
		
		param_val = new Vector<String>(0,1);
		
		//choose a parameter to be converted to an array of strings
		switch(param)
		{
			case EP_ACTIVATION:
				for(i=0; i<_func.length; i++)
				{
					str = _func[i].getActivation().toString();
					param_val.add(str);
				}
				break;
			case EP_SIZE:
				str = "";
				str+= _nodes.getRowDimension();
				param_val.add(str);
				break;
			case EP_BIAS:
				for(i=0; i<_bias.getRowDimension(); i++)
				{
					str  = "";
					str += _bias.get(i, 0);
					param_val.add(str);
				}
				break;
			case EP_LEAKAGE_RATE:
				for(i=0; i<_leakage_rate.length; i++)
				{
					str = "";
					str+= _leakage_rate[i];
					param_val.add(str);
				}
				break;
			default:
				System.err.println("Layer.getLayerParamAsStr: conversion is not implemented for a specified parameter");
				System.exit(1);
				break;
		}

		return param_val;
	}
	
	/**
	 * The function converts an array of smallest and largest values of neurons to an array of strings.
	 * Each element of this array contains the statistics of the corresponding neuron.
	 * The statistics are represented in form of an interval where the left bound is the smallest value and
	 * the right bound is the largest value.
	 * 
	 * @return: values of the statistics as an array of strings
	 */
	public Vector<String> getLayerValidRangeAsStr()
	{
		int i, j;
		int max_num_valid_intervals;//max number of intervals in a valid range among nodes of one layer
		String str;
		Vector<String> str_array;
		
		str_array = new Vector<String>(0,1);
		
		//find the largest number of valid intervals among nodes of a layer
		//(It is needed to define a number of elements in a vector of valid intervals.)
		max_num_valid_intervals = 1;//there must be at least one valid interval
		for(i=0; i<_valid_range.length; i++)
		{
			if(max_num_valid_intervals < _valid_range[i].length)
			{
				max_num_valid_intervals = _valid_range[i].length;
			}
		}

		//create a vector of valid intervals
		for(i=0; i<_valid_range.length; i++)
		{
			//show all valid intervals as a vector
			str = "";
			for(j=0; j<max_num_valid_intervals; j++)
			{
				if(j < _valid_range[i].length)//store existing intervals first
				{
					str += "(";
					str += _valid_range[i][j].getLowerLimitAsDouble();
					str += ",";
					str += _valid_range[i][j].getUpperLimitAsDouble();
					str += ")";
				}
				else//store dummy elements in the end of a vector
				{
					str += "(-,-)";
				}
				if(j!=max_num_valid_intervals-1)//no " " after the last interval
				{
					str += " ";
				}
			}
			str_array.add(str);
		}

		return str_array;
	}
	
	/**
	 * The function updates a statistics about the smallest and largest values of each output.
	 * The current value is assigned to the smallest value if the current value is smaller than the smallest one.
	 * The current value is assigned to the largest value if the current value is larger than the largest one.
	 */
	public void updateValidRange()
	{
		int i, j, k;
		int num_nodes;
		int idx_interval;//index of valid interval to be updated
		int idx_interval_new;//index in array of new intervals
		double curr_state;
		double width_lower, width_upper;//widths of lower and upper intervals
		double largest_width;//width of the largest interval
		double dist_between_intervals;//distance between intervals
		double lower_lim, upper_lim;//left and right borders of an interval
		interval_C[] new_valid_intervals;//set of valid intervals after range
		
		
		num_nodes = _nodes.getRowDimension();
		for(i=0; i<num_nodes; i++)
		{
			//first allocate valid range if they have not been allocated yet
			if(_valid_range[i]==null)
			{
				allocateValidRange(i);
			}
			
			curr_state = _nodes.get(i, 0);
			//number of valid intervals depends on an activation function of a node
			if(_func[i].getActivation()==activation_E.TANH)
			{
				if(_valid_range[i].length > 1)//if intervals have not been merged yet
				{
					//decide on whether to update a positive or negative intervals
					if(curr_state < 0)
					{
						idx_interval = 0;
					}
					else
					{
						idx_interval = 1;
					}
				}
				else
				{
					idx_interval = 0;
				}
			}
			else
			{
				idx_interval = 0;
			}

			lower_lim = _valid_range[i][idx_interval].getLowerLimitAsDouble();
			upper_lim = _valid_range[i][idx_interval].getUpperLimitAsDouble();
			if(curr_state < lower_lim)
			{
				_valid_range[i][idx_interval].setLeftBorder(curr_state);
			}
			if(curr_state > upper_lim)
			{
				_valid_range[i][idx_interval].setRightBorder(curr_state);
			}
			
			//check whether intervals can be merged if several intervals are available 
			for(j=0; j<_valid_range[i].length-1; j++)
			{
				lower_lim = _valid_range[i][j].getLowerLimitAsDouble();
				upper_lim = _valid_range[i][j].getUpperLimitAsDouble();
				//if data are available in the interval
				width_lower = Double.NaN;
				if(((Double)lower_lim).isInfinite()==false)
				{
					width_lower = upper_lim - lower_lim;
				}
				lower_lim = _valid_range[i][j+1].getLowerLimitAsDouble();
				upper_lim = _valid_range[i][j+1].getUpperLimitAsDouble();
				//if data are available in the interval
				width_upper = Double.NaN;
				if(((Double)lower_lim).isInfinite()==false)
				{
					width_upper = upper_lim - lower_lim;
				}
				//do both intervals exist?
				if(((Double)width_lower).isNaN()==false && ((Double)width_upper).isNaN()==false)
				{
					dist_between_intervals = _valid_range[i][j+1].getLowerLimitAsDouble() -
					                         _valid_range[i][j  ].getUpperLimitAsDouble();
					if(width_lower > width_upper)
					{
						largest_width = width_lower;
					}
					else
					{
						largest_width = width_upper;
					}
					//merge intervals if distance between them is smaller than width of a wider of them
					if(dist_between_intervals < largest_width)
					{
						_valid_range[i][j].setRightBorder(_valid_range[i][j+1].getUpperLimitAsDouble());
						//remove the upper intervals
						new_valid_intervals = new interval_C[_valid_range[i].length-1];
						idx_interval_new = 0;
						for(k=0; k<_valid_range[i].length; k++)
						{
							//skip the checked upper interval
							if(k!=j+1)
							{
								new_valid_intervals[idx_interval_new] = new interval_C(_valid_range[i][k]);
								idx_interval_new++;
							}
						}//for k
						_valid_range[i] = new_valid_intervals;
					}
				}//both intervals exist
			}//for j (go over available intervals)
		}//for i (go over nodes)
	}
	
	/**
	 * The function returns valid ranges of states for each neuron.
	 * 
	 * @return: array of valid ranges of states of neurons
	 */
	public interval_C[][] getMinMaxState()
	{
		int i, j;
		interval_C[][] min_max_state;//output array
		
		//assign values of output neurons
		min_max_state = new interval_C[_num_neurons][];
		for(i=0; i<_num_neurons; i++)
		{
			min_max_state[i] = new interval_C[_valid_range[i].length];
			for(j=0; j<_valid_range[i].length; j++)
			{
				min_max_state[i][j] = new interval_C(_valid_range[i][j]);
			}
		}
		
		return min_max_state;
	}
	
	/**
	 * The function assigns intervals of a provided array as intervals of valid ranges of all nodes of a layer.
	 * 
	 * @param min_max_state: provided array of intervals
	 */
	public void setMinMaxState(interval_C[][] min_max_state)
	{
		int i, j;
		
		//check the input values
		if(min_max_state.length!=_valid_range.length)
		{
			System.err.println("OutputLayer.setMinMaxState: source and destination arrays have different lengths");
			System.exit(1);
		}
		
		for(i=0; i<min_max_state.length; i++)
		{
			_valid_range[i] = new interval_C[min_max_state[i].length];
			for(j=0; j<min_max_state[i].length; j++)
			{
				_valid_range[i][j] = new interval_C(min_max_state[i][j]);
			}
		}
	}
}
