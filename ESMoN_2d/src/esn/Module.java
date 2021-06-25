package esn;

import java.util.Vector;

import types.interval_C;
import types.seq_parameter_C;

import experiment.ExpOutput;
import experiment.ExpParam;
import experiment.ExpParam.exp_param_E;

public abstract class Module
{
	public enum storage_type_E
	{
		ST_INIT,//storage of initial states of nodes
		ST_TMP//storage of temporary states of nodes
	};
	
	public enum module_type_E
	{
		MT_ESN,//Echo-State Network
		MT_SIN,//sine function from a math library
		MT_FFANN,//Feed-Forward ANN
		MT_INTERPOLATION,//linear Interpolation
		MT_UNKNOWN;//unknown module type
		
		/**
		 * The function extracts a module type as a string for a host enumeration value.
		 * 
		 * @return: module type as a string 
		 */
		public String extractName()
		{
			String str;
			
			str = this.toString();
			str = str.substring(3);//remove the prefix "MT_"
			
			return str;
		}
	};
	
	/**
	 * Types of node layers in a module for all possible types of modules
	 * @author Danil Koryakin
	 */
	public enum layer_type_E
	{
		LT_INPUT,
		LT_RES,
		LT_OFB,//layer without neurons, only weights; therefore has no history
		LT_OUTPUT
	};
	
	protected boolean  _is_configured;//indicator that a module is configured
	protected int      _seed;//seeding value of random number generators
	protected double[] _norm_coef_input;//coefficient for normalization of values from the input vector
	protected double[] _norm_coef_output;//coefficient for normalization of values from the output vector
	protected double[] _responsibility;//responsibilities of an ESN module for output neuron of an mESN
	protected double[] _responsibility_backup;//stored values of responsibilities to restore them later
	protected interval_C[] _range_responsibility;//valid range of responsibilities supported by a module
	protected double[] _output_bias;//module output bias
                                    //1) It is not a bias of output neurons because does not influence possible OFB.
                                    //2) It cannot be replaced by responsibility because latter corresponds to scaling.
    protected interval_C[] _range_output_bias;//valid range of output bias supported by a module
	protected seq_parameter_C _seq_param;//signature parameters of oscillator to recognize it in target dynamics
	protected module_type_E _module_type;//module type
	
	final double _norm_coef_init = 1;//value for initialization of the normalization coefficients 
	final double _responsibility_init = 1;//value for initialization of module responsibility
	final double _output_bias_init = 1;//value for initialization of module's output bias
	final double _output_bias_min_init = 0;//value for initialization of the lower border of the output bias range
	final double _output_bias_max_init = 1;//value for initialization of the upper border of the output bias range
	
	/**
	 * This is a common constructor which either generates an ESN module or a SIN module.
	 * The choice depends on contents of a loaded file: if it does not contain a SIN module then an ESN module is
	 * either loaded or generated.
	 * 
	 * @param module_idx
	 * @param load_files
	 * @param exp_param
	 * @param exp_output
	 * @param commonPathCurrRun
	 * @param seed
	 * @param seed_load
	 */
	public Module(module_type_E module_type, int module_idx, String load_files, ExpParam exp_param,ExpOutput exp_output,
			      String commonPathCurrRun, int seed, boolean seed_load)
	{
		_module_type           = module_type;
		_responsibility        = null;
		_responsibility_backup = null;
		_range_responsibility  = null;
		_norm_coef_input       = null;
		_norm_coef_output      = null;
	}
	
	/**
	 * The function computes a module output for the current time step.
	 *
	 * @return: computed output vector
	 */
	public abstract double[] calculateOutputVector();
	
	/**
	 * The function shall indicate a number of nodes in a specified layer.
	 */
	public abstract int getNumNodes(layer_type_E layer_type);
	
	/**
	 * The function shall assign values from a provided array to nodes of a specified layer.
	 * If applicable and requested, the function shall apply an activation function to the provided values before
	 * assignment.
	 * 
	 * @param node: values to be assigned
	 * @param layer: specified layer
	 * @param apply_activation: request to apply an activation function
	 */
	public abstract void setNodes(double[] node, layer_type_E layer_type, boolean apply_activation);
	
	/**
	 * The function shall return an array with values of nodes of a specified layer.
	 *
	 * @param storage: identifier of the storage array
	 */
	public abstract double[] getNodes(layer_type_E layer_type);
	
	/**
	 * The function shall store states of nodes of a specified layer in a provided storage array.
	 *
	 * @param storage: identifier of the storage array
	 */
	public abstract void getNodes(double[] storage, layer_type_E layer_type);
	
	/**
	 * The function shall indicate a valid range for each output element of a module.
	 * 
	 * @return valid ranges for all output elements of a module
	 */
	public abstract interval_C[][] getOutputValidRange();
	
	/**
	 * The function shall indicate a valid range for each input element of the module.
	 * 
	 * @return valid ranges for all input elements of the module
	 */
	public abstract interval_C[][] getInputValidRange();
	
	/**
	 * The function shall indicate a valid range for each internal node of a module.
	 * 
	 * @return valid ranges for all internal nodes of a module
	 */
	public abstract interval_C[][] getInternalValidRange();
	
	/**
	 * The function stores a valid range of a specified node in a provided storage.
	 * The node is specified by its index in a specified range.
	 * 
	 * @param node_idx: index of a node
	 * @param layer_type: layer type
	 * @param range: storage where retrieved valid range must be stored
	 */
	public abstract void getNodeValidRange(int node_idx, layer_type_E layer_type, interval_C[] range);
	
	/**
	 * The function shall indicate TRUE if there are OFB connections in an ESN module; FALSE - otherwise.
	 * The function shall indicate FALSE for a SIN module.
	 */
	public abstract boolean ExistBackLayer();
	
	/**
	 * The function shall initiate storing all states of all nodes in a specified storage.
	 * The storage is specified by its type.
	 * 
	 * @param type: provided storage type
	 */
	public abstract void storeNodes(storage_type_E type);
	
	/**
	 * The function is an interface to restore preserved states of all nodes of a module.
	 * The states are retrieved from a specified storage. The storage is specified by its type.
	 * 
	 * @param type: provided storage type
	 */
	public abstract void restoreNodes(storage_type_E type);
	
	/**
	 * The function is an interface to clear current states of all neurons of a module.
	 * The function shall initiates a setting of concerned module states to "0". 
	 */
	public abstract void clearNodes();
	
	/**
	 * The function shall create a history array for each node in a module.
	 * All created history arrays shall have the same length which is specified as an input parameter.
	 * 
	 * @param hist_len: required length of history arrays
	 */
	public abstract void createNodesHistory(int hist_len);
	
	/**
	 * The function shall reset a counter of elements in a history array to store next elements.
	 */
	public abstract void startNodesHistory();
	
	/**
	 * The function shall store current states of nodes of all layers at the last element of a history array.
	 */
	public abstract void storeNodesHistory();
	
	/**
	 * The function shall retrieve a specified element from a history array of a specified layer.
	 * The element is specified by its index.
	 * The layer is specified by its type.
	 *
	 * @param idx_element: index of required element
	 * @param layer_type: layer type
	 * @param storage: array where values must be stored
	 */
	public abstract void getNodesHistory(int idx_element, layer_type_E layer_type, double[] storage);
	
	/**
	 * The function shall retrieve the whole history array of a specified layer.
	 * The layer is specified by its type.
	 *
	 * @param layer_type: layer type
	 * @param history: history array
	 */
	public abstract void getNodesHistory(layer_type_E layer_type, double[][] history);
	
	/**
	 * The function shall indicate whether a spectral radius of a module is not zero.
	 * The function shall indicate TRUE if spectral radius is a not a parameter of a module.
	 */
	public abstract boolean getNonZeroSR();
	
	/**
	 * The function shall indicate whether a module is trained.
	 * The function shall indicate TRUE if a module does not need training.
	 */
	public abstract boolean isTrained();
	
	/**
	 * The function shall enable a contribution from a module to the whole modular network.
	 * 
	 * @param restore_states (ESN-specific parameter): request to restore states of all neurons in a module
	 */
	public abstract void activate(boolean restore_states);
	
	/**
	 * The function shall disable a contribution from a module to the whole modular network. 
	 */
	public abstract void deactivate();
	
	/**
	 * The function shall perform an update of a module under transition to the next time step.
	 * The functions shall store a computed output neuron in the history if it is required.
	 * 
	 * @param sample_in: input sample at the next time step
	 * @param do_store: request to store computed output neurons in the history
	 */
	public abstract void advance(double[] sample_in, boolean do_store);
	
	/**
	 * The function shall indicate a matrix of initial weights of a specified layer as an array of strings.
	 * 
	 * @param layer_type: specified layer
	 */
	public abstract Vector<String> getInitWeightsAsStr(layer_type_E layer_type);
	
	/**
	 * This function shall store current weights for their possible activation later.
	 * Since weights are available only in ESN modules, a call of this function for a SIN module leads to an error.
	 */
	public abstract void storeWeightsInit();
	
	/**
	 * The function shall convert values of a parameter which is common for all ESN layers to an array of strings.
	 * The function shall terminate a program with an error, if no conversion is implemented for a specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @param layer_type: requested layer
	 * @return: values of the specified parameter as an array of strings
	 */
	public abstract Vector<String> getCommonLayerParamAsStr(exp_param_E param, layer_type_E layer_type);
	
	/**
	 *
	 * The function shall terminate a program with an error, if no conversion is implemented for a specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @param layer_type: requested layer
	 * @return: values of the specified parameter as an array of strings
	 */
	public abstract Vector<String> getSpecificLayerParamAsStr(exp_param_E param, layer_type_E layer_type);
	
	/**
	 * The function shall present a connectivity matrix of reservoir weights as an array of strings.
	 * This function is necessary only for an ESN module.
	 * 
	 * @return: connectivity matrix as an array of strings
	 */
	public abstract Vector<String> getConnectivityMatrixAsStr();
	
	/**
	 * The function shall indicate valid ranges of nodes of a specified layer as an array of strings.
	 */
	public abstract Vector<String> getValidRangeAsStr(layer_type_E layer_type);
	
	/**
	 * The function shall check whether a submitted parameter value goes out of a valid parameter range.
	 * The function shall map a submitted parameter value to an equivalent value in the valid range if it goes out of
	 * the range. Otherwise, the function shall return the submitted value without mapping it.
	 * 
	 * @param param_idx: index of module parameter
	 * @param init_value: submitted parameter value
	 * @return: mapped parameter value
	 */
	public abstract double mapParamToEquivalentValue(int param_idx, double init_value);
	
	/**
	 * The function assigns initial values of the module's coefficient for normalization of the input values.
	 * 
	 * @param size_out: number of input neurons
	 */
	public void initNormalizationInput(int size_in)
	{
		int i;
		
		_norm_coef_input = new double[size_in];
		for(i=0; i<_norm_coef_input.length; i++)
		{
			_norm_coef_input[i] = _norm_coef_init;
		}
	}
	
	/**
	 * The function assigns initial values of the module's coefficient for normalization of the output values.
	 * 
	 * @param size_out: number of output neurons
	 */
	public void initNormalizationOutput(int size_out)
	{
		int i;
		
		_norm_coef_output = new double[size_out];
		for(i=0; i<_norm_coef_output.length; i++)
		{
			_norm_coef_output[i] = _norm_coef_init;
		}
	}
	
	/**
	 * The function assigns provided values to the normalization coefficients of the input vectors.
	 * 
	 * @param norm_coef_input_new: array of values to be assigned as normalization coefficients for the input vectors
	 */
	public void setNormalizationInput(double[] norm_coef_input_new)
	{
		int i;
		
		for(i=0; i<_norm_coef_input.length; i++)
		{
			_norm_coef_input[i] = norm_coef_input_new[i];
		}
	}
	
	/**
	 * The function assigns provided values to the normalization coefficients of the output vectors.
	 * 
	 * @param norm_coef_output_new: array of values to be assigned as normalization coefficients for the output vectors
	 */
	public void setNormalizationOutput(double[] norm_coef_output_new)
	{
		int i;
		
		for(i=0; i<_norm_coef_output.length; i++)
		{
			_norm_coef_output[i] = norm_coef_output_new[i];
		}
	}
	
	/**
	 * The function stores currently available responsibilities to restore them later.
	 */
	private void backupResponsibility()
	{
		int i;
		
		if(_responsibility_backup==null)
		{
			_responsibility_backup = new double[_responsibility.length];
		}
		for(i=0; i<_responsibility.length; i++)
		{
			_responsibility_backup[i] = _responsibility[i];
		}
	}
	
	/**
	 * The function assigns initial values to module responsibility.
	 * 
	 * @param size_out: number of output neurons
	 */
	public void initResponsibility(int size_out)
	{
		int i;
		
		_responsibility        = new double[size_out];
		_responsibility_backup = new double[size_out];
		_range_responsibility  = new interval_C[size_out];
		for(i=0; i<_responsibility.length; i++)
		{
			_responsibility       [i] = _responsibility_init;
			_responsibility_backup[i] = _responsibility_init;
			_range_responsibility[i]  = new interval_C();
		}
	}
	
	/**
	 * The function indicates current values of module responsibility.
	 * 
	 * @return: module responsibility for each output element
	 */
	public double[] getResponsibility()
	{
		return _responsibility;
	}
	
	/**
	 * The function stores module responsibilities in a provide array.
	 * 
	 * @param storage: array to store module responsibilities
	 */
	public void getResponsibility(double[] storage)
	{
		int i;
		
		if(storage.length!=_responsibility.length)
		{
			System.err.println("Module.getResponsibility: storage and responsibility have different sizes");
			System.exit(1);
		}
		
		for(i=0; i<_responsibility.length; i++)
		{
			storage[i] = _responsibility[i];
		}
	}
	
	/**
	 * The function assigns provided values to module responsibility.
	 * 
	 * @param responsibility_new: array of values to be assigned as a module responsibility
	 */
	public void setResponsibility(double[] responsibility_new)
	{
		int i;
		
		for(i=0; i<_responsibility.length; i++)
		{
			_responsibility[i] = responsibility_new[i];
		}
	}
	
	/**
	 * The function indicates a valid range of responsibilities of the current module.
	 * 
	 * @return: array of intervals with valid ranges of responsibilities
	 */
	public interval_C[] getResponsibilityRange()
	{	
		return _range_responsibility;
	}
	
	/**
	 * The function stores a valid range of responsibility of a specified output neuron in a provided interval.
	 * 
	 * @param sub_idx: index of output neuron 
	 * @param range: provide interval
	 */
	public void getResponsibilityRange(int sub_idx, interval_C range)
	{
		range.copy(_range_responsibility[sub_idx]);
	}
	
	/**
	 * The function assigns provided intervals as module responsibilities for all output nodes.
	 * 
	 * @param range_responsibility: array of intervals to be assigned as valid ranges of responsibilities
	 */
	public void setResponsibilityRange(interval_C[] range_responsibility)
	{
		int i;
		
		for(i=0; i<_responsibility.length; i++)
		{
			_range_responsibility[i].copy(range_responsibility[i]);
			//lower limit of valid range of responsibility can not be negative
			if(_range_responsibility[i].getLowerLimitAsDouble() < 0)
			{
				System.err.println("Module.setResponsibilityRange: invalid lower limit of responsibility range");
				System.exit(1);
			}
		}
	}
	
	/**
	 * The function sets all responsibilities to 0.
	 * The function stores currently available responsibilities to restore them later.
	 */
	public void resetResponsibility()
	{
		int i;
		
		backupResponsibility();
		for(i=0; i<_responsibility.length; i++)
		{
			_responsibility[i] = 0;
		}
	}
	
	/**
	 * The function restores from their backup.
	 */
	public void restoreResponsibility()
	{
		int i;
		
		for(i=0; i<_responsibility.length; i++)
		{
			_responsibility[i] = _responsibility_backup[i];
		}
	}
	
	/**
	 * The function returns a responsibility of a module for each output neuron in a string array.
	 * 
	 * @return: responsibility in a string array
	 */
	public Vector<String> getResponsibilityAsStr()
	{
		int i;
		String str_val;//string with a value
		Vector<String> responsibility;//returned array
		
		responsibility = new Vector<String>(0, 1);
		for(i=0; i<_responsibility.length; i++)
		{
			str_val  = "";
			str_val += _responsibility[i];
			responsibility.add(str_val);
		}
		
		return responsibility;
	}
	
	/**
	 * The function returns a valid range of responsibility for each output neuron in a string array.
	 * 
	 * @return: responsibility in a string array
	 */
	public Vector<String> getResponsibilityRangeAsStr()
	{
		int i;
		String str_val;//string with a value
		Vector<String> responsibility;//returned array
		
		responsibility = new Vector<String>(0, 1);
		for(i=0; i<_responsibility.length; i++)
		{
			str_val  = "(0.0,";
			str_val += _responsibility[i];
			str_val += ")";
			responsibility.add(str_val);
		}
		
		return responsibility;
	}
	
	/**
	 * The function assigns initial values to module's output bias and to its its range.
	 * 
	 * @param size_out: number of output neurons
	 */
	public void initOutputBias(int size_out)
	{
		int i;
		
		_output_bias = new double[size_out];
		_range_output_bias  = new interval_C[size_out];
		for(i=0; i<_output_bias.length; i++)
		{
			_output_bias[i] = _output_bias_init;
			_range_output_bias[i] = new interval_C(_output_bias_min_init, _output_bias_max_init);
		}
	}
	
	/**
	 * The function assigns provided values to the network's output bias.
	 * 
	 * @param bias_new: array of values to be assigned as a network's output bias
	 */
	public void setOutputBias(double[] bias_new)
	{
		int i;
		
		for(i=0; i<_output_bias.length; i++)
		{
			_output_bias[i] = bias_new[i];
		}
	}
	
	/**
	 * The function indicates bias of the network's output bias.
	 * 
	 * @return: bias of each output element
	 */
	public double[] getOutputBias()
	{
		return _output_bias;
	}
	
	/**
	 * The function returns an output bias for each output element in a string array.
	 * 
	 * @return: module's output bias in a string array
	 */
	public Vector<String> getOutputBiasAsStr()
	{
		int i;
		String str_val;//string with a value
		Vector<String> bias;//returned array
		
		bias = new Vector<String>(0, 1);
		for(i=0; i<_output_bias.length; i++)
		{
			str_val  = "";
			str_val += _output_bias[i];
			bias.add(str_val);
		}
		
		return bias;
	}
	
	/**
	 * The function stores module's output bias in a provide array.
	 * 
	 * @param storage: array to store the module's output bias
	 */
	public void getOutputBias(double[] storage)
	{
		int i;
		
		if(storage.length!=_output_bias.length)
		{
			System.err.println("Module.getOutputBias: storage and output bias have different sizes");
			System.exit(1);
		}
		
		for(i=0; i<_output_bias.length; i++)
		{
			storage[i] = _output_bias[i];
		}
	}
	
	/**
	 * The function assigns provided intervals as module's bias for all output nodes.
	 * 
	 * @param range_output_bias: array of intervals to be assigned as valid ranges of output bias
	 */
	public void setOutputBiasRange(interval_C[] range_output_bias)
	{
		int i;
		
		for(i=0; i<_output_bias.length; i++)
		{
			_range_output_bias[i].copy(range_output_bias[i]);
		}
	}
	
	/**
	 * The function indicates a valid range of the network's output bias.
	 * 
	 * @return: array of intervals with valid ranges of output bias
	 */
	public interval_C[] getOutputBiasRange()
	{	
		return _range_output_bias;
	}
	
	/**
	 * The function returns a valid range of bias for each output element of a module in a string array.
	 * 
	 * @return: bias in a string array
	 */
	public Vector<String> getOutputBiasRangeAsStr()
	{
		int i;
		String str_val;//string with a value
		Vector<String> bias;//returned array
		
		bias = new Vector<String>(0, 1);
		for(i=0; i<_responsibility.length; i++)
		{
			str_val = "(0.0, 1.0)";
			bias.add(str_val);
		}
		
		return bias;
	}
	
	/**
	 * The function returns a module type in a string array.
	 * 
	 * @return: module type in a string array
	 */
	public Vector<String> getModuleTypeAsStr()
	{
		String str_val;//string with a value
		Vector<String> str_array;//returned array
		
		str_array = new Vector<String>(0, 1);
		
		str_val = _module_type.extractName();
		str_array.add(str_val);
		
		return str_array;
	}
	
	/**
	 * The function shows an indicator whether a module was configured and active.
	 * 
	 * @return: TRUE if configured and active; FALSE - otherwise 
	 */
	public boolean getConfigured()
	{
		return _is_configured;
	}
	
	/**
	 * The function sets an indicator that a module was configured.
	 */
	public void setConfigured()
	{
		_is_configured = true;
	}
	
	/**
	 * The function clears an indicator that a module was configured.
	 */
	public void resetConfigured()
	{
		_is_configured = false;
	}
	
	/**
	 * The function returns a set of oscillator parameters as a signature of a module.
	 * 
	 * @return: signature of a module
	 */
	public seq_parameter_C getSeqParam()
	{
		return _seq_param;
	}
	
	/**
	 * The functions stores provided parameters as training sequence parameters of the module.
	 */
	public void setSeqParam(Vector<String> seq_parameters)
	{
		_seq_param = new seq_parameter_C(seq_parameters);
	}
	
	/**
	 * The function indicates parameters of individual dynamics that constituted a training sequence of the module.
	 * Parameters of each dynamics are indicated as a separate string in the output array.
	 * 
	 * @return: array strings with parameters of the corresponding dynamics in the training sequence  
	 */
	public Vector<String> getSeqParametersAsString()
	{
		return _seq_param.getSeqParametersAsStr();
	}
	
	/**
	 * The function checks whether a specified module is available in loaded ESN module data. The loaded data are
	 * provided with an input parameter. The module to be checked is specified by its index.
	 * 
	 * @param esn_data: loaded ESN module data
	 * @param module_idx: module index
	 * @return: "true" if the module is available; "false" - otherwise
	 */
	static public boolean isModuleIdxAvailable(Vector<String> esn_data, int module_idx)
	{
		int i;
		String tmp_str;
		String module_suffix;//module suffix which is added to its saved parts
		int idx_substring;
		int idx_name;//index of the 1st title string with a part of the module
		boolean is_available;//output variable
		
		module_suffix = "_MODULE_" + module_idx;
		idx_name = -1;
		for(i=0; i<esn_data.size() && idx_name==-1; i++)
		{
			tmp_str = esn_data.get(i);
			idx_substring = tmp_str.indexOf(module_suffix);
			if(idx_substring!=-1)
			{
				idx_name = i;
			}
		}
		
		if(idx_name!=-1)
		{
			is_available = true;
		}
		else
		{
			is_available = false;
		}
		
		return is_available;
	}
	
	/**
	 * The function shall indicate a seeding value of random number generators of an module.
	 */
	public int getSeed()
	{
		return _seed;
	}
	
	/**
	 * The function returns a seeding value of random generators in a string array.
	 * 
	 * @return: seeding value in a string array
	 */
	public Vector<String> getSeedAsStr()
	{
		String str_val;//string with a value
		Vector<String> seed;//returned array
		
		seed = new Vector<String>(0, 1);

		str_val  = "";
		str_val += _seed;
		seed.add(str_val);
		
		return seed;
	}
}
