package esn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import experiment.ExpOutput;
import experiment.ExpParam;
import experiment.ExpOutput.esn_output_C;
import experiment.ExpOutput.network_part_save_E;
import experiment.ExpOutput.esn_state_C;
import experiment.ExpParam.exp_param_E;
import experiment.ExpParam.req_val_E;

import types.conversion_C;
import types.interval_C;
import types.seq_parameter_C;

/**
 * This class implements procedures for loading and computing a single SIN module.
 * 
 * @author Danil Koryakin
 *
 */
public class SinModule extends Module
{
	
	//enumeration to define indices of sine parameters
	enum sin_param_E
	{
		SP_FREQ,//frequency of sine
		SP_PHASE//phase of sine
	};

	private int        _idx_history;
	private double     _input;//input value is a time step
	private double[]   _param;//parameters of a SIN modules: [0] - frequency, [1] - phase
	private double[]   _output;//module output
	private double[][] _param_history;//array of consecutive values of parameters
	private double[][] _output_history;//array of consecutive output vectors
	private interval_C[] _range_param;//loaded ranges of parameters: [0] of frequency, [1] for phase
	private interval_C[] _range_output;//loaded ranges of sine output; at the first glance this field makes no sense
	                                   //because it is always [-1,+1] for a sine wave.
	                                   //The main purpose of this field is to realize a common interface for loading and
	                                   //tuning modules with the "(ES)2N". This field is defined as a common attribute
	                                   //of any module.
	private double[] _param_init;//stored initial states of all parameters
	private double[] _param_tmp;//stored temporary states of all parameters
	private double[] _output_init;//stored initial states of all outputs
	private double[] _output_tmp;//stored temporary states of all outputs
	private double   _2_pi;//pre-computed constant 2*PI

	private esn_state_C   _esn_state_save;//object which keeps parameters of the file where current ESN state is saved
	private esn_output_C  _esn_save;//object which keeps parameters of the file where the host ESN is saved	

	final double _error_max = 1E-5;//max error to indicate successful configuration
	
	/**
	 * This is a constructor generating an ESN module using loaded parameters.
	 * 
	 * @param module_idx: index of loaded module (needed under loading the module)
	 * @param sin_load_file: file with parameters of ESN module
	 * @param exp_param: required parameters of ESN module
	 * @param exp_output: object of a class responsible for the file output
	 * @param commonPathCurrRun: common incomplete path to save algorithm specific data
	 * @param seed: seeding value of random numbers generator (new values can be assigned)
	 * @param seed_load: indicator to load seeding value for ESN module
	 */
	public SinModule(int module_idx, String sin_load_files, ExpParam exp_param, ExpOutput exp_output,
			         String commonPathCurrRun, int seed, boolean seed_load)
	{
		super(module_type_E.MT_SIN, module_idx, sin_load_files, exp_param, exp_output,commonPathCurrRun,seed,seed_load);
		
		int i;
		boolean f_save;//indicator of a request to save the run data in files
		
		_2_pi = 2*Math.PI;
			
		//allocate global data
		_param = new double[sin_param_E.values().length];
		_param_init = new double[sin_param_E.values().length];
		_param_tmp = new double[sin_param_E.values().length];
		_range_param = new interval_C[sin_param_E.values().length];
		for(i=0; i<_range_param.length; i++)
		{
			_range_param[i] = new interval_C();
		}
		_input = 0;//input is always initialized with time step 0
		_output = null;
		_output_init = null;
		_output_tmp = null;
		_range_output = null;
		
		_seed = seed;//another seeding value can be assigned under loading an ESN module
		_is_configured = false;
		
		f_save = (Boolean)exp_param.getParamVal(exp_param_E.EP_IO_SAVE_DATA, req_val_E.RV_CUR);
		
		//prepare an object for saving the host ESN
		_esn_save = prepareSavingEsn(f_save, exp_output, commonPathCurrRun);
		        
		//check whether a module must be loaded
		if(sin_load_files.contains("*")==false)
		{
			loadSinModule(module_idx, sin_load_files, seed_load);
		}
		else
		{
			System.err.println("SinModule: missing file with a SIN module");
			System.exit(1);//missing name of a module file
		}
	}
	
	/**
	 * This constructor creates a new SIN module as a copy of all values related to a provided SIN module.
	 * 
	 * @param sin_init: provided SIN module
	 */
	public SinModule(SinModule sin_init)
	{
		super(module_type_E.MT_SIN, 0, null, null, null, null, 0, false);
		
		int i;
		
		_is_configured = false;
		
		//copy frequency and phase
		_param = new double[sin_init._param.length];
		_param_init = new double[sin_init._param_init.length];
		_param_tmp = new double[sin_init._param_tmp.length];
		_range_param = new interval_C[sin_param_E.values().length];
		for(i=0; i<_range_param.length; i++)
		{
			_param[i] = sin_init._param[i];
			_range_param[i] = new interval_C( sin_init._range_param[i] );
		}
		
		//copy ranges of outputs
		_output = new double[sin_init._output.length];
		_output_init = new double[sin_init._output.length];
		_output_tmp = new double[sin_init._output.length];
		_range_output = new interval_C[sin_init._range_output.length];
		for(i=0; i<_range_output.length; i++)
		{
			_range_output[i] = new interval_C( sin_init._range_output[i] );
		}
		
		//copy responsibilities
		_responsibility = new double[sin_init._responsibility.length];
		for(i=0; i<_responsibility.length; i++)
		{
			_responsibility[i] = sin_init._responsibility[i];
		}
		
		//copy the output bias
		_output_bias = new double[sin_init.getOutputBias().length];
		_range_output_bias  = new interval_C[sin_init.getOutputBiasRange().length];
		for(i=0; i<_output_bias.length; i++)
		{
			_output_bias[i] = sin_init.getOutputBias()[i];
			_range_output_bias[i] = new interval_C(sin_init.getOutputBiasRange()[i]);
		}
	}

	/**
	 * The function calls for creating an object for saving an ESN.
	 *  
	 * @param f_save: indicator whether the ESN states must be saved
	 * @param exp_output: object with the classes responsible for the file output
	 * @param commonPathCurrRun: common incomplete path to save algorithm specific data
	 * @return: object for saving the host ESN
	 */
	private esn_output_C prepareSavingEsn(boolean f_save, ExpOutput exp_output, String commonPathCurrRun)
	{
		String esn_file;//path to the file to save the host ESN
		esn_output_C esn_save;//output variable
		
		if(exp_output==null)
		{
			System.err.println("ESN: no classes are provided for the file output");
			System.exit(1);
			esn_save = null;
		}
		else
		{
			//open a file for saving an ESN
			if(f_save==true)
			{			
				esn_file = commonPathCurrRun + "_esn.dat";
			}
			else
			{
				esn_file = null;
			}
			esn_save = exp_output.prepareSavingEsn(esn_file);
		}
		
		return esn_save;
	}
	
	/**
	 * The function extracts values of the provided type from an array of strings. 
	 * Each values is stored in a separate string of the provided array.
	 * 
	 * @param str_vals: array of strings with values sub-reservoir sizes
	 * @param c: type of values to be extracted
	 * @return: array of extracted values
	 */
	private Object[] get1DArrayFromStr(Vector<String> str_vals, Class<?> c)
	{
		int i;
		Object[] vals;
		
		vals = new Object[str_vals.size()];
		
		for(i=0; i<vals.length; i++)
		{
			if(c==Integer.class)
			{
				vals[i] = Integer.valueOf(str_vals.get(i));
			}
			else if(c==Double.class)
			{
				vals[i] = Double.valueOf(str_vals.get(i));
			}
			else if(c==interval_C.class)
			{
				vals[i] = str_vals.get(i);
			}
			else
			{
				vals = null;
				System.err.println("get1DArrayFromStr: unknown class of values");
				System.exit(1);
			}
		}
		
		return vals;
	}
	
	/**
	 * The function assigns values of the provided array as "min_max" statistics of a specified module parameter.
	 * 
	 * @param valid_range: provided array with statistics
	 * @param module_part: name of a module part to be set
	 */
	private void setValidRange(interval_C[] valid_range, network_part_save_E module_part)
	{
		int i;
		interval_C[] range;//name of an array where value must be saved
		
		//choose an array with parameters of a specified part
		switch(module_part)
		{
			case SPS_PARAM_MIN_MAX:
				range = _range_param;
			break;
			case SPS_OUT_MIN_MAX:
				range = _range_output;
			break;
			default:
				System.err.println("SinModule.setValidRange: assignment of requested module part is not implemented");
				System.exit(1);
				range = null;
			break;
		}
		
		for(i=0; i<valid_range.length; i++)
		{
			range[i].copy(valid_range[i]);
		}
	}
	
	/**
	 * This function loads a SIN module which was specified by its index.
	 * Parameters are loaded from a file which is specified by the provided path.
	 * It is necessary that the SIN file contains a module with the provided index.
	 * Otherwise, it issues an error message.
	 * 
	 * @param module_idx: index of a module to be loaded
	 * @param sin_path: path to an SIN file to be loaded
	 * @param seed_load: TRUE is a request to load a seed from a specified file
	 */
	private void loadSinModule(int module_idx, String sin_path, boolean seed_load)
	{
		int        i;
		int        out_size;
		int[]      param_size;
		int[]      seedInternal;
		double[]   param_value;
		double[]   responsibility;
		boolean    is_available;//indicator whether a required module index is available in a loaded file
		interval_C[] min_max_val;//array of smallest and largest values of neurons
		File         esn_file;//file object
		Object[]     obj_vals1D;//one-dimensional array of extracted values as the type "Object"
		Vector<String> data;//vector of strings loaded from the file
		Vector<String> sin_part;//vector of strings with loaded values of a certain part
		
		data = new Vector<String>(0,1);
		esn_file = new File(sin_path);
		try {
			BufferedReader reader = new BufferedReader(new FileReader(esn_file));
			
			while (reader.ready())
			{
				String line = reader.readLine();
				
				//filter out lines with comments
				if(!line.startsWith("#"))
				{
					data.add(line);
				}
			}
			
			reader.close();
			reader = null;
		} catch (FileNotFoundException fnf) {
			data = null;
		} catch (IOException ioe) {
			data = null;
		}
		
		//parse the loaded data
		if(data!=null)
		{
			//check a number of modules in a specified file
			is_available = isModuleIdxAvailable(data, module_idx);
			//the file probably contains a module with index "0" if a specified index is not available
			if(is_available==false)
			{
				module_idx = 0;
				is_available = isModuleIdxAvailable(data, module_idx);
				if(is_available==false)
				{
					System.err.println("SinModule.loadSinModule: file does not contain any sine module");
					System.exit(1);
				}
			}
			
			if(seed_load==true)
			{
				sin_part     = _esn_save.getEsnPart(data, network_part_save_E.MPS_SEED, module_idx);
				obj_vals1D   = get1DArrayFromStr(sin_part, Integer.class);
				seedInternal = conversion_C.ObjToInt1D(obj_vals1D);

				//extracted array always consists of one element
				_seed = seedInternal[0];
			}
			
			//*** SIGNATURE PARAMETERS
			
			sin_part = _esn_save.getEsnPart(data, network_part_save_E.MPS_SEQ_PARAM, module_idx);
			if(sin_part.isEmpty()==false)
			{
				_seq_param = new seq_parameter_C(sin_part);
			}
			else
			{
				System.err.println("SinModule.loadSinModule: signature missing parameters");
				System.exit(1);
			}
			
			//determine number of parameters in specified module
			sin_part   = _esn_save.getEsnPart(data, network_part_save_E.SPS_PARAM_SIZE, module_idx);
			obj_vals1D = get1DArrayFromStr(sin_part, Integer.class);
			param_size   = conversion_C.ObjToInt1D(obj_vals1D);
			
			//check whether the specified module is available in the file
			if(param_size.length==0)
			{
				System.err.println("loadSinModule: specified module is not available in the file");
				System.exit(1);
			}
			
			//*** PARAMETER VALUES ***
			
			sin_part    = _esn_save.getEsnPart(data, network_part_save_E.SPS_PARAM_VALUE, module_idx);
			obj_vals1D  = get1DArrayFromStr(sin_part, Double.class);
			param_value = conversion_C.ObjToDouble1D(obj_vals1D);
			for(i=0; i<param_value.length; i++)
			{
				_param[i] = param_value[i];
			}
			
			//*** PARAMETER RANGES ***
			
			sin_part    = _esn_save.getEsnPart(data, network_part_save_E.SPS_PARAM_MIN_MAX, module_idx);
			obj_vals1D  = get1DArrayFromStr(sin_part, interval_C.class);
			min_max_val = conversion_C.ObjToInterval1D(obj_vals1D);
			setValidRange(min_max_val, network_part_save_E.SPS_PARAM_MIN_MAX);
			
			//*** OUTPUT RANGE ***
			
			sin_part    = _esn_save.getEsnPart(data, network_part_save_E.SPS_OUT_MIN_MAX, module_idx);
			obj_vals1D  = get1DArrayFromStr(sin_part, interval_C.class);
			min_max_val = conversion_C.ObjToInterval1D(obj_vals1D);
			out_size = min_max_val.length;
			_output      = new double[out_size];
			_output_init = new double[out_size];
			_output_tmp  = new double[out_size];
			calculateOutputVector();//compute OUTPUT from loaded values for time step 0
			_range_output = new interval_C[out_size];
			for(i=0; i<_range_output.length; i++)
			{
				_range_output[i] = new interval_C();
			}
			setValidRange(min_max_val, network_part_save_E.SPS_OUT_MIN_MAX);

			initResponsibility(out_size);
			sin_part = _esn_save.getEsnPart(data, network_part_save_E.MPS_RESPONSIBILITY, module_idx);
			obj_vals1D = get1DArrayFromStr(sin_part, Double.class);
			responsibility = conversion_C.ObjToDouble1D(obj_vals1D);
			setResponsibility(responsibility);
			
			sin_part    = _esn_save.getEsnPart(data, network_part_save_E.MPS_RESPONSIBILITY_MIN_MAX, module_idx);
			obj_vals1D  = get1DArrayFromStr(sin_part, interval_C.class);
			min_max_val = conversion_C.ObjToInterval1D(obj_vals1D);
			setResponsibilityRange(min_max_val);
		}
		else
		{
			System.err.println("loadSinModule: specified SIN module file does not exist");
			System.exit(1);
		}
	}
	
	/**
	 * The function returns responsibility of an ESN module for each output neuron in a string array.
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
	 * The function indicates whether the saving was configured for the current ESN. 
	 * @return: "true" if the saving was configured; "false" otherwise
	 */
	public boolean isSavingRequired()
	{
		boolean is_saving;//output variable
		
		if(_esn_state_save!=null)
		{
			is_saving = true;
		}
		else
		{
			is_saving = false;
		}
		return is_saving;
	}
	
	/**
	 * The function indicates whether an ESN has been configured.
	 * 
	 * @return TRUE: if configured; FALSE: otherwise
	 */
	public boolean isConfigured()
	{
		return _is_configured;
	}
	
	/**
	 * The function calls an output object to close saving the ESN states.
	 */
	public void closeSavingEsnState()
	{
		_esn_state_save.closeSavingEsnStates();
	}
	
	/**
	 * The function computes a mean squared error of the module at the current time step.
	 * The output of the module shall already be computed.
	 * It stores a computed ESN output in the provided array. The function does not try to store the computed value
	 * if "null" is provided instead of the array.
	 * 
	 * @param sample_out: target output vector
	 * @param sin_output: computed output of the SIN module (output array)
	 * @return computed error
	 */
	public double computeError(double[] sample_out, double[] sin_output)
	{
		int i;
		double deviation;
		double mse;//output variable

		mse = 0;
		for(i=0; i<_output.length; i++)
		{
			if(sin_output!=null)
			{
				sin_output[i] = _output[i];
			}
			deviation = (sample_out[i] - _output[i]);
			deviation*= deviation;
			mse += deviation;
		}
		mse /= _output.length;
		
		return mse;
	}
	
	/**
	 * The function restores states of an output vector and of internal parameters from a specified element
	 * of a history array.
	 * The element of a history array is requested by its index.
	 * 
	 * @param element_idx: index of the requested element of a history array
	 */
	public void restoreNodesHistory(int element_idx)
	{
		int i;
		
		for(i=0; i<_param.length; i++)
		{
			_param[i] = _param_history[element_idx][i];
		}
		for(i=0; i<_output.length; i++)
		{
			_output[i] = _output_history[element_idx][i];
		}
	}
	
	/**
	 * The function computes a sine function for the current time step.
	 *
	 * @return Matrix: computed output vector
	 */
	public double[] calculateOutputVector()
	{
		int    i;
		double arg;//temporary variable - argument of a sine function
		
		//Since all values are the same, only the 1st value must be computed, the other values are simply assigned
		//from the 1st one.
		arg  = _param[sin_param_E.SP_FREQ.ordinal()]*_input;
		arg += _param[sin_param_E.SP_PHASE.ordinal()];
		
		_output[0] = Math.sin(arg);
		for(i=1; i<_output.length; i++)
		{
			_output[i] = _output[0];
		}
		
		return _output;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function shall indicate a number of nodes in a specified layer.
	 * 
	 * @param layer_type: specified layer
	 * @return: number of nodes in a specified layer
	 */
	public int getNumNodes(layer_type_E layer_type)
	{
		int num_nodes;//output variable
		
		switch(layer_type)
		{
			case LT_INPUT:
				num_nodes = 1;//"1" because a SIN module has a single input node - time step.
				break;
			case LT_RES:
				num_nodes = _param.length;
				break;
			case LT_OUTPUT:
				num_nodes = _output.length;
				break;
			case LT_OFB:
				System.err.println("SinModule.getNumNodes: SIN module has no OFB");
				System.exit(1);
				num_nodes = -1;
				break;
			default:
				System.err.println("SinModule.getNumNodes: unknown layer type");
				System.exit(1);
				num_nodes = -1;
				break;
		}
		return num_nodes;
	}
	
	/**
	 * The function is dummy function of a parent's abstract method.
	 * The function shall assign values from a provided array to nodes of a specified layer.
	 * 
	 * @param node: values to be assigned
	 * @param layer_type: requested layer
	 * @param apply_activation: dummy parameter
	 */
	public void setNodes(double[] node, layer_type_E layer_type, boolean apply_activation)
	{
		int i;
		
		switch(layer_type)
		{
			case LT_INPUT:
				_input = node[0];//there is always a single input node in a SIN module
				break;
			case LT_RES:
				for(sin_param_E param : sin_param_E.values())
				{
					_param[param.ordinal()] = node[param.ordinal()];
				}
				break;
			case LT_OUTPUT:
				for(i=0; i<node.length; i++)
				{
					_output[i] = node[i];
				}
				break;
			case LT_OFB:
				System.err.println("SinModule.setNodes: no OFB layer in a SIN module");
				System.exit(1);
				break;
			default:
				System.err.println("SinModule.setNodes: unknown layer type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is a dummy function of a parent's abstract method.
	 * The function returns an array with values of nodes of a specified layer.
	 * 
	 * @param layer_type: requested layer
	 * @return: array of output values
	 */
	public double[] getNodes(layer_type_E layer_type)
	{
		double[] nodes;//output array
		
		switch(layer_type)
		{
			case LT_INPUT:
				nodes = new double[1];
				nodes[0] = _input;
				break;
			case LT_RES:
				nodes = _param;
				break;
			case LT_OUTPUT:
				nodes = _output;
				break;
			case LT_OFB:
				nodes = null;
				System.err.println("SinModule.getNodes: no OFB layer in a SIN module");
				System.exit(1);
				break;
			default:
				nodes = null;
				System.err.println("SinModule.getNodes: unknown layer type");
				System.exit(1);
				break;
		}
		
		return nodes;
	}
	
	/**
	 * The function is a dummy function of a parent's abstract method.
	 * The function stores states of nodes of a specified layer in a provided storage array.
	 * 
	 * @param layer_type: requested layer
	 * @param storage: provided storage
	 */
	public void getNodes(double[] storage, layer_type_E layer_type)
	{
		int i;
		
		switch(layer_type)
		{
			case LT_INPUT:
				storage[0] = _input;//there is always a single input node in a SIN module
				break;
			case LT_RES:
				for(sin_param_E param : sin_param_E.values())
				{
					storage[param.ordinal()] = _param[param.ordinal()];
				}
				break;
			case LT_OUTPUT:
				for(i=0; i<storage.length; i++)
				{
					storage[i] =  _output[i];
				}
				break;
			case LT_OFB:
				System.err.println("SinModule.getNodes: no OFB layer in a SIN module");
				System.exit(1);
				break;
			default:
				System.err.println("SinModule.getNodes: unknown layer type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates a valid range for each output element of a module.
	 * An output array is 2D for compatibility with an ESN module.
	 * 
	 * @return valid ranges for all output elements of a module
	 */
	public interval_C[][] getOutputValidRange()
	{
		int i;
		interval_C[][] out_intervals;//output array
		
		out_intervals = new interval_C[_range_output.length][1];
		for(i=0; i<out_intervals.length; i++)
		{
			out_intervals[i] = new interval_C[1];
			out_intervals[i][0] = new interval_C(_range_output[i]);
		}
		
		return out_intervals;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates a valid range for each input element of the module.
	 * An output array is 2D for compatibility with the ESN module.
	 * 
	 * @return valid ranges for all input elements of the module
	 */
	public interval_C[][] getInputValidRange()
	{
		int i;
		interval_C[][] out_intervals;//output array
		
		//"[1][1]" because sine have always 1 input and there is always 1 interval for it
		out_intervals = new interval_C[1][1];
		for(i=0; i<out_intervals.length; i++)
		{
			out_intervals[i][0] = new interval_C(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
		}
		
		return out_intervals;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates a valid range for each internal parameter of a module.
	 * An output array is 2D for compatibility with an ESN module.
	 * 
	 * @return valid ranges for all internal parameter of a module
	 */
	public interval_C[][] getInternalValidRange()
	{
		int i;
		interval_C[][] out_intervals;//output array
		
		out_intervals = new interval_C[_range_param.length][1];
		for(i=0; i<out_intervals.length; i++)
		{
			out_intervals[i] = new interval_C[1];
			out_intervals[i][0] = new interval_C(_range_param[i]);
		}
		
		return out_intervals;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function stores a valid range of a specified node in a provided storage.
	 * The node is specified by its index in a specified range.
	 * 
	 * @param node_idx: index of a node
	 * @param layer_type: layer type
	 * @param range: storage where retrieved valid range must be stored
	 */
	public void getNodeValidRange(int node_idx, layer_type_E layer_type, interval_C[] range)
	{
		int i;
		
		for(i=0; i<range.length; i++)
		{
			switch(layer_type)
			{
				case LT_INPUT:
					System.err.println("SinModule.getNodeValidRange: no valid range of input nodes in SIN module");
					System.exit(1);
					break;
				case LT_RES:
					range[i].copy(_range_param[node_idx]);
					break;
				case LT_OUTPUT:
					range[i].copy(_range_output[node_idx]);
					break;
				case LT_OFB:
					System.err.println("SinModule.getNodeValidRange: no OFB layer in a SIN module");
					System.exit(1);
					break;
				default:
					System.err.println("SinModule.getNodeValidRange: unknown layer type");
					System.exit(1);
					break;
			}
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates FALSE because there are no OFB connections in a SIN module.
	 */
	public boolean ExistBackLayer()
	{
		return false;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function shall initiate storing all states of all ESN modules in a specified storage.
	 * The storage is specified by its type.
	 * 
	 * @param type: provided storage type
	 */
	public void storeNodes(storage_type_E type)
	{
		int i;
		
		switch(type)
		{
			case ST_INIT:
				for(i=0; i<_param.length; i++)
				{
					_param_init[i] = _param[i];
				}
				for(i=0; i<_output.length; i++)
				{
					_output_init[i] = _output[i];
				}
				break;
			case ST_TMP:
				for(i=0; i<_param.length; i++)
				{
					_param_tmp[i] = _param[i];
				}
				for(i=0; i<_output.length; i++)
				{
					_output_tmp[i] = _output[i];
				}
				break;
			default:
				System.err.println("SinModule.storeNodes: unknown storage type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The states are retrieved from a specified storage. The storage is specified by its type.
	 * 
	 * @param type: provided storage type
	 */
	public void restoreNodes(storage_type_E type)
	{
		int i;
		
		switch(type)
		{
			case ST_INIT:
				for(i=0; i<_param.length; i++)
				{
					_param[i] = _param_init[i];
				}
				for(i=0; i<_output.length; i++)
				{
					_output[i] = _output_init[i];
				}
				break;
			case ST_TMP:
				for(i=0; i<_param.length; i++)
				{
					_param[i] = _param_tmp[i];
				}
				for(i=0; i<_output.length; i++)
				{
					_output[i] = _output_tmp[i];
				}
				break;
			default:
				System.err.println("SinModule.restoreNodes: unknown storage type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function produces an error if called because the storage concept is not applicable to SIN modules.
	 * 
	 * @param type: storage type
	 */
	public void clearNodes()
	{
		System.err.println("SinModule.clearNodes: storage concept is not applicable to SIN modules");
		System.exit(1);
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function create a history array for output vectors and for internal parameters in a module.
	 * All created history array have the same length which is specified as an input parameter.
	 * 
	 * @param hist_len: required length of history arrays
	 */
	public void createNodesHistory(int hist_len)
	{
		_param_history  = new double[hist_len][];
		_output_history = new double[hist_len][];
		_idx_history = 0;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function prepares history arrays of a SIN module to store next elements.
	 */
	public void startNodesHistory()
	{
		_idx_history = 0;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function stores current states of nodes of all layers at the last element of a history array.
	 */
	public void storeNodesHistory()
	{
		int i;
		
		//store parameter vector as the last element in its history array
		if(_param_history[_idx_history]==null)
		{
			_param_history[_idx_history] = _param.clone();
		}
		else
		{
			for(i=0; i<_param.length; i++)
			{
				_param_history[_idx_history][i] = _param[i];
			}
		}
		
		//store an output vector as the last element in its history array
		if(_output_history[_idx_history]==null)
		{
			_output_history[_idx_history] = _output.clone();
		}
		else
		{
			for(i=0; i<_output.length; i++)
			{
				_output_history[_idx_history][i] = _output[i];
			}
		}
		
		//update a counter of stored elements if necessary
		_idx_history++;
		if(_idx_history==_param_history.length)//all history arrays have the same length
		{
			_idx_history = 0;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function retrieves a specified element from a history array of a specified layer.
	 * The element is specified by its index.
	 * The layer is specified by its type.
	 *
	 * @param idx_element: index of required element
	 * @param layer_type: layer type
	 * @param storage: array where values must be stored
	 */
	public void getNodesHistory(int idx_element, layer_type_E layer_type, double[] storage)
	{
		int i;
		
		switch(layer_type)
		{
			case LT_RES:
				for(i=0; i<_param_history[idx_element].length; i++)
				{
					storage[i] = _param_history[idx_element][i];
				}
				break;
			case LT_OUTPUT:
				for(i=0; i<_output_history[idx_element].length; i++)
				{
					storage[i] = _output_history[idx_element][i];
				}
				break;
			default:
				System.err.println("SinModule.getNodesHistory: unknown layer type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function retrieves the whole history array of a specified layer.
	 * The layer is specified by its type.
	 *
	 * @param layer_type: layer type
	 * @param history: history array
	 */
	public void getNodesHistory(layer_type_E layer_type, double[][] storage)
	{
		int i, j;
		
		switch(layer_type)
		{
			case LT_RES:
				for(i=0; i<_param_history.length; i++)
				{
					for(j=0; j<_param_history[i].length; j++)
					{
						storage[i][j] = _param_history[i][j];
					}
				}
				break;
			case LT_OUTPUT:
				for(i=0; i<_output_history.length; i++)
				{
					for(j=0; j<_output_history[i].length; j++)
					{
						storage[i][j] = _output_history[i][j];
					}
				}
				break;
			case LT_INPUT:
				System.err.println("SinModule.getNodesHistory: no history at input layer in a SIN module");
				System.exit(1);
				break;
			case LT_OFB:
				System.err.println("SinModule.getNodesHistory: no OFB layer in a SIN module");
				System.exit(1);
				break;
			default:
				System.err.println("SinModule.getNodesHistory: unknown layer type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is dummy function of a parent's abstract method.
	 * The function indicates TRUE because spectral radius is a not a parameter of a SIN module.
	 */
	public boolean getNonZeroSR()
	{
		return true;
	}
	
	/**
	 * The function is dummy function of a parent's abstract method.
	 * The function indicates TRUE because a SIN module does not need training.
	 */
	public boolean isTrained()
	{
		return true;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function enables a contribution from a module to the whole modular network.
	 * 
	 * @param restore_states: dummy parameter which is not used for a SIN module
	 */
	public void activate(boolean restore_states)
	{
		restoreResponsibility();
		setConfigured();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function disables a contribution from a module to the whole modular network.
	 */
	public void deactivate()
	{
		resetResponsibility();
		resetConfigured();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function performs an update of a SIN module under transition to the next time step.
	 * The function assigns a value from a provided array as a state of an input node.
	 * This value is the next time step.
	 * 
	 * The functions stores computed output neuron in the history if it is required.
	 * 
	 * @param sample_in: input sample at the next time step
	 * @param do_store: request to store computed output neurons in the history
	 */
	public void advance(double[] sample_in, boolean do_store)
	{
		//put the input values to the input nodes
		setNodes(sample_in, layer_type_E.LT_INPUT, false);
		
		//compute outputs of a SIN module at the current time step
		calculateOutputVector();
		
		//store output value in the history if it is required
		if(do_store==true)
		{
			storeNodesHistory();
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * Since weights are available in a SIN module, a call of this function leads to an error.
	 */
	public Vector<String> getInitWeightsAsStr(layer_type_E layer_type)
	{
		System.err.println("SinModule.getInitWeightsAsStr: weight are not available in a SIN module");
		System.exit(1);
		return null;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * Since weights are available in a SIN module, a call of this function leads to an error.
	 */
	public void storeWeightsInit()
	{
		System.err.println("SinModule.storeWeightsInit: weight are not available in a SIN module");
		System.exit(1);
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function converts values of a parameter which is common for all module layers to an array of strings.
	 * The function terminates a program with an error, if no conversion is implemented for a specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @param layer_type: requested layer
	 * @return: values of the specified parameter as an array of strings
	 */
	public Vector<String> getCommonLayerParamAsStr(exp_param_E param, layer_type_E layer_type)
	{
		String str;
		Vector<String> param_val;
		
		param_val = new Vector<String>(0,1);
		
		//choose a layer
		switch(layer_type)
		{
			case LT_RES:
				//choose a parameter to be converted to an array of strings
				switch(param)
				{
					case EP_SIN_PARAM_SIZE:
						str = "2";
						param_val.add(str);
						break;
					default:
						System.err.println("SinModule.getCommonLayerParamAsStr: no conversion for given parameter");
						System.exit(1);
						break;
				}
				break;
			case LT_INPUT:
			case LT_OFB:
			case LT_OUTPUT:
				System.err.println("SinModule.getCommonLayerParamAsStr: no conversion for specified layer");
				System.exit(1);
				break;
			default:
				System.err.println("SinModule.getCommonLayerParamAsStr: unknown layer");
				System.exit(1);
				break;
		}

		return param_val;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function converts values of a parameter which is specific to a specified module layer to an array of strings.
	 * The function terminates a program with an error, if no conversion is implemented for a specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @param layer_type: requested layer
	 * @return: values of the specified parameter as an array of strings
	 */
	public Vector<String> getSpecificLayerParamAsStr(exp_param_E param, layer_type_E layer_type)
	{
		String str;
		Vector<String> param_val;
		
		param_val = new Vector<String>(0,1);
		
		//choose a layer
		switch(layer_type)
		{
			case LT_RES:
				//choose a parameter to be converted to an array of strings
				switch(param)
				{
					case EP_SIN_PARAM:
						str = "";
						str+= _param[sin_param_E.SP_FREQ.ordinal()];
						param_val.add(str);
						str+= _param[sin_param_E.SP_PHASE.ordinal()];
						param_val.add(str);
						break;
					default:
						System.err.println("SinModule.getSpecificLayerParamAsStr: no conversion for given parameter");
						System.exit(1);
						break;
				}
				break;
			case LT_INPUT:
			case LT_OFB:
			case LT_OUTPUT:
				System.err.println("SinModule.getSpecificLayerParamAsStr: no conversion for specified layer");
				System.exit(1);
				break;
			default:
				System.err.println("SinModule.getSpecificLayerParamAsStr: unknown layer");
				System.exit(1);
				break;
		}

		return param_val;
	}
	
	/**
	 * The function is a dummy function of a parent's abstract method.
	 * The function indicates "null" because there is no connectivity in a SIN module.
	 */
	public Vector<String> getConnectivityMatrixAsStr()
	{
		return null;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates valid ranges of nodes of a specified layer as an array of strings.
	 * 
	 * @param layer_type: requested layer
	 * @return: values of the specified parameter as an array of strings
	 */
	public Vector<String> getValidRangeAsStr(layer_type_E layer_type)
	{
		int    i;
		String str;
		Vector<String> connect_str;
		
		connect_str = new Vector<String>(0,1);
		
		switch(layer_type)
		{
			case LT_RES:
				for(i=0; i<_range_param.length; i++)
				{
					str  = "(";
					str += _range_param[i].getLowerLimitAsDouble();
					str += ",";
					str += _range_param[i].getUpperLimitAsDouble();
					str += ")";
					connect_str.add(str);
				}
				break;
			case LT_OUTPUT:
				for(i=0; i<_range_output.length; i++)
				{
					str  = "(";
					str += _range_output[i].getLowerLimitAsDouble();
					str += ",";
					str += _range_output[i].getUpperLimitAsDouble();
					str += ")";
					connect_str.add(str);
				}
				break;
			case LT_INPUT:
			case LT_OFB:
			default:
				connect_str = null;
				System.err.println("SinModule.getLayerMinMaxAsStr: unknown layer type");
				System.exit(1);
				break;
		}
		
		return connect_str;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function checks whether a submitted parameter value goes out of a valid parameter range.
	 * The function maps a submitted parameter value to an equivalent value in the valid range if it goes out of
	 * the range. Otherwise, the function returns the submitted value without mapping it.
	 * 
	 * @param param_idx: index of module parameter
	 * @param init_value: submitted parameter value
	 * @return: mapped parameter value
	 */
	public double mapParamToEquivalentValue(int param_idx, double init_value)
	{
		int num_intervals;//number of the whole ranges in the submitted value
		double new_value;//output variable
		
		if(param_idx==sin_param_E.SP_FREQ.ordinal())
		{
			new_value = init_value;//no mapping exists
		}
		else if(param_idx==sin_param_E.SP_PHASE.ordinal())
		{
			if(init_value < 0 || init_value > _2_pi)
			{
				num_intervals = (int)(init_value / _2_pi);
				if(init_value < 0)//compute for the lower border
				{
					init_value = init_value - num_intervals*_2_pi;
					new_value = _2_pi + init_value;
				}
				else//compute for the upper border 
				{
					new_value = init_value - num_intervals*_2_pi;
				}
			}
			else
			{
				new_value = init_value;//no mapping is needed
			}
		}
		else//none of possible indices; for example, this can be responsibility
		{
			new_value = init_value;//no mapping exists
		}
		return new_value;
	}
	
	/**
	 * The function indicates a frequency of the SIN module.
	 * 
	 * @return: frequency of the SIN module
	 */
	public double getFrequency()
	{
		return _param[sin_param_E.SP_FREQ.ordinal()];
	}
}
