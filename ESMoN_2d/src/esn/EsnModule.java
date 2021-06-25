package esn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import types.conversion_C;
import types.interval_C;
import types.multi_val_C;
import types.seq_parameter_C;
import Jama.LUDecomposition;
import Jama.Matrix;
import Jama.QRDecomposition;
import MathDiff.MathNoise;
import MathDiff.MathNoise.noise_E;
import esn.Activation.activation_E;
import esn.Layer.leakage_assign_E;
import experiment.ExpOutput;
import experiment.ExpParam;
import experiment.ExpOutput.esn_output_C;
import experiment.ExpOutput.network_part_save_E;
import experiment.ExpOutput.esn_state_C;
import experiment.ExpParam.exp_param_E;
import experiment.ExpParam.req_val_E;
import experiment.ExpSeq.seq_C;

/**
 * This class implements procedures for creating, loading, training and testing a single ESN module.
 * 
 * @author Danil Koryakin
 *
 */
public class EsnModule extends Module
{
	/**
	 * methods for configuration of the sub-reservoirs
	 * @author Danil Koryakin
	 */
	public enum config_method_E
	{
		CM_DirectTeacherForce,//to call the function "configDirectTeacherForce()"
		CM_DiffEvolution,     //to call the function "configDiffEvolution()"
		CM_None,              //no configuration is required
		CM_Unknown;           //unknown configuration method is required
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CM_None", if the provided string was not found in any of the enumeration values
		 */
		public static config_method_E fromString(String str)
		{
			config_method_E method;//output variable
			
			method = CM_Unknown;
			for(config_method_E value : config_method_E.values())
			{
				if(value.name().contains(str)==true)
				{
					method = value;
				}
			}
			if(method==CM_Unknown)
			{
				System.err.println("ConfigMethod.fromString: invalid string to convert into configuration method");
				System.exit(1);
			}
			
			return method;
		}
	};
	
	/**
	 * This enumeration defines possible types of configuration of ESN modules.
	 * @author Danil Koryakin
	 */
	public enum config_type_E
	{
		CT_Decomposition,  //active ESN modules are not known a priori
		CT_Synchronization,//active ESN modules are known a priori
		CT_None;           //enumeration value which corresponds to none of available types of configuration 
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CT_None", if the provided string was not found in any of the enumeration values
		 */
		public static config_type_E fromString(String str)
		{
			config_type_E type;//output variable
			
			type = CT_None;
			for(config_type_E value : config_type_E.values())
			{
				if(value.name().contains(str)==true)
				{
					type = value;
				}
			}
			if(type==CT_None)
			{
				System.err.println("ConfigType.fromString: invalid string to convert into configuration type");
				System.exit(1);
			}
			
			return type;
		}
	};
	
	/**
	 * modes of teacher-forcing under evolutionary adaptation
	 * @author Danil Koryakin
	 */
	public enum config_ea_mode_E
	{
		CFG_EA_MODE_PERMANENT, //evolutionary algorithm is started at every time step
		CFG_EA_MODE_INTERLEAVE,//evolutionary algorithm is started every N-th time step; free-run between these steps
		CFG_EA_MODE_UNKNOWN;//not known how to apply the evolutionary algorithm
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CFG_EA_MODE_UNKNOWN", if the provided string was not found in any of the enumeration values
		 */
		public static config_ea_mode_E fromString(String str)
		{
			config_ea_mode_E mode;//output variable
			
			mode = CFG_EA_MODE_UNKNOWN;
			for(config_ea_mode_E value : config_ea_mode_E.values())
			{
				if(value.name().contains(str)==true)
				{
					mode = value;
				}
			}
			if(mode==CFG_EA_MODE_UNKNOWN)
			{
				System.err.println("config_ea_mode_E.fromString: invalid string");
				System.exit(1);
			}
			
			return mode;
		}
	};

	private InputLayer    _inputLayer = null;//object with input neurons and input connections
	private InternalLayer _internalLayer = null;//object with reservoir neurons and their connections
	private BackLayer     _backLayer = null;//objects with OFB connections
	private OutputLayer   _outputLayer = null;//object with output neurons and their connections
	private esn_state_C   _esn_state_save;//object which keeps parameters of the file where current ESN state is saved
	private esn_output_C  _esn_save;//object which keeps parameters of the file where the host ESN is saved
	private InterpolationModule[] _interpolation_module;//FF-ANN modules for the output weights for each ESN output

	final int _window_sub_target_approx = 10;//length of the time window for computing the error under configuration
	                                         //through estimation of the target values for the separate sub-reservoirs
	final int _max_num_seq_combinations = 10000000;//maximum number of sequence combinations when the brute force
	                                               //approach is applied;
	                                       //!!!This value was found empirically.
	                                       //!!!For larger numbers of combinations, the brute force search is too long.
	final double _error_max = 1E-5;//max error to indicate successful configuration
	
	/**
	 * This is a constructor generating an ESN module using loaded parameters.
	 * 
	 * @param module_idx: index of loaded module (needed under loading the module)
	 * @param esn_load_file: file with parameters of ESN module
	 * @param exp_param: required parameters of ESN module
	 * @param exp_output: object of a class responsible for the file output
	 * @param commonPathCurrRun: common incomplete path to save algorithm specific data
	 * @param seed: seeding value of random numbers generator (new values can be assigned)
	 * @param seed_load: indicator to load seeding value for ESN module
	 */
	public EsnModule(int module_idx, String esn_load_files, ExpParam exp_param, ExpOutput exp_output, String commonPathCurrRun, int seed, boolean seed_load)
	{
		super(module_type_E.MT_ESN, module_idx, esn_load_files, exp_param, exp_output,commonPathCurrRun,seed,seed_load);
		
		boolean f_save;//indicator of a request to save the run data in files
		
		_seed = seed;//another seeding value can be assigned under loading an ESN module
		_is_configured = false;
		
		f_save = (Boolean)exp_param.getParamVal(exp_param_E.EP_IO_SAVE_DATA, req_val_E.RV_CUR);
		
		//prepare an object for saving the host ESN
		_esn_save = prepareSavingEsn(f_save, exp_output, commonPathCurrRun);
		        
		//check whether a module must be loaded
		if(esn_load_files.contains("*")==false)
		{
			loadEsnModule(module_idx, esn_load_files, seed_load);
		}
		else
		{
			//create an echo-state network
			initNetwork(module_idx, exp_param);
		}
		
		_interpolation_module = new InterpolationModule[_outputLayer.getNumNodes()];
	}
	
	/**
	 * This constructor creates a new ESN as a copy of all values related to a provided ESN module.
	 * 
	 * @param esn_init: provided ESN module
	 */
	public EsnModule(EsnModule esn_init)
	{
		super(module_type_E.MT_ESN, 0, null, null, null, null, 0, false);
		
		int          i;
		int          size_sub;//number of neurons in the specified sub-reservoir
		int          seed_new;//array of seeding values of a new ESN
		double       sr;//array of spectral radii of all sub-reservoirs of initial ESN
		double       connect_new;//connectivity of a new ESN module
		double       sr_new;//SR of new ESN module (needed to call constructor of intern layer)
		double[]     output_bias_orig;//output bias of the original module
		Matrix       weights_orig;//original weights of original ESN module
		Matrix       weights_sub;//array of weights of current sub-reservoir
		Matrix       nodes_init;//states of neurons of a layer
		Activation[] activation;//array with activations of all neurons of current layer
		Matrix bias;//array with biases of neurons
		interval_C[] output_bias_range_orig;//output bias range of the original module
		
		_is_configured = false;
		sr             = esn_init.getInternalLayer().getSpectralRadius();
		size_sub       = esn_init.getInternalLayer().getSize();
		
		//COPY the output bias
		output_bias_orig = esn_init.getOutputBias();
		output_bias_range_orig = esn_init.getOutputBiasRange();
		_output_bias = new double[output_bias_orig.length];
		_range_output_bias  = new interval_C[output_bias_range_orig.length];
		for(i=0; i<_output_bias.length; i++)
		{
			_output_bias[i] = output_bias_orig[i];
			_range_output_bias[i] = new interval_C(output_bias_range_orig[i]);
		}
		
		//COPY the RESRVOIR

		connect_new    = esn_init.getInternalLayer().getConnectivity();
		sr_new         = sr;
		seed_new       = esn_init.getInternalLayer().getSeed();
		_internalLayer = new InternalLayer(size_sub, connect_new, sr_new, seed_new);
		//assign initial and active reservoir weights
		weights_orig   = esn_init.getInternalLayer().getWeights();
		weights_sub    = weights_orig.copy();
		_internalLayer.setWeights(weights_sub);
		weights_orig   = esn_init.getInternalLayer().getWeightsInit();
		weights_sub    = weights_orig.copy();
		_internalLayer.setWeightsInit(weights_sub);
		
		//extract and assign the activations and the biases of the reservoir neurons
		activation = esn_init.getInternalLayer().getActivation();
		bias       = esn_init.getInternalLayer().getBias();
		nodes_init = esn_init.getInternalLayer().getNodes();
		_internalLayer.setActivation(activation);
		_internalLayer.setBias(bias);
		_internalLayer.setLeakageRate(esn_init.getInternalLayer().getLeakageRate());
		_internalLayer.setNoiseType  (esn_init.getInternalLayer().getNoiseType());
		_internalLayer.setNoiseBounds(esn_init.getInternalLayer().getNoiseBounds());
		_internalLayer.setNodes(nodes_init, false);
		
		//COPY the OUTPUT LAYER
		
		seed_new         = esn_init.getOutputLayer().getSeed();
		_outputLayer = new OutputLayer(esn_init.getOutputLayer().getRows(), size_sub, seed_new);
		//assign initial and active weights
		weights_orig   = esn_init.getOutputLayer().getWeights();
		weights_sub    = weights_orig.copy();
		_outputLayer.setWeights(weights_sub);
		weights_orig   = esn_init.getOutputLayer().getWeightsInit();
		weights_sub    = weights_orig.copy();
		_outputLayer.setWeightsInit(weights_sub);
		
		activation = esn_init.getOutputLayer().getActivation();
		bias       = esn_init.getOutputLayer().getBias();
		nodes_init = esn_init.getOutputLayer().getNodes();
		_outputLayer.setActivation(activation);
		_outputLayer.setBias(bias);
		_outputLayer.setNodes(nodes_init, false);
		
		//COPY the INPUT LAYER
		
		if(esn_init.getInputLayer()!=null)
		{
			seed_new    = esn_init.getInputLayer().getSeed();
			_inputLayer = new InputLayer(size_sub, esn_init.getInputLayer().getCols(), seed_new);
			
			//assign initial and active input weights
			weights_orig   = esn_init.getInputLayer().getWeights();
			weights_sub    = weights_orig.copy();
			_inputLayer.setWeights(weights_sub);
			weights_orig   = esn_init.getInputLayer().getWeightsInit();
			weights_sub    = weights_orig.copy();
			_inputLayer.setWeightsInit(weights_sub);

			//copy activations and biases
			activation = esn_init.getInputLayer().getActivation();
			bias       = esn_init.getInputLayer().getBias();
			_inputLayer.setActivation(activation);
			_inputLayer.setBias(bias);
		}
		else
		{
			_inputLayer = null;
		}
		
		//copy the OFB
		if(esn_init.getBackLayer()!=null)
		{
			_backLayer = new BackLayer(size_sub, esn_init.getBackLayer().getCols(), seed_new);
			
			//assign initial and active weights
			weights_orig   = esn_init.getBackLayer().getWeights();
			weights_sub    = weights_orig.copy();
			_backLayer.setWeights(weights_sub);
			weights_orig   = esn_init.getBackLayer().getWeightsInit();
			weights_sub    = weights_orig.copy();
			_backLayer.setWeightsInit(weights_sub);
		}
		else
		{
			_backLayer = null;
		}
	}
	
	/**
	 * This methods performs the application of noise according to the settings of the
	 * esn.
	 * @param states, the Matrix object containing the values of the internal nodes. 
	 * @return states, the Matrix that was assigned to this method, now containing the
	 * values of the internal states with noise applied.
	 */
	private Matrix applyNoise(Matrix states)
	{
		Matrix output;
		noise_E noise_type;
		interval_C noise_bound;
		double lower_lim, upper_lim;//left and right borders of an interval
		
		noise_type  = _internalLayer.getNoiseType();
		noise_bound = _internalLayer.getNoiseBounds();
		lower_lim = noise_bound.getLowerLimitAsDouble();
		upper_lim = noise_bound.getUpperLimitAsDouble();
		switch(noise_type)
		{
			case NOISE_UNIFORM:
			case NOISE_UNIFORM_SYNC:
				output = MathNoise.applyUniformNoise(states, lower_lim, upper_lim, noise_type);
				break;
			case NOISE_GAUSSIAN:
			case NOISE_GAUSSIAN_SYNC:
				output = MathNoise.applyGaussianNoise(states, lower_lim, upper_lim, noise_type);
				break;
			case NOISE_NONE:
				output = states;
				break;
			default:
				System.err.println("applyNoise: unknown noise type");
				System.exit(1);
				output = null;
				break;
		}
		return output;
	}
	
	/**
	 * This method update's the internal states according to the formula
	 * x(n+1) = f_trans(W_in * u(n+1) + W_int * x(n) + W_back * y(n) + res_bias),
	 * where:
	 * 	x(n+1) is the internal vector in the next trial,
	 * 	f_trans is the transfer-function applied to the state-values (e.g. tanh),
	 * 	W_in   denotes the input-weight-matrix (N x K),
	 * 	u(n+1) is the next input-vector (K x 1),
	 * 	W_int  is the internal-weight-matrix (N x N),
	 * 	x(n)   is the current internal-state-vector (N x 1),
	 * 	W_back is the back-projection-matrix (N x L),
	 *  y(n)   is the current output-vector (L x 1), and
	 *  res_bias is a constant bias of the reservoir neurons.
	 *  
	 * @return Matrix, the new internal-states
	 */
	private Matrix calculateInternalState()
	{
		int i;
		double[][] internalValues;//states of reservoir neurons
		double[]   leakage_rate;
		Matrix  input;
		Matrix  internal;
		Matrix  internal_prev;//outputs of the reservoir neurons at the previous time step; used for leaky-integrators
		Matrix  back;
		noise_E noise_type;

		leakage_rate = _internalLayer.getLeakageRate();
		noise_type   = _internalLayer.getNoiseType();
		
		/*
		 * if there is a non-null input-layer, there is an input-layer in use respectively,
		 * it is multiplied with the input-vector
		 */
		input = (_inputLayer!=null) ?                  //check existence of input layer
                 _inputLayer.getWeights().copy().times(//get W_in and compute R_inp = W_in*u(n+1)*F_scale_in
                 _inputLayer.getNodes().copy()) :      //get u(n+1)
                 null;
		
		//save outputs of reservoir neurons for computation of the leaky-integrator
		internal_prev = _internalLayer.getNodes();
		
		//multiplication of the internal-weight's with the internal-vector
		internal = _internalLayer.getWeights().copy().times(//get W and compute R_int = W*x(n)
				   _internalLayer.getNodes().copy());       //get x(n)
		                                                    //"internal[Nx1] = W_int[NxN] x X_n[Nx1]"
                                                            //"input": number of rows equals to the number of internal neurons
                                                            //       : number of columns is 1
		//the multiplication of the back-projection-weight's with the output-vector
		back = (_backLayer!=null) ?                  //check, whether output feedback are used
				_backLayer.getWeights().copy().times(//get W_back and compute R_back = W_back*y(n)*F_scale_back
				_outputLayer.getNodes().copy()):     //get y(n)
				null;                                //"input": number of rows equals to the number of internal neurons
				                                     //       : number of columns is 1
				
		//the following steps are the necessary additions
		internal = (input != null) ? internal.plus(input) : internal;//check presence of input and compute R_ii = R_int+R_inp  
		internal = (back != null)  ? internal.plus(back)  : internal;//check presence of output feedback and compute R_iio = R_ii+R_back
		internal = internal.plus(_internalLayer._bias);
		
		//if we use noise, a random value is added to the new internal-states
		if(noise_type != MathNoise.noise_E.NOISE_NONE)//is noise required?
		{
			internal = applyNoise(internal);//add noise to each element of R_iio
		}
		
		internalValues = internal.getArray();
		
		//application of a transfer-function takes place here
		for(i=0; i < internalValues.length; i++)
		{
			//states are always a one-column matrix
			internalValues[i][0] = (1.0D - leakage_rate[i]) * internal_prev.get(i, 0) +
			                       _internalLayer._func[i].calculateValue(internalValues[i][0]);
		}
		
		return new Matrix(internalValues);
	}
	
	/**
	 * With this method the network is initialized according to the required ESN parameters.
	 * This includes the setup of the layers with randomized weight-
	 * matrices, the definition of internal transfer-functions etc.
	 * 
	 * @param module_idx: index of module to be created
	 * @param exp_param: object with the parameters of the experiment 
	 */
	private void initNetwork(int module_idx, ExpParam exp_param)
	{
		int outsize;//number of output connections
		int size_in;
		int size_out;
		int res_size;//total number of neurons in the whole dynamic reservoir
		double     activ_ratio;
		double     activ_logistic_param;
		double     leakage_rate;
		double     connect;
		double     sr;
		boolean    is_input;
		boolean    is_out_recurrence;
		boolean    is_w_ofb;
		interval_C in_w;
		interval_C res_w;
		interval_C ofb_w;
		interval_C res_bias;
		activation_E activ_res;//required activation of reservoir neurons
		activation_E activ_out;
		leakage_assign_E leakage_assign;
		ReservoirInitialization     topology;
		noise_E noise_types;
		interval_C noise_bounds;
		multi_val_C tmp;//temporary variable for conversion of a multi_val_C object into an interval_C object
		
		size_in        = (Integer)exp_param.getParamVal(exp_param_E.EP_INPUT_SIZE, req_val_E.RV_CUR);
		size_out       = (Integer)exp_param.getParamVal(exp_param_E.EP_OUTPUT_SIZE, req_val_E.RV_CUR);
		activ_ratio    = (Double)exp_param.getParamVal(exp_param_E.EP_RES_ACTIVATION_RATIO, req_val_E.RV_CUR);
		leakage_assign = (leakage_assign_E)exp_param.getParamVal(exp_param_E.EP_LEAKAGE_ASSIGN, req_val_E.RV_CUR);
		activ_logistic_param = (Double)exp_param.getParamVal(exp_param_E.EP_ACTIVATION_LOGISTIC_PARAM, req_val_E.RV_CUR);
		is_input          = (Boolean)exp_param.getParamVal(exp_param_E.EP_INPUT_USE, req_val_E.RV_CUR);
		is_out_recurrence = (Boolean)exp_param.getParamVal(exp_param_E.EP_OUTPUT_RECURRENCE, req_val_E.RV_CUR);
		is_w_ofb          = (Boolean)exp_param.getParamVal(exp_param_E.EP_OFB_USE, req_val_E.RV_CUR);
		tmp      = (multi_val_C)exp_param.getParamVal(exp_param_E.EP_INPUT_W, req_val_E.RV_CUR);
		in_w     = tmp.getInterval();
		tmp      = (multi_val_C)exp_param.getParamVal(exp_param_E.EP_RES_W, req_val_E.RV_CUR);
		res_w    = tmp.getInterval();
		tmp      = (multi_val_C)exp_param.getParamVal(exp_param_E.EP_BIAS, req_val_E.RV_CUR);
		res_bias = tmp.getInterval();
		tmp      = (multi_val_C)exp_param.getParamVal(exp_param_E.EP_OFB_W, req_val_E.RV_CUR);
		ofb_w    = tmp.getInterval();
		activ_res = (activation_E)exp_param.getParamVal(exp_param_E.EP_ACTIVATION, req_val_E.RV_CUR);
		activ_out = (activation_E)exp_param.getParamVal(exp_param_E.EP_OUTPUT_ACTIVATION, req_val_E.RV_CUR);
		topology  = (ReservoirInitialization)exp_param.getParamVal(exp_param_E.EP_RES_TOPOLOGY, req_val_E.RV_CUR);
		noise_types  = (noise_E)exp_param.getParamVal(exp_param_E.EP_PERFORM_NOISE_TYPE, req_val_E.RV_CUR);
		tmp          = (multi_val_C)exp_param.getParamVal(exp_param_E.EP_PERFORM_NOISE_BOUNDS, req_val_E.RV_CUR);
		noise_bounds = tmp.getInterval();
		
		res_size     = exp_param.getSize(module_idx);
		connect      = exp_param.getConnectivity(module_idx);
		leakage_rate = exp_param.getLeakageRate(module_idx);
		sr           = exp_param.getSpectralRadius(module_idx);
		
		//add a number of output neurons to number of output connections, if there are recurrent connections inside output layer
		//(these are connections from output to output)
		outsize  = res_size;
		outsize	+= (is_out_recurrence) ? size_out : 0;
		
		if(is_out_recurrence==true)
		{
			System.err.println("initNetwork: concept subreservoirs is not working when output neurons have self-recurrence");
		}
		
		if(is_input)
		{
			_inputLayer = new InputLayer(res_size, size_in, _seed);
			outsize	+= size_in;
		}
		else
		{
			_inputLayer = null;
		}
		
		//create a reservoir for the current module 
		_internalLayer = new InternalLayer(res_size,
				                           connect,
				                           sr,
				                           _seed);
		
		if(is_w_ofb)
		{
			_backLayer = new BackLayer(res_size, size_out, _seed);
		}
		else
		{
			_backLayer = null;
		}
		
		_outputLayer = new OutputLayer(size_out, outsize, _seed);
		_outputLayer.setSelfRecurrent(is_out_recurrence);
		
		if(_inputLayer != null)
		{
			_inputLayer.setActivation(activation_E.ID, -1);//activation function of input neurons is always linear
			_inputLayer.setBias(0, -1);//currently there is no need to assign non-zero biases to the input neurons
			_inputLayer.setWeightBounds(in_w.getArray());
			_inputLayer.initializeWeights();
		}

		_internalLayer.setWeightBounds            (res_w.getArray());
		_internalLayer.setInitializationPolicy(topology);
		_internalLayer.setLeakageRate       (leakage_assign, leakage_rate);
		_internalLayer.generateActivations  (activ_res, activation_E.ID, activ_ratio, activ_logistic_param);
		_internalLayer.initializeBias       (res_bias);
		_internalLayer.setNoiseType         (noise_types);
		_internalLayer.setNoiseBounds       (noise_bounds);
		_internalLayer.initializeWeights();
		_internalLayer.computeMaxEigenvalue();
		
		if(_backLayer != null)
		{
			_backLayer.setWeightBounds(ofb_w.getArray());
			_backLayer.initializeWeights();
		}

		initResponsibility(size_out);
		initOutputBias(size_out);
		
		_outputLayer.setActivation(activ_out, -1);
		_outputLayer.setBias(0, -1);//currently there is no need to assign non-zero biases to the input neurons
		_outputLayer.initializeWeights();
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
	 * The function determines a number of columns in a matrix whose rows are given by the provided array of strings.
	 * In the provided array, values of the same row shall be separated with spaces.
	 * 
	 * @param str_vals: array of strings with values of a matrix
	 * @return: number of columns
	 */
	private int getNumCols(Vector<String> str_vals)
	{
		int i, j;
		int num_cols;//number of columns in the 1st row (returned value)
		int num_cols_i;//number of columns in the 2nd and further rows
		String[] str_dummy;//dummy array with values of the same row
		
		i = 0;
		str_dummy = str_vals.get(i).split(" ");
		//number of columns is a number of not-empty strings in the same row
		num_cols = 0;
		for(j=0; j<str_dummy.length; j++)
		{
			if(str_dummy[j].isEmpty()==false)
			{
				num_cols++;
			}
		}
		//check the number of columns in all rows;
		//it must be equal
		for(i=1; i<str_vals.size(); i++)
		{
			str_dummy = str_vals.get(i).split(" ");
			num_cols_i = 0;
			for(j=0; j<str_dummy.length; j++)
			{
				if(str_dummy[j].isEmpty()==false)
				{
					num_cols_i++;
				}
			}
			
			if(num_cols!=num_cols_i)
			{
				System.err.println("getNumCols: provided strings conain different numbers of values");
				System.exit(1);
			}
		}
		
		return num_cols;
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
	 * The function extracts values of the provided type from an array of strings. Each string can contain one or
	 * several values. All value of the same string are stored in one row of the output array.
	 * 
	 * @param str_vals: array of strings with values to be extracted
	 * @return: array of extracted values
	 */
	private String[][] get2DArrayFromStr(Vector<String> str_vals)
	{
		int i,j;
		int idx_col;//index of a column of the currently assigned element
		int num_cols;//number of columns
		String[][] vals;
		String[] str_row;//string with values of the same row
		
		vals = new String[str_vals.size()][];
		num_cols = getNumCols(str_vals);

		for(i=0; i<vals.length; i++)
		{
			str_row = str_vals.get(i).split(" ");

			//extract values of the same row
			vals[i] = new String[num_cols];
			idx_col = 0;
			for(j=0; j<str_row.length; j++)
			{
				if(str_row[j].isEmpty()==false)
				{
					vals[i][idx_col] = str_row[j];
					idx_col++;
				}
			}
		}
		
		return vals;
	}
	
	/**
	 * This function loads an ESN module which was specified by its index.
	 * Parameters are loaded from a file which is specified by the provided path.
	 * It is necessary that the ESN file contains a module with the provided index.
	 * Otherwise, it issues an error message.
	 * 
	 * @param module_idx: index of a module to be loaded
	 * @param esn_path: path to an ESN file to be loaded
	 * @param seed_load: TRUE is a request to load a seed from a specified file
	 */
	private void loadEsnModule(int module_idx, String esn_path, boolean seed_load)
	{
		int[] seedInternal;
		int[] sub_size;
		double[]   leakage_rate;
		double[]   responsibility, output_bias;
		double[]   connect;
		double[]   sr;
		boolean    is_available;//indicator whether a required module index is available in a loaded file
		interval_C[]   min_max_val;//smallest and largest value for each node
		interval_C[][] valid_range;//array of valid intervals for all nodes of a layer
		activation_E[] activation;//array of activation functions
		File esn_file;//file object
		Matrix weights;//matrix of all extracted weights
		Matrix bias;//matrix of extracted biases
		Object[]   obj_vals1D;//one-dimensional array of extracted values as the type "Object"
		String[][] str_vals2D;//two-dimensional array of extracted values as the type "String"
		Vector<String> data;//vector of strings loaded from the file
		Vector<String> esn_part;//vector of strings with loaded values of a certain part
		ReservoirInitialization topology;//reservoir topology
		
		data = new Vector<String>(0,1);
		esn_file = new File(esn_path);
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
					System.err.println("EsnModule.loadEsnModule: file does not contain any ESN module");
					System.exit(1);
				}
			}
			
			if(seed_load==true)
			{
				esn_part     = _esn_save.getEsnPart(data, network_part_save_E.MPS_SEED, module_idx);
				obj_vals1D   = get1DArrayFromStr(esn_part, Integer.class);
				seedInternal = conversion_C.ObjToInt1D(obj_vals1D);

				//extracted array always consists of one element
				_seed = seedInternal[0];
			}
			
			//determine number of neurons in specified module
			esn_part   = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_SIZE, module_idx);
			obj_vals1D = get1DArrayFromStr(esn_part, Integer.class);
			sub_size   = conversion_C.ObjToInt1D(obj_vals1D);
			
			//check whether the specified module is available in the file
			if(sub_size.length==0)
			{
				System.err.println("loadEsnModule: specified module is not available in the file");
				System.exit(1);
			}
			
			/*** parameters of training sequence ***/
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.MPS_SEQ_PARAM, module_idx);
			if(esn_part.isEmpty()==false)
			{
				_seq_param = new seq_parameter_C(esn_part);
			}
			else
			{
				System.err.println("EsnModule.loadEsnModule: missing parameters of training sequence");
				System.exit(1);
			}

			/*** INPUT LAYER ***/
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.EPS_IN_W, module_idx);
			if(esn_part.isEmpty()==false)
			{
				str_vals2D  = get2DArrayFromStr(esn_part);
				weights = conversion_C.StrToMatrix2D(str_vals2D);
				
				//call the constructor when modules a stored as separate ESNs
				_inputLayer = new InputLayer(sub_size[0], weights.getColumnDimension(), _seed);

				_inputLayer.setWeightsInit(weights);
				_inputLayer.setActivation(activation_E.ID, -1);//activation function of input neurons is always linear
				
				esn_part    = _esn_save.getEsnPart(data, network_part_save_E.EPS_IN_MIN_MAX, module_idx);
				str_vals2D  = get2DArrayFromStr(esn_part);
				valid_range = conversion_C.StrToInterval2D(str_vals2D);
				_inputLayer.setMinMaxState(valid_range);
			}
			
			/*** OFB LAYER ***/
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.EPS_OFB_W, module_idx);
			if(esn_part.isEmpty()==false)
			{
				str_vals2D  = get2DArrayFromStr(esn_part);
				weights = conversion_C.StrToMatrix2D(str_vals2D);
				
				//call the constructor when modules a stored as separate ESNs
				_backLayer  = new BackLayer(sub_size[0], weights.getColumnDimension(), _seed);
				
				_backLayer.setWeightsInit(weights);
			}
			
			/*** RESERVOIR ***/

			esn_part   = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_CONNECT, module_idx);
			obj_vals1D = get1DArrayFromStr(esn_part, Double.class);
			connect    = conversion_C.ObjToDouble1D(obj_vals1D);
			
			esn_part   = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_SR, module_idx);
			obj_vals1D = get1DArrayFromStr(esn_part, Double.class);
			sr         = conversion_C.ObjToDouble1D(obj_vals1D);
			
			//call the constructor when modules a stored as separate ESNs
			_internalLayer = new InternalLayer(sub_size[0], connect[0], sr[0], _seed);
			
			esn_part    = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_W_INIT, module_idx);
			str_vals2D  = get2DArrayFromStr(esn_part);
			weights = conversion_C.StrToMatrix2D(str_vals2D);
			_internalLayer.setWeightsInit(weights);
			_internalLayer.computeMaxEigenvalue();
			
			esn_part    = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_MIN_MAX, module_idx);
			str_vals2D  = get2DArrayFromStr(esn_part);
			valid_range = conversion_C.StrToInterval2D(str_vals2D);
			_internalLayer.setMinMaxState(valid_range);
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_ACT, module_idx);
			activation = activation_E.fromStringArray(esn_part);
			_internalLayer.setActivation(activation);
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_BIAS, module_idx);
			str_vals2D = get2DArrayFromStr(esn_part);
			bias     = conversion_C.StrToMatrix2D(str_vals2D);
			_internalLayer.setBias(bias);
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_TOPOLOGY, module_idx);
			topology = ReservoirInitialization.fromString(esn_part.get(0));
			_internalLayer.setInitializationPolicy(topology);
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.EPS_RES_LEAKAGE_RATE, module_idx);
			obj_vals1D = get1DArrayFromStr(esn_part, Double.class);
			leakage_rate = conversion_C.ObjToDouble1D(obj_vals1D);
			_internalLayer.setLeakageRate(leakage_rate);
			
			/*** OUTPUT LAYER ***/
			
			esn_part    = _esn_save.getEsnPart(data, network_part_save_E.EPS_OUT_W, module_idx);
			str_vals2D  = get2DArrayFromStr(esn_part);
			weights = conversion_C.StrToMatrix2D(str_vals2D);

			//call the constructor when modules a stored as separate ESNs
			_outputLayer = new OutputLayer(weights.getRowDimension(), weights.getColumnDimension(), _seed);
			
			_outputLayer.setWeightsInit(weights);
			
			esn_part    = _esn_save.getEsnPart(data, network_part_save_E.EPS_OUT_MIN_MAX, module_idx);
			str_vals2D  = get2DArrayFromStr(esn_part);
			valid_range = conversion_C.StrToInterval2D(str_vals2D);
			_outputLayer.setMinMaxState(valid_range);
			
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.EPS_OUT_ACT, module_idx);
			activation = activation_E.fromStringArray(esn_part);
			_outputLayer.setActivation(activation);
			
			esn_part   = _esn_save.getEsnPart(data, network_part_save_E.EPS_OUT_BIAS, module_idx);
			str_vals2D = get2DArrayFromStr(esn_part);
			bias       = conversion_C.StrToMatrix2D(str_vals2D);
			_outputLayer.setBias(bias);
			
			//load responsibility
			initResponsibility(weights.getRowDimension());
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.MPS_RESPONSIBILITY, module_idx);
			obj_vals1D = get1DArrayFromStr(esn_part, Double.class);
			responsibility = conversion_C.ObjToDouble1D(obj_vals1D);
			setResponsibility(responsibility);
			
			esn_part    = _esn_save.getEsnPart(data, network_part_save_E.MPS_RESPONSIBILITY_MIN_MAX, module_idx);
			obj_vals1D  = get1DArrayFromStr(esn_part, interval_C.class);
			min_max_val = conversion_C.ObjToInterval1D(obj_vals1D);
			setResponsibilityRange(min_max_val);
			
			//load output bias
			initOutputBias(_outputLayer._num_neurons);
			esn_part = _esn_save.getEsnPart(data, network_part_save_E.MPS_OUTPUT_BIAS, module_idx);
			obj_vals1D = get1DArrayFromStr(esn_part, Double.class);
			output_bias = conversion_C.ObjToDouble1D(obj_vals1D);
			setOutputBias(output_bias);
			
			esn_part    = _esn_save.getEsnPart(data, network_part_save_E.MPS_OUTPUT_BIAS_MIN_MAX, module_idx);
			obj_vals1D  = get1DArrayFromStr(esn_part, interval_C.class);
			min_max_val = conversion_C.ObjToInterval1D(obj_vals1D);
			setOutputBiasRange(min_max_val);
		}
		else
		{
			System.err.println("loadEsnModule: specified ESN file does not exist");
			System.exit(1);
		}
	}
	
	/**
	 * The function sets all weights of the module to "0".
	 */
	private void clearWeights()
	{
		//set all reservoir weights to "0"
		_internalLayer.clearWeights();
		
		//set all output weights to "0"
		_outputLayer.clearWeights();
		
		//set possible OFB weights to "0"
		if(_backLayer!=null)
		{
			_backLayer.clearWeights();
		}
		
		//set possible input weights to "0"
		if(_inputLayer!=null)
		{
			_inputLayer.clearWeights();
		}
	}
	
	/**
	 * The function sets reservoir weights, output weights and output feedback weights to the values that were
	 * obtained either in the training or under random generation.
	 */
	private void restoreWeights()
	{		
	 	//set the reservoir weights to generated values and scale them
		_internalLayer.setActiveWeights(true);
		
		//set the output weights to the trained values;
		//input "false" is a dummy value
		_outputLayer.setActiveWeights(false);
		
		//set the OFB weights of the basic and complementary sub-reservoirs to the generated values
		if(_backLayer!=null)
		{
			//input "false" is a dummy value
			_backLayer.setActiveWeights(false);
		}
		
		//set the input weights of the basic and complementary sub-reservoirs to the generated values
		if(_inputLayer!=null)
		{
			//input "false" is a dummy value
			_inputLayer.setActiveWeights(false);
		}
	}
	
	/**
	 * The function assigns the provided interpolation module for the specified ESN output.
	 * 
	 * @param interpolation_module: provided FF-ANN module
	 * @param idx_out: index of the ESN output
	 */
	public void setFfannModule(InterpolationModule interpolation_module, int idx_out)
	{
		_interpolation_module[idx_out] = interpolation_module;
	}
	
	/**
	 * @return the backLayer
	 */
	public BackLayer getBackLayer() {
		return _backLayer;
	}
	
	/**
	 * @return the inputLayer
	 */
	public InputLayer getInputLayer() {
		return _inputLayer;
	}
	
	/**
	 * @return the internalLayer
	 */
	public InternalLayer getInternalLayer() {
		return _internalLayer;
	}
	
	/**
	 * @param backLayer the backLayer to set
	 */
	public void setBackLayer(BackLayer backLayer) {
		this._backLayer = backLayer;
	}
	
	/**
	 * @return the outputLayer
	 */
	public OutputLayer getOutputLayer() {
		return _outputLayer;
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
	 * In this function the initial reservoir state is washed out. Afterwards the reservoir states are collected in
	 * the matrices for the training. These matrices are used to compute the output weights.
	 *   
	 * @param train: training sequence
	 */	
	public void training(seq_C train)
	{
		int i, j;
		int state_len;//length of one state vector; number of columns in the state collector
		int idx_shift;//shift of index used for storing a state vector
		int size_in;//number of input neurons
		int size_res;//number of reservoir neurons
		int size_out;//number of output neurons
		boolean is_out_recurrence;
		double[] sample_in;//temporary array keeping the current input vector 
		double[] sample_out;//temporary array keeping the current output vector
		double[] sample_out_inv;//array of sub-reservoir output for output neurons before applying output activation
		QRDecomposition qrd;//QR decomposition of the matrix with the states of the reservoir neurons
		LUDecomposition lud;//QR decomposition of the matrix with the states of the reservoir neurons
		
		/*
		 * Length of a single row of the output state collector equals
		 * 1) to number of reservoir neurons, if no input neurons and no recurrent connections at output neurons are
		 *    used
		 * 2) to number of reservoir and input neurons, if input neurons are used but no recurrent connections at
		 *    output neurons are used
		 * 3) to number of reservoir, input and output neurons, if input neurons and recurrent connections at output
		 *    neurons are used
		 */
		double[][] stateCollector  = new double[train.getSeqLen()][];
		double[][] targetCollector = new double[train.getSeqLen()][_outputLayer.getRows()];
		
		is_out_recurrence = false;//(!)current modularization concept does not allow using self-recurrent connections
		size_res = _internalLayer.getCols();
		size_out = _outputLayer.getRows();
		sample_out_inv = new double[size_out];
		
		//assign number of input neurons
		if(_inputLayer!=null)
		{
			size_in  = _inputLayer.getCols();
		}
		else
		{
			size_in  = 0;
		}
		
		state_len  = size_res;
		state_len += (_inputLayer!=null) ? size_in  : 0;
		state_len += (is_out_recurrence) ? size_out : 0;
		
		//training
		for(i=0; i < train.getSeqLen(); i++)
		{
			//allocate the next row of the state collector matrix
			stateCollector [i] = new double[state_len];
			targetCollector[i] = new double[size_out];
			//assign the current sample
			sample_in  = train.getSampleIn(i);
			sample_out = train.getSampleOut(i);
			
			//put the input values to the input nodes
			if(_inputLayer!=null)
			{
				_inputLayer.setNodes(sample_in, true);
			}
			
			//calculate states of reservoir neurons
			_internalLayer.setNodes(calculateInternalState(), false);
			
			//collect statistics about min and max values of reservoir states
			_internalLayer.updateValidRange();

			/* store the input values and the current reservoir state which are used to compute output values at
			 * the current training sample
			 */
			idx_shift = 0;
			//store input values of current sample
			if(_inputLayer!=null)
			{
				for(j=0; j < size_in; j++)
				{
					stateCollector[i][j] = _inputLayer.getNodes().transpose().getArray()[0][j];
				}
				idx_shift = size_in;
			}
			//store the reservoir state used for computing the very 1st sample of training sequence
			for(j=0; j < size_res; j++)
			{
				stateCollector[i][idx_shift + j] = _internalLayer.getNodes().transpose().getArray()[0][j];
			}
			idx_shift += size_res;
			//store output feedback used for computing the reservoir state at the current time step
			if(is_out_recurrence)
			{
				for(j=0; j < size_out; j++)
				{
					stateCollector[i][idx_shift + j] = _outputLayer.getNodes().transpose().getArray()[0][j];	
				}
			}
			
			/*assign the current output values to the network output;
			  Remark: it must be done only after saving the previous output values in the state collector*/
			_outputLayer.setNodes(sample_out, true);
			//compute inverse function before the sub-reservoir state is set
			for(j=0; j<size_out; j++)
			{
				sample_out_inv[j] = _outputLayer._func_inverse[j].calculateValue(sample_out[j]);
			}
			//update statistics of the sub-reservoir (param=0 because trained ESN has always a single sub-reservoir)
			_outputLayer.updateValidRange();
			
			//store a target vector at the current time step
			for(j=0; j<size_out; j++)
			{
				targetCollector[i][j] = sample_out_inv[j];
			}
		}
		
		//the computation of the output-weight-matrix
		Matrix outputWeights = new Matrix(stateCollector);
		
		//matrix of reservoir states must be non-singular for the inversion
		qrd = new QRDecomposition(outputWeights);
		lud = new LUDecomposition(outputWeights);
		if(qrd.isFullRank()==true && lud.isNonsingular()==true)
		{
			outputWeights = outputWeights.inverse();
			outputWeights = outputWeights.times(new Matrix(targetCollector));
			outputWeights = outputWeights.transpose();
		
			_outputLayer.setWeights(outputWeights);
			_outputLayer.setWeightsInit(outputWeights);
			_outputLayer.setTrained();
		}
		else
		{
			_outputLayer.resetTrained();
		}
	}
	
	/**
	 * The function calls an output object to close saving the ESN states.
	 */
	public void closeSavingEsnState()
	{
		_esn_state_save.closeSavingEsnStates();
	}
	
	/**
	 * This method updates the output states according to the following formula, as it
	 * can be found in Jaeger's tutorial (Jaeger 2002: A tutorial on training recurrent neural
	 * networks, covering BPPT, RTRL, EKF and the "echo state network" approach):
	 * y(n+1) = f_out (W_out * (u(n+1),x(n+1),y(n)))
	 * were:
	 * 	y(n+1) is the next output-vector
	 * 	f_out is the output-transfer-function
	 * 	W_out is the offline-computed output-weight-matrix (L x (K + N + L))
	 * 	(u(n+1),x(n+1),y(n)) denotes a concatenated vector, consisting of the next
	 * 	input-vector, the formerly computed new internal-state-vector and the current
	 * 	output-vector.
	 * @return Matrix: new output-vector.
	 */
	public double[] calculateOutputVector()
	{
		int i, j;
		int len;//number of neurons which connect to the output neurons
		double[] input_state;//input vector of FF-ANN which is obtained from the parent ESN module
		double[] concatVector;//array which keeps states of all neurons connected to the output neurons
		int shift;//offset for storing states of each type of neurons: input, inner and then output neurons
		double[] inArray;//array of states of input neurons (if the input layer is used)
		double[] intArray;//array of states of internal neurons
		double[] outArray;//array of states of output neurons (if there is recurrence in the output layer)
		double[] outputValues;//array which keeps the product "W_out*(u(n+1),x(n+1),y(n))"
		boolean is_out_recurrence;
		Matrix tmp_matrix;//temporary matrix to compute an output vector
		Matrix wout_0;//output weights of ESN output 0
		Matrix wout_1;//output weights of ESN output 1
		Matrix wout_total;//matrix with the output weights for all ESN outputs together
		
		//*** calculate the linked FF-ANN modules

		input_state = new double[_inputLayer.getNumNodes()];
		_inputLayer.getNodes(input_state);
		for(i=0; i<_interpolation_module.length; i++)
		{
			//compute the hidden states
			_interpolation_module[i].advance(input_state);
		}
		
		//*** compute the output of the ESN module
		
		is_out_recurrence = false;//(!)current modularization concept does not allow using self-recurrent connections
		
		//at first we have to compute the concatenated vector, it's length depends on the
		//usage of an input-layer
		len  = (_inputLayer!=null) ? (_inputLayer.getCols()) : 0;
		len += _internalLayer.getCols();
		len += (is_out_recurrence) ? _outputLayer.getRows() : 0;
		
		concatVector = new double[len];

		//store states of all neurons in one array
		shift = 0;

		//if there is an inputLayer, it's values are added to the vector
		if(_inputLayer!=null)
		{
			inArray = _inputLayer.getNodes().transpose().getArray()[0];
			
			for(j=0; j < inArray.length; j++)
			{
				concatVector[j] = inArray[j];
			}
			shift += inArray.length;//increase the offset for neurons of next type
		}

		//the internal-states are added
		intArray = _internalLayer.getNodes().transpose().getArray()[0];
		for(j=0; j < intArray.length; j++)
		{
			concatVector[j + shift] = intArray[j];
		}
		shift += intArray.length;//increase the offset for neurons of next type

		//the output-states are added, if there is self-recurrence
		if(is_out_recurrence)
		{
			outArray = _outputLayer.getNodes().transpose().getArray()[0];
			for(j=0; j < outArray.length; j++)
			{
				concatVector[j + shift] = outArray[j];
			}
		}
		
		//matrix multiplication W_out * (u(n+1),x(n+1),y(n)) => OUT_input;
		//result is always an array of size [N_out, 1] where "N_out" is a number of the output neurons
		tmp_matrix = new Matrix(concatVector, concatVector.length);
		
		//output weights are obtained from the linked FF-ANN modules
		wout_0 = _interpolation_module[0].GetOutputVector(input_state[0]);
		wout_1 = _interpolation_module[1].GetOutputVector(input_state[0]);
		wout_total = new Matrix(_interpolation_module.length, wout_0.getColumnDimension());
		wout_total.setMatrix(0, 0, 0, wout_0.getColumnDimension()-1, wout_0);
		wout_total.setMatrix(1, 1, 0, wout_1.getColumnDimension()-1, wout_1);
		
		//compute the ESN output
		tmp_matrix = wout_total.times(tmp_matrix);
		
		//copy the ESN output is an array of double
		outputValues = tmp_matrix.getColumnPackedCopy();

		//apply the activation function of the output neurons, it there is one
		for(i=0; i < outputValues.length; i++)
		{
			//states of output neurons are always in a one-column matrix
			outputValues[i] = _outputLayer._func[i].calculateValue(outputValues[i]);
		}

		return outputValues;
	}
	
	/**
	 * The function computes a mean squared error of the ESN at the current time step.
	 * It stores a computed ESN output in the provided array. The function does not try to store the computed value
	 * if "null" is provided instead of the array.
	 * 
	 * @param sample_out: target output vector
	 * @param esn_output: computed ESN output (output array)
	 * @return computed error
	 */
	public double computeError(double[] sample_out, double[] esn_output)
	{
		int i;
		int size_out;//number of output neurons
		double deviation;
		double mse;//output variable
		Matrix output;//computed ESN output
		
		//compute the ESN output
		size_out = _outputLayer.getNumNodes();
		_outputLayer.setNodes(calculateOutputVector(), false);
		output = _outputLayer.getNodes();
		
		mse = 0;
		for(i=0; i<size_out; i++)
		{
			if(esn_output!=null)
			{
				esn_output[i] = output.get(i, 0);
			}
			deviation = (sample_out[i] - output.get(i, 0));
			deviation*= deviation;
			mse += deviation;
		}
		mse /= size_out;
		
		return mse;
	}
	
	/**
	 * The function restores current states of nodes of all layers from specified elements of history arrays.
	 * The elements of history arrays are specified by their index.
	 * 
	 * @param element_idx: index of elements of history arrays where nodes' states must be taken
	 */
	public void restoreNodesHistory(int element_idx)
	{
		if(_inputLayer!=null)
		{
			_inputLayer.restoreNodesHistory(element_idx);
		}
		_internalLayer.restoreNodesHistory(element_idx);
		_outputLayer.restoreNodesHistory(element_idx);
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates a number of nodes in a specified layer.
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
				if(_inputLayer!=null)
				{
					num_nodes = _inputLayer.getNumNodes();
				}
				else
				{
					num_nodes = 0;
				}
				break;
			case LT_RES:
				num_nodes = _internalLayer.getNumNodes();
				break;
			case LT_OUTPUT:
				num_nodes = _outputLayer.getNumNodes();
				break;
			case LT_OFB:
				System.err.println("EsnModule.getNumNodes: OFB does not contain nodes");
				System.exit(1);
				num_nodes = -1;
				break;
			default:
				System.err.println("EsnModule.getNumNodes: unknown layer type");
				System.exit(1);
				num_nodes = -1;
				break;
		}
		
		return num_nodes;
	}
	
	/**
	 * The function is dummy function of a parent's abstract method.
	 * The function shall assign values from a provided array to nodes of a specified layer.
	 * If requested, the function shall apply an activation function to the provided values before assignment.
	 * 
	 * @param node: values to be assigned
	 * @param layer_type: requested layer
	 * @param apply_activation: request to apply an activation function
	 */
	public void setNodes(double[] node, layer_type_E layer_type, boolean apply_activation)
	{
		switch(layer_type)
		{
			case LT_INPUT:
				if(_inputLayer!=null)
				{
					_inputLayer.setNodes(node, apply_activation);
				}
				break;
			case LT_RES:
				_internalLayer.setNodes(node, apply_activation);
				break;
			case LT_OFB:
				_backLayer.setNodes(node, apply_activation);
				break;
			case LT_OUTPUT:
				_outputLayer.setNodes(node, apply_activation);
				break;
			default:
				System.err.println("EsnModule.setNodes: unknown layer type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function returns an array with values of nodes of a specified layer.
	 *
	 * @param layer_type: requested layer
	 * @return: array of output values
	 */
	public double[] getNodes(layer_type_E layer_type)
	{
		double[] nodes;//output array
		Matrix mtx_nodes;
		
		switch(layer_type)
		{
			case LT_INPUT:
				nodes = _inputLayer.getNodes().getArray()[0];
				break;
			case LT_RES:
				nodes = _internalLayer.getNodes().getArray()[0];
				break;
			case LT_OUTPUT:
				mtx_nodes = _outputLayer.getNodes();
				nodes = mtx_nodes.getColumnPackedCopy(); //a single column must be assigned to "nodes"
				break;
			case LT_OFB:
				nodes = null;
				System.err.println("EsnModule.getNodes: node nodes in OFB layer");
				System.exit(1);
				break;
			default:
				nodes = null;
				System.err.println("EsnModule.setNodes: unknown layer type");
				System.exit(1);
				break;
		}
		return nodes;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function stores states of nodes of a specified layer in a provided storage array.
	 *
	 * @param layer_type: requested layer
	 * @param storage: provided storage
	 */
	public void getNodes(double[] storage, layer_type_E layer_type)
	{
		switch(layer_type)
		{
			case LT_INPUT:
				if(_inputLayer!=null)
				{
					_inputLayer.getNodes(storage);
				}
				break;
			case LT_RES:
				_internalLayer.getNodes(storage);
				break;
			case LT_OUTPUT:
				_outputLayer.getNodes(storage);
				break;
			case LT_OFB:
				System.err.println("EsnModule.getNodes: node nodes in OFB layer");
				System.exit(1);
				break;
			default:
				System.err.println("EsnModule.setOutputNodes: unknown layer type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates a valid range for each output element of a module.
	 * 
	 * @return valid ranges for all output elements of a module
	 */
	public interval_C[][] getOutputValidRange()
	{
		return _outputLayer.getMinMaxState();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates a valid range for each input element of the module.
	 * 
	 * @return valid ranges for all input elements of the module
	 */
	public interval_C[][] getInputValidRange()
	{
		return _inputLayer.getMinMaxState();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates a valid range for each internal node of a module.
	 * 
	 * @return valid ranges for all internal nodes of a module
	 */
	public interval_C[][] getInternalValidRange()
	{
		return _internalLayer.getMinMaxState();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function stores a valid intervals of a specified node in a provided storage.
	 * The node is specified by its index in a specified range.
	 * 
	 * @param node_idx: index of a node
	 * @param layer_type: layer type
	 * @param range: storage where retrieved valid intervals must be stored
	 */
	public void getNodeValidRange(int node_idx, layer_type_E layer_type, interval_C[] range)
	{
		int i;
		
		for(i=0; i<range.length; i++)
		{
			switch(layer_type)
			{
				case LT_INPUT:
					range[i].copy(_inputLayer._valid_range[node_idx][i]);
					break;
				case LT_RES:
					range[i].copy(_internalLayer._valid_range[node_idx][i]);
					break;
				case LT_OUTPUT:
					range[i].copy(_outputLayer._valid_range[node_idx][i]);
					break;
				case LT_OFB:
					range[i].copy(_backLayer._valid_range[node_idx][i]);
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
	 * The function indicates TRUE if there are OFB connections in an ESN module; FALSE - otherwise.
	 */
	public boolean ExistBackLayer()
	{
		boolean does_backlayer_exist;//output variable
		
		if(_backLayer!=null)
		{
			does_backlayer_exist = true;
		}
		else
		{
			does_backlayer_exist = false;
		}
		
		return does_backlayer_exist;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function is an interface to store current states of all neurons of the module for their possible activation
	 * later. The states are stored in a specified storage. The storage is specified by its type.
	 * 
	 * @param type: storage type
	 */
	public void storeNodes(storage_type_E type)
	{
		if(_inputLayer!=null)
		{
			_inputLayer.storeNodes(type);
		}
		_internalLayer.storeNodes(type);
		_outputLayer.storeNodes(type);
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function is an interface to restore preserved initial states of all neurons of the module.
	 * The states are retrieved from a specified storage. The storage is specified by its type.
	 */
	public void restoreNodes(storage_type_E type)
	{
		if(_inputLayer!=null)
		{
			_inputLayer.restoreNodes(type);
		}
		_internalLayer.restoreNodes(type);
		_outputLayer.restoreNodes(type);
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function is an interface to clear current states of all neurons of the module.
	 * The function initiates a setting of concerned module states to "0". 
	 */
	public void clearNodes()
	{
		if(_inputLayer!=null)
		{
			_inputLayer.clearNodes();
		}
		_internalLayer.clearNodes();
		_outputLayer.clearNodes();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function create a history array for each layer of an ESN module.
	 * All created history array have the same length which is specified as an input parameter.
	 * 
	 * @param hist_len: required length of history arrays
	 */
	public void createNodesHistory(int hist_len)
	{
		if(_inputLayer!=null)
		{
			_inputLayer.createNodesHistory(hist_len);
		}
		_internalLayer.createNodesHistory(hist_len);
		_outputLayer.createNodesHistory(hist_len);
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function prepares history arrays of all layers of a module for storing the next elements.
	 */
	public void startNodesHistory()
	{
		if(_inputLayer!=null)
		{
			_inputLayer.startNodesHistory();
		}
		_internalLayer.startNodesHistory();
		_outputLayer.startNodesHistory();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function stores current states of nodes of all layers at the last element of a history array.
	 */
	public void storeNodesHistory()
	{
		if(_inputLayer!=null)
		{
			_inputLayer.storeNodesHistory();
		}
		_internalLayer.storeNodesHistory();
		_outputLayer.storeNodesHistory();
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
		InputLayer input_layer;
		
		switch(layer_type)
		{
			case LT_INPUT:
				input_layer = getInputLayer();
				if(input_layer!=null)
				{
					input_layer.getNodesHistory(idx_element, storage);
				}
				break;
			case LT_RES:
				getInternalLayer().getNodesHistory(idx_element, storage);
				break;
			case LT_OUTPUT:
				getOutputLayer().getNodesHistory(idx_element, storage);
				break;
			case LT_OFB:
				System.err.println("EsnModule.getNodesHistory: no history at the OFB layer");
				System.exit(1);
				break;
			default:
				System.err.println("EsnModule.getNodesHistory: unknown layer type");
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
		InputLayer input_layer;
		
		switch(layer_type)
		{
			case LT_INPUT:
				input_layer = getInputLayer();
				if(input_layer!=null)
				{
					input_layer.getNodesHistory(storage);
				}
				break;
			case LT_RES:
				getInternalLayer().getNodesHistory(storage);
				break;
			case LT_OUTPUT:
				getOutputLayer().getNodesHistory(storage);
				break;
			case LT_OFB:
				System.err.println("EsnModule.getNodesHistory: no history at the OFB layer");
				System.exit(1);
				break;
			default:
				System.err.println("EsnModule.getNodesHistory: unknown layer type");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function indicates whether a spectral radius of a module is not zero.
	 * 
	 * @return: TRUE if spectral radius is not zero; FALSE - otherwise
	 */
	public boolean getNonZeroSR()
	{
		return _internalLayer.isMaxEigenvalueNotZero();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function shall indicate whether a module is trained.
	 */
	public boolean isTrained()
	{
		return _outputLayer.isTrained();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function enables a contribution from a module to the whole modular network.
	 * 
	 * @param restore_states (ESN-specific parameter): request to restore states of all neurons in a module
	 */
	public void activate(boolean restore_states)
	{
		restoreWeights();
		if(restore_states==true)
		{
			restoreNodes(storage_type_E.ST_INIT);
		}
		setConfigured();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function disables a contribution from a module to the whole modular network.
	 */
	public void deactivate()
	{
		clearWeights();
		clearNodes();
		resetConfigured();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function performs an update of the ESN module under transition to the next time step.
	 * The function assigns values from a provided array to states of input neurons.
	 * These values are values of the input neurons at the next time step.
	 * If no input neurons exist then no assignment is performed.
	 * 
	 * The functions stores computed output neuron in the history if it is required.
	 * 
	 * The function also initiates calculation of the linked FF-ANN modules.
	 * 
	 * @param sample_in: input sample at the next time step
	 * @param do_store: request to store computed output neurons in the history
	 */
	public void advance(double[] sample_in, boolean do_store)
	{
		//put the input values to the input nodes
		if(_inputLayer!=null)
		{
			if(sample_in!=null)
			{
				_inputLayer.setNodes(sample_in, true);
			}
		}
		
		//compute states of internal neurons
		_internalLayer.setNodes(calculateInternalState(), false);
		
		//compute the ESN output in the current step on training sequence
		_outputLayer.setNodes(calculateOutputVector(), false);
		
		//store output value in the history if it is required
		if(do_store==true)
		{
			storeNodesHistory();
		}
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function shall indicate a matrix of initial weights of a specified layer as an array of strings.
	 * 
	 * @param layer_type: specified layer
	 * @return: initial weights as an array of strings
	 */
	public Vector<String> getInitWeightsAsStr(layer_type_E layer_type)
	{
		Vector<String> str_init_w;
		
		switch(layer_type)
		{
			case LT_INPUT:
				str_init_w = _inputLayer.getInitWeightsAsStr();
				break;
			case LT_RES:
				str_init_w = _internalLayer.getInitWeightsAsStr();
				break;
			case LT_OFB:
				str_init_w = _backLayer.getInitWeightsAsStr();
				break;
			case LT_OUTPUT:
				str_init_w = _outputLayer.getInitWeightsAsStr();
				break;
			default:
				str_init_w = null;
				System.err.println("EsnModule.getNodesHistory: unknown layer type");
				System.exit(1);
				break;
		}
		
		return str_init_w;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function converts values of a parameter which is common for all ESN layers to an array of strings.
	 * The function terminates a program with an error, if no conversion is implemented for a specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @param layer_type: requested layer
	 * @return: values of the specified parameter as an array of strings
	 */
	public Vector<String> getCommonLayerParamAsStr(exp_param_E param, layer_type_E layer_type)
	{
		Vector<String> param_val;
		
		switch(layer_type)
		{
			case LT_INPUT:
				param_val = _inputLayer.getLayerParamAsStr(param);
				break;
			case LT_RES:
				param_val = _internalLayer.getLayerParamAsStr(param);
				break;
			case LT_OFB:
				param_val = _backLayer.getLayerParamAsStr(param);
				break;
			case LT_OUTPUT:
				param_val = _outputLayer.getLayerParamAsStr(param);
				break;
			default:
				param_val = null;
				System.err.println("EsnModule.getCommonLayerParamAsStr: unknown layer type");
				System.exit(1);
				break;
		}
		
		return param_val;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * This function stores current weights for their possible activation later.
	 */
	public void storeWeightsInit()
	{
		_outputLayer.storeWeightsInit();
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function converts values of a parameter which is specific to a specified ESN layer to an array of strings.
	 * The function terminates a program with an error, if no conversion is implemented for a specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @param layer_type: requested layer
	 * @return: values of the specified parameter as an array of strings
	 */
	public Vector<String> getSpecificLayerParamAsStr(exp_param_E param, layer_type_E layer_type)
	{
		Vector<String> param_val;
		
		switch(layer_type)
		{
			case LT_RES:
				param_val = _internalLayer.getParamValAsStr(param);
				break;
			case LT_INPUT:
			case LT_OFB:
			case LT_OUTPUT:
			default:
				param_val = null;
				System.err.println("EsnModule.getSpecificLayerParamAsStr: unknown layer type");
				System.exit(1);
				break;
		}
		
		return param_val;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * The function presents a connectivity matrix of reservoir weights as an array of strings.
	 * This function is necessary only for an ESN module.
	 * 
	 * @return: connectivity matrix as an array of strings
	 */
	public Vector<String> getConnectivityMatrixAsStr()
	{
		Vector<String> connectivity;//output array
		
		connectivity = _internalLayer.getConnectivityMatrixAsStr();
		
		return connectivity;
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
		Vector<String> connect_str;
		
		switch(layer_type)
		{
			case LT_INPUT:
				connect_str = _inputLayer.getLayerValidRangeAsStr();
				break;
			case LT_RES:
				connect_str = _internalLayer.getLayerValidRangeAsStr();
				break;
			case LT_OFB:
				connect_str = _backLayer.getLayerValidRangeAsStr();
				break;
			case LT_OUTPUT:
				connect_str = _outputLayer.getLayerValidRangeAsStr();
				break;
			default:
				connect_str = null;
				System.err.println("EsnModule.getLayerMinMaxAsStr: unknown layer type");
				System.exit(1);
				break;
		}
		
		return connect_str;
	}
	
	/**
	 * The function is an implementation of the abstract method from the class "Module".
	 * Since ESN modules do not have parameters whose values can be mapped, the function always returns a submitted
	 * value.
	 * 
	 * @param param_idx: index of module parameter (dummy parameter)
	 * @param init_value: submitted parameter value
	 * @return: submitted values
	 */
	public double mapParamToEquivalentValue(int param_idx, double init_value)
	{
		return init_value;
	}
}
