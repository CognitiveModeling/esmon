package esn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import adaptation.DiffEvolution;
import adaptation.DiffEvolutionError;
import adaptation.DiffEvolutionMature;
import adaptation.DiffEvolutionParam;
import adaptation.DiffEvolutionParam.config_ea_init_E;

import types.interval_C;
import types.seq_parameter_C;
import types.stat_seq_C;
import types.vector_C;

import Jama.Matrix;
import MathDiff.MathDistortion;
import MathDiff.MathPerformance;

import esn.Module.layer_type_E;
import esn.Module.module_type_E;
import esn.Module.storage_type_E;
import experiment.ExpOutput;
import experiment.ExpOutput.esn_output_C;
import experiment.ExpOutput.network_part_save_E;
import experiment.ExpOutput.esn_state_C;
import experiment.ExpParam;
import experiment.ExpParam.exp_param_E;
import experiment.ExpParam.req_val_E;
import experiment.ExpSeq.seq_C;

/**
 * The main class for creating, training and saving ESN's. In it's current form it only
 * supports batch-learning.
 * @author Danil Koryakin, Johannes Lohmann
 *
 */
public class mESN
{
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

	private Module[]      _module;//array of all created ESN modules
	private esn_state_C   _esn_state_save;//object which keeps parameters of the file where current ESN state is saved
	private esn_output_C  _esn_save;//object which keeps parameters of the file where the host ESN is saved
	private double     _sel_thresh;//threshold for computing the small-error length
	private double     _lel_thresh;//threshold for computing the large-error length
	private double[]   _tmp_responsibility;//exchange storage to retrieve responsibilities from population
	private double[]   _tmp_output_bias;//exchange storage to retrieve the output bias from population
	private double[]   _output;//output states of mESN's output neurons
	private boolean _is_configured;//indicator that modules are configured
	private DiffEvolution[] _diff_evolution;//array of objects to configure ESN modules with differential evolution
	private DiffEvolutionMature _diff_evolution_mature;//maturing pool

	final int _num_esn_output = 2;//number of ESN outputs for loading FF-ANN modules
	final int _window_sub_target_approx = 10;//length of the time window for computing the error under configuration
	                                         //through estimation of the target values for the separate sub-reservoirs
	final int _max_num_seq_combinations = 10000000;//maximum number of sequence combinations when the brute force
	                                               //approach is applied;
	                                       //!!!This value was found empirically.
	                                       //!!!For larger numbers of combinations, the brute force search is too long.
	final double _error_max = 1E-5;//max error to indicate successful configuration
	final double _delta_out_portion = 0.01;//valid range of output bias as a portion of the whole output range
	
	/**
	 * This is a constructor generating an ESN using loaded parameters.
	 * 
	 * @param esn_load_files: array of ESN files for loading corresponding modules
	 * @param ffann_wout0_load_files: array of FF-ANN files for ESN output 0
	 * @param ffann_wout1_load_files: array of FF-ANN files for ESN output 1
	 * @param exp_param: required parameters of ESN
	 * @param exp_output: object of a class responsible for the file output
	 * @param commonPathCurrRun: common incomplete path to save algorithm specific data
	 * @param file_esn_save: path to a file where the host ESN must be saved
	 * @param seed: array of values for seeding a random numbers generator of each module; new values can be assigned
	 * @param seed_load: indicator to load seeding value for ESN modules
	 */
	public mESN(String[] esn_load_files, String[] ffann_wout0_load_files, String[] ffann_wout1_load_files,
			    ExpParam exp_param, ExpOutput exp_output, String commonPathCurrRun, int[] seed, boolean[] seed_load)
	{
		int i;
		int size_out;
		int w_out_num;//number of output weights in ESN
		int num_sub;//number of ESN modules
		boolean f_save;//indicator of a request to save the run data in files
		module_type_E module_type;//type of module in a loaded file
		vector_C output_bias;//loaded bias of the total network output
		InterpolationModule interpolation_module;
		
		_sel_thresh = (Double)exp_param.getParamVal(exp_param_E.EP_PERFORM_SEL_THRESH, req_val_E.RV_CUR);
		_lel_thresh = (Double)exp_param.getParamVal(exp_param_E.EP_PERFORM_LEL_THRESH, req_val_E.RV_CUR);
		_is_configured = false;
		
		f_save = (Boolean)exp_param.getParamVal(exp_param_E.EP_IO_SAVE_DATA, req_val_E.RV_CUR);
		
		//prepare an object for saving the host ESN
		_esn_save = prepareSavingEsn(f_save, exp_output, commonPathCurrRun);
		
		num_sub = seed.length;
		
		//create all ESN modules
		_module = new Module[num_sub];
		
		for(i=0; i<num_sub; i++)
		{
			//check whether a module must be loaded;
			//if no modules are loaded then only ESN modules are created 
			if(esn_load_files[i].contains("*")==false)
			{
				//load oscilator modules
				module_type = getModuleTypeFromFile(esn_load_files[i]);
				switch(module_type)
				{
					case MT_ESN:
						_module[i] = new EsnModule(i, esn_load_files[i], exp_param, exp_output, commonPathCurrRun,
								                   seed[i], seed_load[i]);
						break;
					case MT_FFANN:
						_module[i] = new FfannModule(i, esn_load_files[i], exp_param, exp_output, commonPathCurrRun,
				                   seed[i], seed_load[i]);
						break;
					case MT_SIN:
						_module[i] = new SinModule(i, esn_load_files[i], exp_param, exp_output, commonPathCurrRun,
								                   seed[i], seed_load[i]);
						break;
					default:
						System.err.println("mESN: unknown module type is provided");
						System.exit(1);
						_module[i] = null;
						break;
				}
				
				//load FF-ANN module for output 0
				module_type = getModuleTypeFromFile(ffann_wout0_load_files[i]);
				switch(module_type)
				{
					case MT_INTERPOLATION:
						//establish links to FF-ANN from the corresponding ESN module
						if(_module[i].getClass()==EsnModule.class)
						{
							w_out_num = ((EsnModule)_module[i]).getNumNodes(layer_type_E.LT_RES) +
							            ((EsnModule)_module[i]).getNumNodes(layer_type_E.LT_INPUT);
							//load FF-ANN for output 0 of the ESN module
							interpolation_module = new InterpolationModule(i, ffann_wout0_load_files[i], exp_param,
									                                       exp_output, commonPathCurrRun,w_out_num);
							((EsnModule)_module[i]).setFfannModule(interpolation_module, 0);
						}
						else
						{
							System.err.println("mESN: interpolation is linked only to ESN modules");
							System.exit(1);
						}
						break;
					default:
						System.err.println("mESN: unknown module type is provided instead of INTERPOLATION");
						System.exit(1);
						_module[i] = null;
						break;
				}
				
				//load FF-ANN module for output 1
				module_type = getModuleTypeFromFile(ffann_wout1_load_files[i]);
				switch(module_type)
				{
					case MT_INTERPOLATION:
						//establish links to FF-ANN from the corresponding ESN module
						if(_module[i].getClass()==EsnModule.class)
						{
							w_out_num = ((EsnModule)_module[i]).getNumNodes(layer_type_E.LT_RES) +
				                        ((EsnModule)_module[i]).getNumNodes(layer_type_E.LT_INPUT);
							interpolation_module = new InterpolationModule(i, ffann_wout1_load_files[i], exp_param, exp_output,
                                                                           commonPathCurrRun, w_out_num);
							((EsnModule)_module[i]).setFfannModule(interpolation_module, 1);
						}
						else
						{
							System.err.println("mESN: interpolation is linked only to ESN modules");
							System.exit(1);
						}
						break;
					default:
						System.err.println("mESN: unknown module type is provided instead of INTERPOLATION");
						System.exit(1);
						_module[i] = null;
						break;
				}
			}
			else
			{
				_module[i] = new EsnModule(i, esn_load_files[i], exp_param, exp_output, commonPathCurrRun,
		                                   seed[i], seed_load[i]);
			}
			seed[i] = _module[i].getSeed();
		}
		
		//prepare an object for saving ESN states
		_esn_state_save = prepareSavingEsnState(f_save, exp_output, commonPathCurrRun);
		
		//allocate mESN's output (number of mESN's output neurons is equal to a number of output neurons of each module)
		//therefore, it is enough to address the 1st module for a number of output neurons
		size_out = _module[0].getNumNodes(layer_type_E.LT_OUTPUT);
		_output = new double[size_out];
	}
	
	/**
	 * The function indicates a type of module provided in a specified module file.
	 * The module file is specified by its path.
	 * 
	 * @param module_path: path to a module file to be loaded
	 * @return type of module
	 */
	private module_type_E getModuleTypeFromFile(String module_path)
	{
		int module_idx = 0;//module file shall always contain module index 0
		boolean is_available;//indicator whether a required module index or type is available in a loaded file
		File module_file;//file object
		String tmp_str;//temporary string
		module_type_E module_type;
		Vector<String> data;//vector of strings loaded from the file
		Vector<String> module_part;//vector of strings with loaded values of a certain part
		
		data = new Vector<String>(0,1);
		module_file = new File(module_path);
		try {
			BufferedReader reader = new BufferedReader(new FileReader(module_file));
			
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
			//*** search for a module index in loaded file data
			
			is_available = Module.isModuleIdxAvailable(data, module_idx);
			//the file probably contains a module with index "0" if a specified index is not available
			if(is_available==false)
			{
				module_idx = 0;
				is_available = Module.isModuleIdxAvailable(data, module_idx);
				if(is_available==false)
				{
					System.err.println("mESN.getModuleTypeFromFile: file does not contain any module");
					System.exit(1);
				}
			}
			
			//*** search for a module type after determining a module index
			
			module_type = module_type_E.MT_UNKNOWN;//initialize a module type
			module_part = _esn_save.getEsnPart(data, network_part_save_E.MPS_MODULE_TYPE, module_idx);
			if(module_part.size()!=0)
			{
				tmp_str = module_part.get(module_idx);
				is_available = tmp_str.contains(module_type_E.MT_SIN.extractName());
				if(is_available==true)
				{
					module_type = module_type_E.MT_SIN;
				}
				if(module_type==module_type_E.MT_UNKNOWN)//if not SIN
				{
					is_available = tmp_str.contains(module_type_E.MT_ESN.extractName());
					if(is_available==true)
					{
						module_type = module_type_E.MT_ESN;
					}
				}
				if(module_type==module_type_E.MT_UNKNOWN)//if neither ESN nor SIN
				{
					is_available = tmp_str.contains(module_type_E.MT_FFANN.extractName());
					if(is_available==true)
					{
						module_type = module_type_E.MT_FFANN;
					}
				}
				if(module_type==module_type_E.MT_UNKNOWN)//if neither ESN nor SIN, nor FF-ANN
				{
					is_available = tmp_str.contains(module_type_E.MT_INTERPOLATION.extractName());
					if(is_available==true)
					{
						module_type = module_type_E.MT_INTERPOLATION;
					}
				}
				if(module_type==module_type_E.MT_UNKNOWN)//if not SIN and not ESN
				{
					System.err.println("mESN.getModuleTypeFromFile: no module in provided file");
					System.exit(1);
				}
			}
			else//in ESN modules type was not specified only in ESN modules
			{
				System.err.println("mESN.getModuleTypeFromFile: no module type in provided file");
				System.exit(1);
			}
		}
		else
		{
			System.err.println("mESN.getModuleTypeFromFile: no module in provided file");
			System.exit(1);
			module_type = module_type_E.MT_UNKNOWN;
		}
		
		return module_type;
	}
	
	/**
	 * The function chooses a training time step which can be used as the beginning time step for the started period of
	 * the sub-reservoir adjustment.
	 * The function chooses the time index so that the number of the following time steps are enough to finish
	 * the started period of adjustment. The number of the following time steps together with the chosen beginning
	 * sample must be larger or equal to the provided length of the period of adjustment.
	 * @param step_begin_alternative: ordered array of possible time steps (time steps to the left are more preferable)
	 * @param seq_len: length of the training sequence
	 * @param period: length of the period of adjustment
	 * @return: index of the chosen time step
	 */
	private int chooseStepBeginByRestLength(int[] step_begin_alternative, int seq_len, int period)
	{
		int     i;
		int     rest_len;//number of time steps until the end of the sequence
		int     step_begin;//output variable
		boolean is_chosen;//indicator that the beginning time step is successfully chosen
		
		is_chosen  = false;
		step_begin = -1;
		for(i=0; i<step_begin_alternative.length && is_chosen==false; i++)
		{
			rest_len = seq_len - step_begin_alternative[i];
			
			if(rest_len <= 0)
			{
				System.err.println("chooseBeginIdxByRestLength: invalid time step in provided array");
				System.exit(1);
			}
			if(rest_len > period)
			{
				step_begin = step_begin_alternative[i];
				is_chosen  = true;
			}
		}
		if(step_begin==-1)
		{
			System.err.println("chooseBeginIdxByRestLength: not possible to choose the beginning time step");
			System.exit(1);
		}
		return step_begin;
	}
	
	/**
	 * The function computes the mean-squared error (MSE) over a sliding time window described by the provided buffer.
	 * First, the function uses the provided vector of the target values and the provided ESN output to compute
	 * a vector of deviations of the ESN output from these target values.
	 * Then the function stores the computed deviations in the array of the sliding window. Finally it computes the MSE
	 * over the sliding window. In order to store the current deviations in the buffer, all rows of the buffer are
	 * shifted to the top. The elements of the 0th row are overwritten with values of the 1st row, the 1st - with
	 * the 2nd and so on. The current deviations are stored then in the last row of the buffer.
	 * @param target: provided vector of target values
	 * @param esn_output: provided ESN output
	 * @param output_deviation: provided buffer of the sliding time window
	 * @return: array of errors for all output dimensions over the time window
	 */
	private double[] computeErrorWindow(double[] target, double[] esn_output, double[][] output_deviation)
	{
		int i, j;
		int size_out;//number of the output neurons
		double[] output_deviation_cur;//vector of current deviations of ESN output from the target dynamics
		double[] error;//output array

		size_out = esn_output.length;
		output_deviation_cur = new double[size_out];
		error                = new double[size_out];
		
		//compute the deviation of the ESN output from the target at the current time step		
		for(i=0; i<size_out; i++)
		{
			output_deviation_cur[i] = esn_output[i] - target[i];
		}
		//shift the contents of the window buffer by one row to the top
		for(i=0; i<_window_sub_target_approx-1; i++)
		{
			output_deviation[i] = output_deviation[i+1];
		}
		//store the deviation in the buffer
		output_deviation[_window_sub_target_approx-1] = output_deviation_cur;
		
		//compute the MSE in the window
		for(i=0; i<size_out; i++)
		{
			error[i] = 0;
			for(j=0; j<_window_sub_target_approx; j++)
			{
				error[i] += Math.pow(output_deviation[j][i], 2);
			}
			error[i] /= _window_sub_target_approx;
		}
		
		return error;
	}
	
	/**
	 * The function searches for a sequence of time steps in the training sequences of separate sub-reservoirs which
	 * should produce the configuration sequence with the smallest possible error. The relevant sub-reservoirs are
	 * specified by their largest index. The sequence of time steps is sought by a brute-force approach. In this
	 * approach all possible combinations of sequences are analyzed.
	 * The obtained sequence represents the best possible sequence of time steps which can be found with other
	 * configuration approaches.
	 * @param seq_config: configuration approach
	 * @param seq_train: array of training sequences of separate sub-reservoirs
	 * @param max_idx_sub: maximum index of the active sub-reservoir
	 * @return: array of time steps which produce the configuration sequence from the provided training sequences
	 */
	private int[][] findSubTargetsByBruteForce(seq_C seq_config, seq_C[] seq_train, int max_idx_sub)
	{
		int i,j;
		int idx_comb;//index of the currently analyzed combination
		int num_all_comb;//total number of possible combinations which can be produced by provided training sequences
		int num_sub;//number of sub-reservoirs
		int len_seq_config;//length of the configuration sequence
		int[]   time_step_begin;//array of beginning time steps in train sequences to produce the config sequence
		int[][] time_step_cur;//currently analyzed sequence of time steps: 1D: time step, 2D sub-reservoir
		int[][] time_step_best;//output array: 1D: time step, 2D sub-reservoir
		double mse;//mean-squared error (MSE) by the currently checked sequence of time steps 
		double mse_min;//smallest MSE among already checked sequences
		double[][] esn_output;//ESN output which can be computed from the sub-reservoir outputs
		
		num_sub = max_idx_sub+1;
		len_seq_config = seq_config.getSeqLen();
		//initialize the arrays to be used
		time_step_cur   = new int[len_seq_config][];
		time_step_best  = new int[len_seq_config][];
		time_step_begin = new int[num_sub];
		esn_output      = new double[len_seq_config][];
		for(i=0; i<len_seq_config; i++)
		{
			time_step_cur[i]  = new int[num_sub];
			time_step_best[i] = new int[num_sub];
		}
		
		//compute the number of possible combinations to be analyzed		
		num_all_comb = seq_train[0].getSeqLen();
		time_step_begin[0] = 0;
		for(i=1; i<num_sub; i++)
		{			
			num_all_comb *= seq_train[i].getSeqLen();
			//cut the number of sequence combinations, if their computed number is larger than largest allowed number
			if(num_all_comb > _max_num_seq_combinations)
			{
				num_all_comb = _max_num_seq_combinations;
			}
			time_step_begin[i] = 0;
		}
		
		mse_min = Double.MAX_VALUE;
		for(idx_comb=0; idx_comb<num_all_comb; idx_comb++)
		{
			//initialize indices of the current time steps with the indices of beginning time steps
			for(i=0; i<num_sub; i++)
			{
				time_step_cur[0][i] = time_step_begin[i];
			}
			
			//check the match between the configuration sequence and the current combination of train sequences
			for(i=0; i<len_seq_config; i++)
			{
				esn_output[i] = computeSumSeq(seq_train, time_step_cur[i]);
				//update the current indices in the separate train sequences, if not the last time step
				if(i < len_seq_config-1)
				{
					for(j=0; j<num_sub; j++)
					{
						//ATTENTION: think about how to make a wrap around when the next index is already out of bounds;
						//           at the moment this is just a simple incrementing and can go beyond the last index
						//           of the sequence
						time_step_cur[i+1][j] = time_step_cur[i][j] + 1;
					}
				}
			}
			mse = MathPerformance.computeMseTotal(esn_output, seq_config.getSeq());
			//store the analyzed sequence as a possible candidate for output, if computed MSE is smaller than before
			if(mse < mse_min)
			{
				mse_min = mse;
				//store the sequence of indices corresponding to the computed MSE
				for(i=0; i<len_seq_config; i++)
				{
					for(j=0; j<num_sub; j++)
					{
						time_step_best[i][j] = time_step_cur[i][j];
					}
				}
			}
			
			//update the indices of the beginning time steps
			i = num_sub - 1;
			do{
				if(time_step_begin[i]==seq_train[i].getSeqLen())
				{
					time_step_begin[i] = 0;
					i--;
				}
				//"-1" appears when all time steps of all training sequences were checked
				if(i > -1)
				{
					time_step_begin[i]++;
				}
			}while(i!=-1 && time_step_begin[i]==seq_train[i].getSeqLen());
		}
		
		return time_step_best;
	}
	
	/**
	 * The function computes a sum of the output samples of the provided sequences. In general, the samples can be taken
	 * at different time steps.
	 * 
	 * @param seq: array of provided sequences
	 * @param time_step: array of the time steps
	 * @return: sum of the samples
	 */
	private double[] computeSumSeq(seq_C[] seq, int[] time_step)
	{
		int i,j;
		int num_out;//number of the output elements in the sample
		double[] sample;//currently considered sample
		double[] sum;//output array

		num_out = _output.length;
		//initialize the sum
		sum = new double[num_out];
		
		//compute the sum
		for(i=0; i<num_out; i++)
		{
			sum[i] = 0;
			//go over the provided sequences
			for(j=0; j<time_step.length; j++)
			{
				//compute the ESN output only, if the sample index is within the bounds
				if(time_step[j] < seq[j].getSeqLen())
				{
					sample = seq[j].getSampleOut(time_step[j]);
					sum[i]+= sample[i];
				}
				else
				{
					sum[i]+= 0;//add only "0", if the index is out of bounds
				}
			}
		}
		return sum;
	}
	
	/**
	 * The function computes the target values for all ESN module. The target value for every active complementary
	 * module is computed as a difference of the corresponding output sequence value and the output value from
	 * the sequence with the next smaller index. The target value of the basic module obtains output values from
	 * the sequence with index 0. All target values of inactive reservoirs are assigned "0".
	 * 
	 * @param seq: array of sequences to compute the target values
	 * @param idx_time: index of time step where the target values should be computed
	 * @return: 2D array of target values for all possible sub-reservoirs. The 1st dimension corresponds to indices
	 *          of modules. The 2nd dimension corresponds to indices of the output neurons of the whole ESN.
	 */
	private double[][] computeTargetSub(seq_C seq[], int idx_time)
	{
		int i, j;
		int num_out;//number of output neurons
		int num_sub;//number of sub-reservoirs
		double[]   sample_out;//output elements of current sample
		double[]   sample_out2;//output elements of current sample in 2nd sequence
		double[][] sub_output;
		
		num_out = seq[0].getSampleOut(0).length;
		num_sub = _module.length;
		//allocate the output array
		sub_output = new double[num_sub][];
		
		//allocate memory for the output of sub-reservoirs
		for(i=0; i<num_sub; i++)
		{
			sub_output[i] = new double[num_out];
		}
		
		//assign the output of the basic sub-reservoir
		sample_out = seq[0].getSampleOut(idx_time);
		for(j=0; j<num_out; j++)
		{
			sub_output[0][j] = sample_out[j];
		}
		
		//assign difference of sequences to the active complementary sub-reservoirs
		for(i=1; i<seq.length; i++)
		{
			sample_out  = seq[i-1].getSampleOut(idx_time);
			sample_out2 = seq[i].getSampleOut(idx_time);
			for(j=0; j<num_out; j++)
			{
				sub_output[i][j] = sample_out2[j] - sample_out[j];
			}
		}
		
		//assign the output of inactive sub-reservoirs
		for(i=seq.length; i<num_sub; i++)
		{
			for(j=0; j<num_out; j++)
			{
				sub_output[i][j] = 0;
			}
		}
		
		return sub_output;
	}
	
	/**
	 * The function checks whether there is an increasing error trend for at least one output neuron at the latest
	 * time step.
	 * @param out_deviation: array of deviations of the ESN output from the target over time steps of the error window
	 * @return true: if the error is increasing for at least one ESN's output neuron;
	 *         false: error is decreasing for all ESN's output neurons
	 */
	private boolean isErrorIncreasing(double[][] out_deviation)
	{
		int i;
		boolean is_increasing;/*indicator that the error trend is increasing for at least one output neuron*/

		is_increasing = false;
		for(i=0; i<out_deviation[_window_sub_target_approx-1].length && is_increasing==false; i++)
		{
			//for the increasing trend, the difference between last and one before the last deviations must be positive
			if((Math.abs(out_deviation[_window_sub_target_approx-1][i])  -
				Math.abs(out_deviation[_window_sub_target_approx-2][i])) > 0)
			{
				is_increasing = true;
			}
		}
		
		return is_increasing;
	}
	
	/**
	 * The function calls for creating an object for saving ESN states.
	 *  
	 * @param f_save: indicator whether the ESN states must be saved
	 * @param exp_output: object with the classes responsible for the file output
	 * @param commonPathCurrRun: common incomplete path to save algorithm specific data
	 * @return: object for saving ESN states
	 */
	private esn_state_C prepareSavingEsnState(boolean f_save, ExpOutput exp_output, String commonPathCurrRun)
	{
		int   i;
		int   num_in;//number of the input neurons
		int[] num_res;//number of reservoir neurons for all ESN modules
		String esn_state_file;//path to the file where the ESN states are saved
		esn_state_C esn_state_save;//output variable
		
		//open a file for saving status information
		if(f_save==true)
		{
			if(exp_output==null)
			{
				System.err.println("ESN: no classes are provided for the file output");
				System.exit(1);
				esn_state_save = null;
			}
			else
			{
				//prepare the output object to save the ESN states;
				//since an input layer is created or not created for all modules, it is enough to check it at one module
				num_in = _module[0].getNumNodes(layer_type_E.LT_INPUT);
				if(num_in==0)
				{
					num_in = -1;//"-1" is needed for further operations
				}
				//create an array with reservoir sizes
				num_res = new int[_module.length];
				for(i=0; i<num_res.length; i++)
				{
					num_res[i] = _module[i].getNumNodes(layer_type_E.LT_RES);
				}
				esn_state_file = commonPathCurrRun + "_states.dat";
				esn_state_save = exp_output.prepareSavingEsnState(esn_state_file, true, true, num_in, num_res,
						                                          _module[0].getNumNodes(layer_type_E.LT_OUTPUT));
			}
		}
		else
		{
			esn_state_save = null;
		}
		
		return esn_state_save;
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
	 * The function updates a previously collected range of a target dynamics with information from a new sample.
	 * For each output, the function sets a lower limit to a new value if the new sample is smaller than current limit.
	 * For each output, the function sets a upper limit to a new value if the new sample is larger than current limit.
	 * For the very first sample, the function does only initialization of a target range.
	 * 
	 * @param esn_range: array with lower and upper limits of a target dynamics
	 * @param sample_out: new sample
	 * @param sample_idx: index of a sequence sample
	 */
	private void updateEsnOutRangeFromSample(interval_C[] esn_range, double[] sample_out, int sample_idx)
	{
		int i;
		double lower_lim, upper_lim;//left and right borders of an interval
		
		for(i=0; i<sample_out.length; i++)
		{
			//very 1st sample must be simply assigned without comparison
			if(sample_idx==0)
			{
				esn_range[i].setLeftBorder (sample_out[i]);
				esn_range[i].setRightBorder(sample_out[i]);
			}
			else
			{
				lower_lim = esn_range[i].getLowerLimitAsDouble();
				upper_lim = esn_range[i].getUpperLimitAsDouble();
				//compare to the lower limit
				if(sample_out[i] < lower_lim)
				{
					esn_range[i].setLeftBorder(sample_out[i]);
				}
				//compare to the upper limit
				if(sample_out[i] > upper_lim)
				{
					esn_range[i].setRightBorder(sample_out[i]);
				}
			}
		}
	}
	
	/**
	 * The function completes a valid range of the total network output with margins.
	 * 
	 * @param esn_range_marg: resulting array with lower and upper limits
	 * @param esn_range: array with lower and upper limits of a target dynamics without considering a margin
	 * @param network_margin: safety margin of the network output
	 */
	private void completeRangeWithMargin(interval_C[] esn_range_marg, interval_C[] esn_range, double network_margin)
	{
		int i;
		double range;//range between the lower and upper limits
		double margin;//absolute value of a margin
		double new_lim;//new limit value
		double lower_lim, upper_lim;//left and right borders of an interval
		
		for(i=0; i<esn_range.length; i++)
		{
			lower_lim = esn_range[i].getLowerLimitAsDouble();
			upper_lim = esn_range[i].getUpperLimitAsDouble();

			//compute a margin to be added
			range = upper_lim - lower_lim;
			margin = network_margin * range;

			//compare to the lower limit
			new_lim = lower_lim - margin;
			esn_range_marg[i].setLeftBorder(new_lim);

			//compare to the upper limit
			new_lim = upper_lim + margin;
			esn_range_marg[i].setRightBorder(new_lim);
		}
	}
	
	/**
	 * Do NOT remove this function! There is a note concerning reduction of a valid output range
	 *                              in "configDiffEvolution()" at a former call of this function.
	 * 
	 * The function reduces the current range of mESN's output neurons linearly from a previously available current
	 * range to the current range obtained from a config sequence.
	 * The result is stored in the array of a previously available current range.
	 * 
	 * @param lim_esn_output_cur: previously available valid range
	 * @param lim_esn_output_config: current range obtained from a config sequence
	 * @param idx_step: current time step
	 * @param len_config: length of config sequence
	 */
	private void reduceEsnOutRange(interval_C[] lim_esn_output_cur, interval_C[] lim_esn_output_config,
			                       int idx_step, int len_config)
	{
		int i;
		int rest_len;//number of time steps from the current step to the end of config sequence
		double delta;//delta to reduce a border of the current output range
		double value;//newvalue to assign
		double lower_lim_cur, upper_lim_cur;//left and right borders of the current range
		double lower_lim_config, upper_lim_config;//left and right borders of a range from config sequence
		
		rest_len = len_config - idx_step;
		for(i=0; i<lim_esn_output_cur.length; i++)
		{
			lower_lim_cur = lim_esn_output_cur[i].getLowerLimitAsDouble();
			upper_lim_cur = lim_esn_output_cur[i].getUpperLimitAsDouble();
			lower_lim_config = lim_esn_output_config[i].getLowerLimitAsDouble();
			upper_lim_config = lim_esn_output_config[i].getUpperLimitAsDouble();
			//reduce the upper border
			if(upper_lim_cur > upper_lim_config)
			{
				delta = upper_lim_cur - upper_lim_config;
				delta /= rest_len;
				value = upper_lim_cur - delta;
				lim_esn_output_cur[i].setRightBorder(value);
			}
			else//current range cannot be smaller than a range from a config sequence
			{
				lim_esn_output_cur[i].setRightBorder(upper_lim_config);
			}
			
			//reduce the lower border
			if(lower_lim_cur < lower_lim_config)
			{
				delta = lower_lim_config - lower_lim_cur;
				delta /= rest_len;
				value = lower_lim_cur + delta;
				lim_esn_output_cur[i].setLeftBorder(value);
			}
			else
			{
				lim_esn_output_cur[i].setLeftBorder(lower_lim_config);
			}
		}
	}
	
	/**
	 * The function searches and deactivates modules that have a very small contribution to the whole mESN output.
	 *
	 * @param activity_thresh: activity threshold
	 * @return "true" is a population is deactivated in the current call of the function; "false" - otherwise
	 */
	private boolean setInactiveBySmallContribution(double activity_thresh)
	{
		int i;
		boolean is_inactive;//indicator that a module is inactive
		boolean is_small_contribution;
		boolean is_deactivated;//output variable
		double  avg_error;
		
		is_deactivated = false;
		for(i=0; i<_diff_evolution.length; i++)
		{
			is_inactive = _diff_evolution[i].isInactive();
			if(is_inactive==false)//module is still active
			{
				is_small_contribution = _diff_evolution[i].checkForInactiveModule(activity_thresh);
				if(is_small_contribution==true)
				{
					//the best individual must model a target dynamics precisely in order to disable its population;
					//at the 1st time steps, it can occasionally happen that the best individual has low activity
					//   because its module was suppressed and the other individuals are much worse than it
					avg_error = _diff_evolution[i].getErrorBestMaxAverage();
					if(avg_error < activity_thresh)
					{
						_diff_evolution[i].resetPopulation();
						_module[i].resetConfigured();
						is_deactivated = true;
					}
				}
			}
		}
		
		return is_deactivated;
	}
	
	/**
	 * The function searches and deactivates SIN modules that operate at the same frequency in the counter phase.
	 * The modules are deactivated only when a sum of their activities at t=0 is below a provided activity threshold.
	 * Frequencies of deactivated modules must differ from each other by not more than a provided frequency threshold.
	 * 
	 * ! This function is applicable only to SIN modules because only their behavior is described by a frequency and a phase.
	 *   For other module type, this is only a special case.
	 *
	 * @param activity_thresh: activity threshold
	 * @param freq_thresh: frequency threshold
	 * @return "true" is a population is deactivated in the current call of the function; "false" - otherwise
	 */
	private boolean setInactiveByCounterphase(double activity_thresh, double freq_thresh)
	{
		int i, j, k;
		int idx_best;//index of the best individiual
		double freq_i;//frequency of the 1st module
		double freq_j;//frequency of the 2nd module
		double avg_error_i, avg_error_j;//average errors of the 1st and 2nd individuals 
		double[] sample_in;//input sample
		double[] param_best;//set of module parameters for the best individual
		double[] out_best_i;//module output of the best individual for the 1st module
		double[] out_best_j;//module output of the best individual for the 2nd module
		boolean is_inactive_i;//indicator of activity in the 1st module
		boolean is_inactive_j;//indicator of activity in the 2nd module
		boolean is_counter_phase;//indicator of modules osillating in the counter phase
		boolean is_deactivated;//output variable
		module_type_E module_type_i;//type of the 1st module
		module_type_E module_type_j;//type of the 2nd module
		
		is_deactivated = false;
		
		//set inputs of all SIN modules to t=0
		sample_in = new double[1];
		sample_in[0] = 0;
		for(i=0; i<_module.length; i++)
		{
			module_type_i = _module[i]._module_type;
			if(module_type_i==module_type_E.MT_SIN)
			{
				_module[i].setNodes(sample_in, layer_type_E.LT_INPUT, false);
			}
		}
		
		param_best = null;
		for(i=0; i<_diff_evolution.length-1; i++)
		{
			is_inactive_i = _diff_evolution[i].isInactive();
			module_type_i = _module[i]._module_type;
			if(is_inactive_i==false && module_type_i==module_type_E.MT_SIN)
			{
				//get output of the 1st module at t=0
				idx_best = _diff_evolution[i].getBestIndividual();
				avg_error_i = _diff_evolution[i].getErrorBestMaxAverage();
				param_best = _diff_evolution[i].getIndividualByIndex(idx_best);
				configDiffEvolutionDecodeIndividual(i, param_best);
				out_best_i = _module[i].calculateOutputVector();
				freq_i = ((SinModule)_module[i]).getFrequency();

				for(j=i+1; j<_diff_evolution.length; j++)
				{
					is_inactive_j = _diff_evolution[j].isInactive();
					module_type_j = _module[j]._module_type;
					if(is_inactive_j==false && module_type_j==module_type_E.MT_SIN)
					{
						//get output of the 2nd module at t=0
						idx_best = _diff_evolution[j].getBestIndividual();
						avg_error_j = _diff_evolution[j].getErrorBestMaxAverage();
						param_best = _diff_evolution[j].getIndividualByIndex(idx_best);
						configDiffEvolutionDecodeIndividual(j, param_best);
						out_best_j = _module[j].calculateOutputVector();
						freq_j = ((SinModule)_module[j]).getFrequency();
						
						if(Math.abs(freq_i - freq_j) < freq_thresh)//frequencies are equal
						{
							is_counter_phase = true;
							for(k=0; k<_output.length && is_counter_phase==true; k++)
							{
								if(Math.abs(out_best_i[k] + out_best_j[k]) > activity_thresh)
								{
									is_counter_phase = false;
								}
							}
							
							//deactivate both populations if they operate in the counter-phase
							if(is_counter_phase==true)
							{
								//both best individuals must model a target dynamics precisely in order to disable
								//   their population;
								//(At the 1st time steps, it can occasionally happen that the best individual has a low
								//   activity because its module was suppressed and the other modules are much worse
								//   than it.)
								if(avg_error_i < activity_thresh &&
								   avg_error_j < activity_thresh)
								{
									_diff_evolution[i].resetPopulation();
									_diff_evolution[j].resetPopulation();
									_module[i].resetConfigured();
									_module[j].resetConfigured();
									is_deactivated = true;
								}
							}
						}
					}
				}
			}
		}
		
		return is_deactivated;
	}
	
	/**
	 * The function check outputs of modules and deactivates those modules that are insignificant for the whole mESN output.
	 * 
	 * @param  activity_thresh: threshold for detection of a small contribution of a module
	 * @return "true" is a population is deactivated in the current call of the function; "false" - otherwise
	 */
	private boolean tryToDeactivateModules(double activity_thresh)
	{
		int j;
		boolean is_deactivated_by_contribution, is_deactivated_by_counterphase;
		boolean is_deactivated;//output variable
		
		//search for the best individual is necessary for possible deactivatioon of modules
		for(j=0; j<_diff_evolution.length; j++)
		{
			_diff_evolution[j].findBestVectorIdx();
		}
		//possible deactivation of modules with low output activity
		is_deactivated_by_contribution = setInactiveBySmallContribution(activity_thresh);
		//possible deactivation of SIN modules that operate at the same frequency in the counter phase
		is_deactivated_by_counterphase = setInactiveByCounterphase(activity_thresh, activity_thresh);
		
		if(is_deactivated_by_contribution==true || is_deactivated_by_counterphase==true)
		{
			is_deactivated = true;
		}
		else
		{
			is_deactivated = false;
		}
		
		return is_deactivated;
	}
	
	/**
	 * The function defines valid ranges of the output bias for each output neuron.
	 * The valid range is defined as a small interval at the symmetry center of the corresponding output element.
	 * 
	 * @param seq_config: target output sequence
	 * @return: array of valid ranges for all output elements
	 */
	private interval_C[] InitializeOutBiasRange(seq_C seq_config)
	{
		int i;
		double   middle;//middle of the corresponding output dimension
		double   delta_out_bias;//largest tolerated deviation of the output bias from the middle value
		double[] min_out;//array of the smallest values for all dimensions of the target values
		double[] max_out;//array of the smallest values for all dimensions of the target values
		interval_C[] lim_module_out_bias;
		
		lim_module_out_bias = new interval_C[_output.length];
		min_out = seq_config.getMinOut();
		max_out = seq_config.getMaxOut();
		
		for(i=0; i<_output.length; i++)
		{
			middle = (min_out[i] + max_out[i])/2;
			delta_out_bias = (max_out[i] - min_out[i]) * _delta_out_portion;
			lim_module_out_bias[i] = new interval_C(middle - delta_out_bias,
					                                middle + delta_out_bias);
		}
		
		return lim_module_out_bias;
	}
	
	/**
	 * The function indicates a number of ESN modules.
	 * 
	 * @return: number of ESN modules
	 */
	public int getNumModules()
	{
		return _module.length;
	}
	
	/**
	 * The function indicates a number of output neurons of an mESN.
	 * 
	 * @return number of output neurons
	 */
	public int getNumOutputNeurons()
	{
		return _output.length;
	}
	
	/**
	 * The function returns a value which was used for seeding the random number generator under creation
	 * of the specified ESN module.
	 * 
	 * @param module_idx: index of specified module
	 */
	public int getSeedModule(int module_idx)
	{
		return _module[module_idx].getSeed();
	}
	
	/**
	 * The function indicates whether a spectral radius is non-zero in reservoirs of all ESN modules.
	 * 
	 * @return TRUE if none of the modules has a spectral radius "0"; FALSE - otherwise.
	 */
	public boolean getNonZeroSR()
	{
		int i;
		int num_sub;//number of ESN modules
		boolean is_non_zero_sr;//output variable
		
		num_sub = _module.length;
		is_non_zero_sr = true;
		for(i=0; i<num_sub; i++)
		{
			is_non_zero_sr &= _module[i].getNonZeroSR();
		}
		
		return is_non_zero_sr;
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
	 * The function clears an indicator that an ESN was configured.
	 */
	public void resetConfigured()
	{
		_is_configured = false;
	}
	
	/**
	 * The function indicates whether output weights of a specified module are trained.
	 * 
	 * @return: "true" if all weights are trained; "false" otherwise
	 */
	public boolean isTrainedModule(int module_idx)
	{
		return _module[module_idx].isTrained();
	}
	
	/**
	 * The function sets the reservoir weights,the output weights and the output feedback weights of all active modules
	 * to the values that were obtained either in the training or under the random generation.
	 * The functions assigns 0 to all weights of all inactive modules.
	 * If it is required, the function restores the previously stored states of the ESN.
	 * 
	 * @param active_modules: indicator array which modules must be active 
	 * @param restore_states: "true" - if the ESN states must be restored; "false" - otherwise
	 */
	public void activateModules(boolean[] active_modules, boolean restore_states)
	{
		int i;
		int num_sub;//number of ESN modules
		
		num_sub = _module.length;
		for(i=0; i<num_sub; i++)
		{
			if(active_modules[i]==true)
			{
				_module[i].activate(restore_states);
			}
			else
			{
				_module[i].deactivate();
			}
		}
		_is_configured = true;
	}
	
	/**
	 * The function configures the ESN with a single reservoir by directly teacher-forcing it with the provided
	 * configuration sequence.
	 * The function issues an error message if the ESN contains more than one modules.
	 * @param seq_config: provided configuration sequence
	 * @param time_step: time step corresponding to the very 1st sample of the provided sequence
	 * @param disturbance: object for simulating a disturbance of the OFB
	 * @return: performance statistics of the ESN on the provided sequence
	 */
	public stat_seq_C configDirectTeacherForce(seq_C seq_config, int time_step, MathDistortion disturbance)
	{
		int i,j;
		int idx_sub;//temporary variable: sub-reservoir index is always 0 because method works only for 1 sub-reservoir
		int num_sub;//number of sub-reservoirs
		int size_out;//number of output neurons
		int[] size_res;//number of reservoir neurons in each module
		int[]      dummy_int;//dummy array of integers to call for saving the current state
		double[]   sample_in;//input vector of config sequence at current time step 
		double[]   esn_target;//target ESN output at current time step
		double[]   dummy_array;//dummy array to call for saving the current state
		double[]   out_bias_seq;//constant bias of the provided sequence
		double[][] res_state;//states of reservoir neurons of each ESN module at the current time step
		double[][] out_bias;//bias of output nodes of each module at the current time step
		double[][] responsibility;//responsibility of each module at each output node
		double[][] ofb_sub;//OFB FOR TEACHER-FORCING of the sub-reservoir		
		double[][] var_ofb;//array with variations of the OFB for each sub-reservoir and for each output neuron
		double[][] output_collector;//array with the ESN output on the whole provided configuration sequence
		double[][] sub_output;//temporary array keeping current outputs of all sub-reservoirs
		double[][] sub_target;//target vectors for all sub-reservoirs at current time step
		stat_seq_C stat_seq;//set of statistics of the ESN performance on the given sequence
		Matrix tmp_matrix;//temporary matrix
		
		num_sub  = _module.length;
		size_out = _output.length;
		stat_seq = new stat_seq_C(num_sub);
		dummy_int   = new int[num_sub];
		dummy_array = new double[num_sub];
		sub_target = new double[num_sub][size_out];
		sub_output = new double[num_sub][size_out];
		out_bias   = new double[num_sub][size_out];
		responsibility = new double[num_sub][size_out];
		ofb_sub = new double[num_sub][];
		var_ofb = new double[num_sub][];
		output_collector = new double[seq_config.getSeqLen()][size_out];
		
		//subtract known output bias from the washout sequence
		out_bias_seq = seq_config.getBias();
		seq_config.subtractBias(out_bias_seq);
		
		//assign module specific values
		size_res  = new int[num_sub];
		res_state = new double[num_sub][];
		for(i=0; i<num_sub; i++)
		{
			size_res[i]  = _module[i].getNumNodes(layer_type_E.LT_RES);
			res_state[i] = new double[size_res[i]];
			dummy_int[i] = 0;
			dummy_array[i] = 0;
		}
		
		if(num_sub > 1)
		{
			System.err.println("configDirectTeacherForce: currently the method works only for a single reservoir");
			System.exit(1);
		}
		idx_sub = num_sub - 1;
		
		for(i=0; i<seq_config.getSeqLen(); i++)
		{
			//assign a temporary variable with the sample
			sample_in  = seq_config.getSampleIn(i);
			
			//teacher-forcing is started at time step 1 with a sample at time step 0;
			//This is necessary to save the ESN output with the corresponding target output vector.
			//Thus, the effective length of the configuration sequence is "provided_length - 1".
			if(i > 0)
			{
				ofb_sub[idx_sub] = seq_config.getSampleOut(i-1);
				
				//simulate a disturbance and add it to each element of the OFB
				var_ofb[idx_sub] = disturbance.computeDisturbance(i);
				for(j=0; j<size_out; j++)
				{
					ofb_sub[idx_sub][j] += var_ofb[idx_sub][j];
				}
				
				//assign the current configuration sample to an output of a single module;
				//this is necessary to compute an ESN output in the teacher-forced mode
				_module[0].setNodes(ofb_sub[idx_sub], layer_type_E.LT_OUTPUT, false);
			}
			else
			{
				//assign 0's to all elements of the variation array
				tmp_matrix = new Matrix(num_sub, size_out);
				var_ofb    = tmp_matrix.getArray();
			}

			//assign OFB here in order to ensure that at i=0 this variable does not stay unassigned
			for(j=0; j<num_sub; j++)
			{
				_module[j].getNodes(ofb_sub[j], layer_type_E.LT_OUTPUT);
			}
			
			//compute all modules to realize teacher-forcing;
			//compute an ESN output after teacher-forcing of the output of separate modules
			advance(sample_in, false);
			calculateEsnOutput();
			
			//get outputs of ESN modules after advancing the mESN
			for(j=0; j<num_sub; j++)
			{
				_module[j].getNodes(sub_output[j], layer_type_E.LT_OUTPUT);
			}
			
			//store the output neurons to compute the total performance statistics below
			for(j=0; j<size_out; j++)
			{
				output_collector[i][j] = _output[j];				
			}
			//save the reservoir state at the current sample
			if(_esn_state_save!=null)
			{
				//assign the states of the reservoir neurons for the saving
				for(j=0; j<num_sub; j++)
				{
					_module[j].getNodes(res_state[j], layer_type_E.LT_RES);
					_module[j].getOutputBias(out_bias[j]);
					_module[j].getResponsibility(responsibility[j]);
				}
				//save the target for a single sub-reservoir
				//Actually "seq_config[seq_len-1]" is not used for the teacher-forcing, only for the saving.
				esn_target = seq_config.getSampleOut(i);
				for(j=0; j<size_out; j++)
				{
					sub_target[0][j] = esn_target[j];
				}
				_esn_state_save.saveEsnState(time_step, "config", sample_in, sample_in, res_state, ofb_sub, var_ofb, dummy_array,
                                             dummy_array, dummy_int, sub_output, output_collector[i], sub_target,
                                             esn_target, out_bias, responsibility);
				time_step++;
			}
		}//for i (steps of configuration sequence)
		
		//add output bias before computing the statistics
		seq_config.addBias(out_bias_seq);
		
		//compute network's performance indicators on the most complicated washout sequence
		stat_seq.comp_ident_ok_incl_error       = 0;//assign dummy because implementation is only for evolutionary config
		stat_seq.comp_ident_ok = 0;//assign dummy because implementation is only for evolutionary config
		stat_seq.mse   = MathPerformance.computeMSE(output_collector, seq_config.getSeq());
		stat_seq.nrmse = MathPerformance.computeNRMSE(output_collector, seq_config.getSeq(), seq_config.computeVarianceOut());
		stat_seq.rmse  = MathPerformance.computeRMSE(output_collector, seq_config.getSeq());
		stat_seq.sel   = MathPerformance.computeSEL(output_collector, seq_config.getSeq(), _sel_thresh);
		stat_seq.lel   = MathPerformance.computeLEL(output_collector, seq_config.getSeq(), _lel_thresh);
		
		return stat_seq;
	}
	
	/**
	 * The function configures a modular ESN through a direct search for reservoirs states to reproduce
	 * the configuration sequence. The function gets a set of sequences at the input. The configuration sequence is
	 * the one with the largest index. The other sequence are provided only to save target values of separate
	 * sub-reservoirs in a file.
	 * Currently the method is implemented only for ESNs with a 1D output. The function issues an error message,
	 * if in the optimized ESN the output has many dimensions.
	 *  
	 * @param seq_config: set of sequences
	 * @param time_step: time step corresponding to the very 1st sample of the provided sequence
	 * @param distort: object for simulating a disturbance of the OFB
	 * @param seed: values for seeding random generators for each population
	 * @return: performance statistics of the ESN on the provided sequence
	 */
	public stat_seq_C configDiffEvolution(seq_C[] seq_config, int time_step, MathDistortion distort, DiffEvolutionParam param, int[] seed)
	{
		int   i, j, k;
		int   i_gen;//counter of generations
		int   idx_sub;//temporary variable: module index is always 0 because method works only for 1 module
		int   num_sub;//number of sub-reservoirs
		int   size_out;//number of output neurons
		int   size_in;//number of input neurons
		int   max_idx_seq;//maximum sequence index
		int   idx_pop_best;//index of a population where the best individual is searched
		int   idx_pop_curr;//index of a currently evolved population
		int   num_gen;//number of generations at the current time step
		int   len_config_seq;//length of configuration sequence
		int   len_effective_config;//effective length of the configuration sequence
		int   cnt_active_populations;//counter of the current populations at the current time step
		int[] num_sort;//number of sorted individuals in a population (statistics for analysis)
		int[] size_res;//number of reservoir neurons in each module
		int[] pop_order;//current order of populations to be evolved
		boolean    is_assigned;//indicator that ESN states were assigned from a maturity pool
		boolean    is_inactive;//indicator that the current module was disabled
		boolean    is_active_assigned;//indicator that values of at least one active were assigned 
		boolean    comp_ident_ok;//result of component identification
		boolean    is_deactivated;//population is deactivated at the current time step
		double     tmp_double;//temporary variable of type Double
		double     tmp_margin_i;//temporary variable to store a safety margin of the i-th module
		double     lower_lim, upper_lim;//left and right borders of an interval parameter
		double     network_margin;//margin of the total network's output
		double[]   var_fit;//variance of fitness in each population
		double[]   var_individ;//variance of individuals in each population
		double[][] res_state;//states of reservoir neurons of each ESN module at the current time step
		double[][] responsibility;//responsibility of each module at each output node
		double[][] out_bias;//bias of each output node in each module
		double[][] sample_in;//input sample of the config sequence at current time step
		double[]   input_current;//vector of the discovered input values at the current time step
		double[][] esn_target;//target ESN output at current time step
		double[][] ofb_sub;//OFB FOR TEACHER-FORCING of the sub-reservoir
		double[][] var_ofb;//array with variations of the OFB for each sub-reservoir and for each output neuron
		double[][] output_collector;//array with the ESN output on the whole provided configuration sequence
		double[][] sub_output;//temporary array keeping current outputs of all sub-reservoirs
		double[][] sub_target;//target vectors for all sub-reservoirs at current time step
		stat_seq_C stat_seq;//set of statistics of the ESN performance on the given sequence
		interval_C[][][] lim_module_out;//limits of variation of module output nodes: 1D-module,2D-node,3D-intervals
		interval_C[][][] lim_module_in;//limits of variation of module input nodes: 1D-module,2D-node,3D-intervals
		interval_C[][][] lim_module_res;//limits of variation of module reservoir nodes: 1D-module,2D-node,3D-intervals
		interval_C[][] lim_module_resp;//limits of variation of a module's responsibilities
		interval_C[][] lim_module_out_bias;//limits for variation of the module's output bias
		interval_C[]   lim_esn_output_config;//observed range of a target dynamics for each output neuron;
		                                     //It is needed additionally to "lim_esn_output_config_marg" because it is
		                                     //  necessary to do a comparison of interval and the current sample on
		                                     //  values without a margin. Margins cannot be involved in comparison
		                                     //  because their absolute values are computed depending on a range between
		                                     //  a lower and upper limits. The latter makes no sense (always = 0) for
		                                     //  a single sample which is used for updating a range.
		                                     //  Therefore, first - compare the limits without the margins;then - update
		                                     //  the range using the margins.
		interval_C[]   lim_esn_output_config_marg;//observed range of target dynamics for each output neuron with margin
		seq_parameter_C seq_param;//parameters of a sequence
		
		num_sub  = _module.length;
		size_out = _output.length;
		//"0" because number of modules is the same for all modules
		size_in  = getNumNodesModule(0, layer_type_E.LT_INPUT);
		max_idx_seq = seq_config.length-1;
		len_config_seq = seq_config[max_idx_seq].getSeqLen();
		
		stat_seq = new stat_seq_C(num_sub);
		num_sort    = new int[num_sub];
		var_fit     = new double[num_sub];
		var_individ = new double[num_sub];
		ofb_sub     = new double[num_sub][];
		sub_output  = new double[num_sub][];
		var_ofb     = new double[num_sub][];
		output_collector = new double[len_config_seq][size_out];
		sample_in = new double[param.fitness_length][];
		esn_target = new double[param.fitness_length][];
		lim_module_out  = new interval_C[num_sub][][];
		lim_module_in   = new interval_C[num_sub][][];
		lim_module_res  = new interval_C[num_sub][][];
		lim_module_resp = new interval_C[num_sub][];
		lim_module_out_bias = new interval_C[num_sub][];
		
		input_current = new double[size_in];
		_tmp_responsibility = new double[size_out];//used for evaluation
		_tmp_output_bias    = new double[size_out];//   allocated here in order to avoid permanent memory allocation
		
		//assign module specific values
		size_res  = new int[num_sub];
		res_state = new double[num_sub][];
		out_bias  = new double[num_sub][size_out];
		responsibility = new double[num_sub][size_out];
		for(i=0; i<num_sub; i++)
		{
			size_res[i]  = _module[i].getNumNodes(layer_type_E.LT_RES);
			res_state[i] = new double[size_res[i]];
		}
		
		//assign output-specific values
		lim_esn_output_config = new interval_C[size_out];
		lim_esn_output_config_marg = new interval_C[size_out];
		for(i=0; i<size_out; i++)
		{
			lim_esn_output_config[i] = new interval_C();
			lim_esn_output_config_marg[i] = new interval_C();
		}
		
		//assign input-specific values
		for(i=0; i<size_in; i++)
		{
			input_current[i] = 0;
		}

		//initialize array for all existing modules
		for(i=0; i<num_sub; i++)
		{
			num_sort[i]       = 0;
			var_fit[i]        = 0;
			var_individ[i]    = 0;
			var_ofb[i]        = new double[size_out];
			ofb_sub[i]        = new double[size_out];
			sub_output[i]     = new double[size_out];
			lim_module_out[i]  = _module[i].getOutputValidRange();
			lim_module_res[i]  = _module[i].getInternalValidRange();
			lim_module_in[i]   = _module[i].getInputValidRange();
			lim_module_resp[i] = _module[i].getResponsibilityRange();
			lim_module_out_bias[i] = InitializeOutBiasRange(seq_config[max_idx_seq]);
			tmp_margin_i       = param.margin_module.getElementAsDouble(i);
			//go over output nodes
			for(j=0; j<size_out; j++)
			{
				var_ofb[i][j] = 0;
				
				//go over valid intervals of output nodes
				//(margin is added to each valid interval)
				for(k=0; k<lim_module_out[i][j].length; k++)
				{
					lower_lim = lim_module_out[i][j][k].getLowerLimitAsDouble();
					tmp_double = lower_lim;
					tmp_double-= (tmp_margin_i * Math.abs(lower_lim));
					lim_module_out[i][j][k].setLeftBorder(tmp_double);

					upper_lim = lim_module_out[i][j][k].getUpperLimitAsDouble();
					tmp_double = upper_lim;
					tmp_double+= (tmp_margin_i * Math.abs(upper_lim));
					lim_module_out[i][j][k].setRightBorder(tmp_double);
				}
			}
			//go over hidden nodes
			for(j=0; j<size_res[i]; j++)
			{
				//go over valid intervals of hidden nodes
				//(margin is added to each valid interval)
				for(k=0; k<lim_module_res[i][j].length; k++)
				{
					lower_lim = lim_module_res[i][j][k].getLowerLimitAsDouble();
					tmp_double = lower_lim;
					tmp_double-= (tmp_margin_i * Math.abs(lower_lim));
					lim_module_res[i][j][k].setLeftBorder(tmp_double);

					upper_lim = lim_module_res[i][j][k].getUpperLimitAsDouble();
					tmp_double = upper_lim;
					tmp_double+= (tmp_margin_i * Math.abs(upper_lim));
					lim_module_res[i][j][k].setRightBorder(tmp_double);
				}
			}
			
			//go over input nodes
			for(j=0; j<size_in; j++)
			{
				//go over valid intervals of input nodes
				//(margin is added to each valid interval)
				for(k=0; k<lim_module_in[i][j].length; k++)
				{
					lower_lim = lim_module_in[i][j][k].getLowerLimitAsDouble();
					tmp_double = lower_lim;
					tmp_double-= (tmp_margin_i * Math.abs(lower_lim));
					lim_module_in[i][j][k].setLeftBorder(tmp_double);

					upper_lim = lim_module_in[i][j][k].getUpperLimitAsDouble();
					tmp_double = upper_lim;
					tmp_double+= (tmp_margin_i * Math.abs(upper_lim));
					lim_module_in[i][j][k].setRightBorder(tmp_double);
				}
			}
		}
		//assign margins of the whole network output to the largest margins among available modules
		network_margin = param.margin_module.getElementAsDouble(0);//initialize the margin with that of the 1st module
		for(i=1; i<num_sub; i++)
		{
			tmp_margin_i = param.margin_module.getElementAsDouble(i);
			if(network_margin < tmp_margin_i)
			{
				network_margin = tmp_margin_i;
			}
		}
		
		//assign objects of all active modules
		_diff_evolution = new DiffEvolution[num_sub];
		_diff_evolution_mature = new DiffEvolutionMature(this, param.fitness_length, param.validation_len, param.ciok_level);
		for(i=0; i<num_sub; i++)
		{
			_module[i].createNodesHistory(param.fitness_length);
			_diff_evolution[i] = new DiffEvolution(this, param.F, param.CR, param.pop_size, param.sort_min_num,
					                               lim_module_res[i], lim_module_out[i], lim_module_out_bias[i],
					                               lim_module_resp[i], lim_module_in[i], i, param.fitness_length,
					                               seed[i]);
			_diff_evolution[i].SetSuppressProbability(false, -1);
		}
		
		//save pre-synchronization ESN states
		//advance the ESN on the first "fitness_length - 1" time steps to get performance statistics for saving
		//before evolution will be started
		for(j=0; j<param.fitness_length-1; j++)
		{
			//assign the current sample always to the 1st element
			sample_in[j]  = seq_config[max_idx_seq].getSampleIn(j);
			esn_target[j] = seq_config[max_idx_seq].getSampleOut(j);
			
			//update output range on the current sample
			updateEsnOutRangeFromSample(lim_esn_output_config, esn_target[j], j);
			completeRangeWithMargin(lim_esn_output_config_marg, lim_esn_output_config, network_margin);
			
			//OFB of modules are their outputs at the previous time step
			for(k=0; k<num_sub; k++)
			{
				_module[k].getNodes(ofb_sub[k], layer_type_E.LT_OUTPUT);
			}
			
			//advance an ESN by one time step
			//"sample_in = null" because the input vector is a product of synchronization
			advance(null, false);
			calculateEsnOutput();
			
			//get computed outputs of modules
			for(k=0; k<num_sub; k++)
			{
				_module[k].getNodes(sub_output[k], layer_type_E.LT_OUTPUT);
			}
			
			//store the output neurons to compute the total performance statistics below
			for(k=0; k<size_out; k++)
			{
				//store the ESN output
				output_collector[j][k] = _output[k];				
			}
			//save the reservoir state at the current sample
			if(_esn_state_save!=null)
			{
				//assign module-specific states for saving
				for(k=0; k<num_sub; k++)
				{
					_module[k].getNodes(res_state[k], layer_type_E.LT_RES);
					_module[k].getOutputBias(out_bias[k]);
					_module[k].getResponsibility(responsibility[k]);
				}
				//save the targets of separate sub-reservoirs
				sub_target = computeTargetSub(seq_config, j);
				//"sample_in[]" and "esn_target[]" were assigned in the previous loop
				_esn_state_save.saveEsnState(time_step, "config", input_current, sample_in[j], res_state, ofb_sub, var_ofb,
						                     var_fit, var_individ, num_sort, sub_output, output_collector[j],
						                     sub_target, esn_target[j], out_bias, responsibility);
				time_step++;
			}
		}//for j
		
		//go along a configuration sequence;
		//the 1st time step corresponds to the last step of the 1st possible window
		len_effective_config = len_config_seq - (param.fitness_length - 1);
		for(i=param.fitness_length-1; i<len_config_seq; i++)
		{
			//assign a sample of the next time step
			sample_in[param.fitness_length-1]  = seq_config[max_idx_seq].getSampleIn(i);
			esn_target[param.fitness_length-1] = seq_config[max_idx_seq].getSampleOut(i);
			
			//update output range on the current sample
			updateEsnOutRangeFromSample(lim_esn_output_config, esn_target[param.fitness_length-1], i);
			completeRangeWithMargin(lim_esn_output_config_marg, lim_esn_output_config, network_margin);
			//Note: Reduction of valid output range in "reduceEsnOutRange()" is HARMFUL because it removes a positive
			//      effect from safety margins.
			//      Safety margins are used to compute an initial valid output range (at the beginning of a config
			//      sequence). At following time steps, a valid range is permanently reduced to approach target values
			//      that come from the config sequence. A a result, a valid output range can be precisely reduced
			//      towards an interval of target values which was observed on a config sequence. This interval does not
			//      contain a safety margin anymore. (That is, the safety margin was "reduced away".) An output of any
			//      EPOS (even a very precise EPOS) can have small fluctuations. When they happen without a safety
			//      margin, a very precise combination of modules can be "sorted away". As an example, it was observed
			//      that a very precise combination of individuals (configMSE = 1E-19) was assigned for sorting because
			//      the EPOS produced a deviation which was beyond a valid limit in the 10th digit after comma (!).
			//reduceEsnOutRange(lim_esn_output_cur, lim_esn_output_config, i, len_config_seq);
			for(j=0; j<_diff_evolution.length; j++)
			{
				//maximum magnitude must be updated with marginized values; otherwise generation of responsibilities
				//   will be done much more often than generation of internal states in the beginning of configuration
				//   sequence
				_diff_evolution[j].setMaxTargetMagnitude(lim_esn_output_config_marg);
				_diff_evolution[j].setRangeEsnOutput(lim_esn_output_config_marg);
			}
			
			//assign OFB here in order to ensure that at i=0 this variable does not stay unassigned
			//TODO: it should be checked whether it corresponds to time step "i - (fit_length-1)"
			for(j=0; j<num_sub; j++)
			{
				_module[j].getNodes(ofb_sub[j], layer_type_E.LT_OUTPUT);
			}
			
			//advance an already assigned ESN
			//"sample_in = null" because the input vector is a product of synchronization
			advance(null, false);
			//backup its states to restore them if no best ESN is found after evolution in the current time step
			for(j=0; j<num_sub; j++)
			{
				storeModuleNodes(j, storage_type_E.ST_TMP);
			}
			
			//teacher-forcing is started at time step 1 with a sample at time step 0;
			//1st time step is "i - (param.fitness_length-1)";
			//This is necessary to save the ESN output with the corresponding target output vector.
			//Thus, the effective length of the configuration sequence is "provided_length - 1".
			if(i - (param.fitness_length-1) > 0)
			{
				if(distort!=null)
				{
					//simulate a disturbance and add it to each element of the OFB;
					//"i - (param.fitness_length-1)" is the 1st time step
					for(idx_sub=0; idx_sub<num_sub; idx_sub++)
					{
						var_ofb[idx_sub] = distort.computeDisturbance(i - (param.fitness_length-1));
						for(j=0; j<size_out; j++)
						{
							ofb_sub[idx_sub][j] += var_ofb[idx_sub][j];
						}

						//assign a distorted value to the module output
						_module[idx_sub].setNodes(ofb_sub[idx_sub], layer_type_E.LT_OUTPUT, false);
					}
				}//OFB distortion is required
				
				for(j=0; j<num_sub; j++)
				{
					//population is advanced only for the 1st step; the rest steps are used only for fitness computation
					_diff_evolution[j].advancePopulation(sample_in[param.fitness_length-1]);
				}
				_diff_evolution_mature.advance(sample_in[param.fitness_length-1], esn_target[param.fitness_length-1]);
			}
			else
			{
				for(j=0; j<num_sub; j++)
				{
					_diff_evolution[j].initializePopulation(param.init_mode, sample_in);
				}
			}

			//update errors of individuals and statistics before evolution
			pop_order = param.getPopOrder(num_sub-1);
			for(j=0; j<pop_order.length; j++)
			{
				idx_pop_curr = pop_order[j];
				//following error update is needed to compute an up-to-date statistics in "computeErrVariance()" below
				_diff_evolution[idx_pop_curr].updateErrorAll(sample_in, esn_target);
				var_fit[idx_pop_curr]     = _diff_evolution[idx_pop_curr].computeErrVariance();
				var_individ[idx_pop_curr] = _diff_evolution[idx_pop_curr].computeIndividVariance();
				
				num_sort[idx_pop_curr] = 0;
			}
			
			is_deactivated = tryToDeactivateModules(param.activ_thresh);
			if(is_deactivated==true)
			{
				len_effective_config = len_config_seq - i;
				
				//re-initialize all active populations
				//for(j=0; j<num_sub; j++)
				//{
				//	is_inactive = _diff_evolution[j].isInactive();
				//	if(is_inactive==false)
				//	{
				//		_diff_evolution[j].initializePopulation(config_ea_init_E.CFG_EA_INIT_RANDOM_RES_STATES_INPUT, sample_in);
				//	}
				//}
				
				//update errors of individuals and statistics before evolution
				//pop_order = param.getPopOrder(num_sub-1);
				//for(j=0; j<pop_order.length; j++)
				//{
				//	idx_pop_curr = pop_order[j];
				//	is_inactive = _diff_evolution[idx_pop_curr].isInactive();
				//	if(is_inactive==false)
				//	{
				//		//following error update is needed to compute an up-to-date statistics in "computeErrVariance()" below
				//		_diff_evolution[idx_pop_curr].updateErrorAll(sample_in, esn_target);
				//		var_fit[idx_pop_curr]     = _diff_evolution[idx_pop_curr].computeErrVariance();
				//		var_individ[idx_pop_curr] = _diff_evolution[idx_pop_curr].computeIndividVariance();
				//	}
				//}
			}
			
			//count active populations
			cnt_active_populations = 0;
			for(j=0; j<num_sub; j++)
			{
				is_inactive = _diff_evolution[j].isInactive();
				if(is_inactive==false)
				{
					cnt_active_populations++;
				}
			}
			
			//choose a probability of suppression depending on a number of active populations;
			//
			if(cnt_active_populations==1)
			{
				for(j=0; j<num_sub; j++)
				{
					is_inactive = _diff_evolution[j].isInactive();
					if(is_inactive==false)
					{
						_diff_evolution[j].SetSuppressProbability(true, len_effective_config);
					}
				}
			}
			else
			{
				//do nothing;
				//this IF branch supposes initialization of probability of suppression;
				//however, it has already been done before evolution (directly after allocation of DiffEvolution)
			}
			
			/*"len_config_seq - param.fitness_length" is a number of steps of a fitness window along a configuration
			  sequence; the fitness window effectively reduces a length of the configuration sequence*/
			num_gen = param.computeNumGen(i, len_config_seq, len_effective_config);
			
			//perform evolution and sorting in parallel
			for(i_gen=0; i_gen<num_gen; i_gen++)
			{
				for(j=0; j<pop_order.length; j++)
				{
					idx_pop_curr = pop_order[j];
					
					is_inactive = _diff_evolution[idx_pop_curr].isInactive();
					if(is_inactive==false)
					{
						//This error update is necessary because after a previous evolution linkages between individuals
						//   could be changed.
						_diff_evolution[idx_pop_curr].updateErrorAll(sample_in, esn_target);

						_diff_evolution[idx_pop_curr].run(sample_in, esn_target, param.activ_thresh);
					}

					//store the best individual after the last generation of the very last population;
					//storing is done BEFORE a sorting in order not to disrupt linkages that were fine-tuned
					//   in preceding evolution
					if(j==pop_order.length-1 && i_gen==num_gen-1)
					{
						//best population is always the last one
						idx_pop_best = pop_order[pop_order.length-1];

						//update attribute "is_increasing of new individuals"
						_diff_evolution[idx_pop_best].updateErrorAll(sample_in, esn_target);

						//TODO: check if this additional search for the best individual is really necessary
						_diff_evolution[idx_pop_best].findBestVectorIdx();

						//store individuals of the best individuals
						configDiffEvolutionStoreBest(idx_pop_best, i, sample_in);
					}

					num_sort[idx_pop_curr] += _diff_evolution[idx_pop_curr].sortPopulation();
				}//for populations
			}//for "i_gen"
			
			//restore states of the best individuals for the case if current configuration time step is the last one
			is_assigned = _diff_evolution_mature.restoreBestIndividual(i);
			if(is_assigned==false)//restore states of previously assigned ESN if no new states are assigned
			{
				for(j=0; j<num_sub; j++)
				{
					restoreModuleNodes(j, storage_type_E.ST_TMP);
				}
				calculateEsnOutput();
			}
			
			//get computed values of the sub-reservoirs
			for(k=0; k<num_sub; k++)
			{
				_module[k].getNodes(sub_output[k], layer_type_E.LT_OUTPUT);
			}
			
			//store the output neurons to compute the total performance statistics below
			for(j=0; j<size_out; j++)
			{
				//store the ESN output only for the 1st sample; therefore "i-(param.fitness_length-1)"
				output_collector[i][j] = _output[j];
			}
			//save the reservoir state at the current sample
			if(_esn_state_save!=null)
			{
				//assign the states of the reservoir neurons for the saving
				is_active_assigned = false;
				for(j=0; j<num_sub; j++)
				{
					_module[j].getNodes(res_state[j], layer_type_E.LT_RES);
					_module[j].getOutputBias(out_bias[j]);
					_module[j].getResponsibility(responsibility[j]);
					
					//extract input values from only from an active module;
					//reset the previously assigned values if there is more than one active module
					if(_module[j].getConfigured()==true)
					{
						if(is_active_assigned==false)//1st active module is detected
						{
							_module[j].getNodes(input_current, layer_type_E.LT_INPUT);
							is_active_assigned = true;
						}
						else//more than one active module is detected
						{
							for(k=0; k<size_in; k++)
							{
								input_current[k] = 0;
							}
						}
					}
				}
				sub_target = computeTargetSub(seq_config, i);				
				_esn_state_save.saveEsnState(time_step, "config", input_current, sample_in[param.fitness_length-1], res_state, ofb_sub,
						                     var_ofb, var_fit, var_individ, num_sort, sub_output, output_collector[i],
						                     sub_target, esn_target[param.fitness_length-1], out_bias, responsibility);
				time_step++;
			}
			
			//move a fitness window
			for(j=0; j<param.fitness_length-1; j++)
			{
				sample_in[j]  = sample_in[j+1];
				esn_target[j] = esn_target[j+1];
			}
			
		}//for i (steps of configuration sequence)
		
		//extract sequence parameters of active modules
		seq_param = new seq_parameter_C();
		for(i=0; i<_diff_evolution.length; i++)
		{
			if(_diff_evolution[i].isInactive()==false)
			{
				seq_param.addOscillatorParam(_module[i].getSeqParam());
			}
		}
		//compare parameters of config sequence and parameters of active modules
		comp_ident_ok = seq_config[max_idx_seq].getSeqParam().compareSeqParameter(seq_param);
		
		//*** compute performance indicators of the best mESN on a provided config sequence
		
		stat_seq.mse = _diff_evolution_mature.getTotalMse();
		
		if(comp_ident_ok==true)
		{
			stat_seq.comp_ident_ok = 1;
			
			//result of component identification depends of the reached error 
			if(stat_seq.mse <= param.ciok_level)
			{
				stat_seq.comp_ident_ok_incl_error = 1;
			}
			else
			{
				stat_seq.comp_ident_ok_incl_error = 0;
			}
		}
		else
		{
			stat_seq.comp_ident_ok = 0;
			stat_seq.comp_ident_ok_incl_error       = 0;
		}
		
		stat_seq.nrmse = _diff_evolution_mature.getTotalNrmse(seq_config[max_idx_seq].computeVarianceOut());
		stat_seq.rmse  = _diff_evolution_mature.getTotalRmse();
		stat_seq.sel   = _diff_evolution_mature.getLifeTimeBest();
		stat_seq.lel   = _diff_evolution_mature.getTimeBest();
		
		return stat_seq;
	}

	/**
	 * The function computes an average error for a specified individual.
	 * The individual is specified by an index in a population and by an index of its ESN module.
	 * If the provided module index is "-1" then, for computing the average error, a history of the individual from its
	 * population is taken.
	 * Otherwise, the average error is computed using data stored in a history of its module.
	 * In both cases, data of related individuals of the other population are taken from their history of their
	 * populations.
	 * 
	 * @param sub_idx: index of a module of the provided individual
	 * @param error: object to store an error of the current individual for the output
	 * @param idx_individual: index of evaluated individual
	 * @param sample_out: target output values for all time steps of a fitness window
	 * @param lim_esn_output: valid ranges of all output neurons of the whole mESN
	 */
	public void configDiffEvolutionEvaluate(int sub_idx, DiffEvolutionError error, int idx_individual,
			                                double[][] sample_out, interval_C[] lim_esn_output)
	{
		int i;
		int idx_out;//index of an output neuron
		int idx_sample;//index of currently considered sample
		int num_sample;//number of samples to be considered
		int num_sub_active;//number of active sub-reservoirs
		double[] sub_output;//ESN outputs of an individual with a given index in the current population
		boolean esn_out_invalid;//invalid ESN output
		
		//assign states of the ESN modules
		num_sub_active  = _diff_evolution.length;
		num_sample = sample_out.length;
		sub_output = new double[sample_out[0].length];
		esn_out_invalid = false;
		error.clean();
		
		//assign responsibilities for each module that are encoded by individuals of index "idx_individual"
		//(only if it is requested to get responsibilities from the population)
		if(sub_idx==-1)
		{
			for(i=0; i<num_sub_active; i++)
			{
				_diff_evolution[i].getOutputBiasByIndex(idx_individual, _tmp_output_bias);
				_module[i].setOutputBias(_tmp_output_bias);
				_diff_evolution[i].getResponsibilityByIndex(idx_individual, _tmp_responsibility);
				_module[i].setResponsibility(_tmp_responsibility);
			}
		}
		
		//go over a fitness window
		for(idx_sample=0; idx_sample<num_sample; idx_sample++)
		{
			for(i=0; i<num_sub_active; i++)
			{
				//it is not necessary to check against "-1" explicitly because "i" is never "-1"
				if(i==sub_idx)
				{
					_module[i].getNodesHistory(idx_sample, layer_type_E.LT_OUTPUT, sub_output);
				}
				else
				{
					_diff_evolution[i].getOutputByIndex(idx_individual, idx_sample, sub_output);
				}
				//set retrieved values at the output of an ESN module
				setModuleNodes(i, layer_type_E.LT_OUTPUT, sub_output);
			}

			//COMPUTE an error
			calculateEsnOutput();
			error.update( computeError(sample_out[idx_sample]) );
			//check whether an mESN output is in a valid range
			for(idx_out=0; idx_out<_output.length; idx_out++)
			{
				if(_output[idx_out] < lim_esn_output[idx_out].getLowerLimitAsDouble() ||
				   _output[idx_out] > lim_esn_output[idx_out].getUpperLimitAsDouble())
				{
					esn_out_invalid = true;
				}
			}
		}
		
		//set an error trend to "increasing" if at least one output was out-of-range
		if(esn_out_invalid==true)
		{
			error.is_increase = true;
		}
	}
	
	/**
	 * The function stores individuals of the best combination in a maturity pool.
	 * The individuals are advanced before their storing in the maturity pool.
	 * The individuals are advanced on provided input samples. A number of the advancing steps must match a number of
	 * errors in the error status.
	 * 
	 * @param idx_pop_best: index of a population to get an index of the best individual
	 * @param time_step: time step for which the individuals are stored
	 * @param sample_in: array of input samples to perform advancing to the last step of a sliding window
	 */
	public void configDiffEvolutionStoreBest(int idx_pop_best, int time_step, double[][] sample_in)
	{
		int i;
		int idx_best;//index of the best individual
		int num_populations;
		int num_out;//number of output neurons
		double[][] individual;
		double[][] sub_output;
		double[][][] sub_output_history;//array of outputs of all modules on the whole fitness window
		                                //1D: index of module
		                                //2D: index of time step
		                                //3D: index of the module's output
		DiffEvolutionError error_status;//full error status obtained on a configuration sequence
		
		idx_best = _diff_evolution[idx_pop_best].getBestIndividual();
		if(idx_best!=-1)
		{
			//retrieve an error status
			error_status = _diff_evolution[idx_pop_best].getErrorFull(idx_best);
			
			//check input data
			if(sample_in.length != error_status.cnt_error)
			{
				System.err.println("configDiffEvolutionStoreBest: mismatch between input sample and sliding window");
				System.exit(1);
			}
			
			//get individuals to be assigned to the modules
			num_populations = _diff_evolution.length;
			num_out = _output.length;
			individual = new double[num_populations][];
			sub_output = new double[num_populations][];
			sub_output_history = new double[num_populations][][];
			for(i=0; i<num_populations; i++)
			{
				sub_output[i] = new double[num_out];
				individual[i] = _diff_evolution[i].getEndIndividualByIndex(idx_best, sub_output[i]);
				sub_output_history[i] = _diff_evolution[i]._sub_output_history[idx_best].getOrderedArray();
			}

			_diff_evolution_mature.storeBestIndividual(individual, sub_output, error_status, time_step, sub_output_history);
		}
	}
	
	/**
	 * The function enables the sorting of all individuals with a provided index.
	 * 
	 * @param idx: provided index of individuals
	 */
	public void configDiffEvolutionEnableSort(int idx)
	{
		int i;
		int num_pop;
		
		num_pop = _diff_evolution.length;
		for(i=0; i<num_pop; i++)
		{
			_diff_evolution[i].enableSort(idx);
		}
	}
	
	/**
	 * The function disables the sorting of all individuals with a provided index.
	 * 
	 * @param idx: provided index of individuals
	 */
	public void configDiffEvolutionDisableSort(int idx)
	{
		int i;
		int num_pop;
		
		num_pop = _diff_evolution.length;
		for(i=0; i<num_pop; i++)
		{
			_diff_evolution[i].disableSort(idx);
		}
	}
	
	/**
	 * The function exchanges individuals at two specified indices in all populations.
	 * 
	 * @param idx_0: index of the 1st individual
	 * @param idx_1: index of the 2nd individual
	 */
	public void configDiffEvolutionExchangeIndividuals(int idx_0, int idx_1)
	{
		int i;
		int num_pop;
		
		num_pop = _diff_evolution.length;
		for(i=0; i<num_pop; i++)
		{
			_diff_evolution[i].exchangeIndividuals(idx_0, idx_1);
		}
	}
	
	/**
	 * The function splits a provided individual into composite parts and assigns them to attributes of a module.
	 * 
	 * @param sub_idx: module index
	 * @param values: provided individual
	 */
	public void configDiffEvolutionDecodeIndividual(int sub_idx, double[] values)
	{
		int      i;
		int      idx_element;//index of the current genotype's element
		int      num_out;//size of an output array
		int      num_in;//size of the input vector
		double[] attrib_values;
		
		num_out = _module[sub_idx].getNumNodes(layer_type_E.LT_OUTPUT);
		num_in  = _module[sub_idx].getNumNodes(layer_type_E.LT_INPUT);
		
		//assign an output bias to the module
		attrib_values = new double[num_out];
		idx_element = 0;
		for(i=0; i<num_out; i++)
		{
			attrib_values[i] = values[idx_element];
			idx_element++;
		}
		setModuleOutputBias(sub_idx, attrib_values);
		
		//assign responsibilities to a module
		attrib_values = new double[num_out];
		for(i=0; i<num_out; i++)
		{
			attrib_values[i] = values[idx_element];
			idx_element++;
		}
		setModuleResponsibility(sub_idx, attrib_values);
		
		//assign states to a module
		attrib_values = new double[values.length - 2*num_out - num_in];
		for(i=0; i<attrib_values.length; i++)
		{
			attrib_values[i] = values[idx_element];
			idx_element++;
		}
		setModuleNodes(sub_idx, layer_type_E.LT_RES, attrib_values);
		
		//assign the rotation angle PHI as the input value
		attrib_values = new double[num_in];
		for(i=0; i<attrib_values.length; i++)
		{
			attrib_values[i] = values[idx_element];
			idx_element++;
		}
		setModuleNodes(sub_idx, layer_type_E.LT_INPUT, attrib_values);
	}
	
	/**
	 * The function extracts relevant attributes of a module and stores them as an individual in a provided array.
	 * 
	 * @param sub_idx: module index
	 * @param values: provided array to store an individual
	 */
	public void configDiffEvolutionEncodeIndividual(int sub_idx, double[] values)
	{
		int      i;
		int      idx_element;//index of the current genotype's element
		int      num_out;//size of an output array
		int      num_in;//size of the input vector
		double[] attrib_values;
		
		num_out = _module[sub_idx].getNumNodes(layer_type_E.LT_OUTPUT);
		num_in  = _module[sub_idx].getNumNodes(layer_type_E.LT_INPUT);
		
		//get the output bias from the module
		attrib_values = new double[num_out];
		getModuleOutputBias(sub_idx, attrib_values);
		idx_element = 0;
		for(i=0; i<num_out; i++)
		{
			values[idx_element] = attrib_values[i];
			idx_element++;
		}
		//setModuleOutputBias(sub_idx, attrib_values);
		
		//get responsibilities from a module
		attrib_values = new double[num_out];
		getModuleResponsibility(sub_idx, attrib_values);
		for(i=0; i<num_out; i++)
		{
			values[idx_element] = attrib_values[i];
			idx_element++;
		}
		//setModuleResponsibility(sub_idx, attrib_values);
		
		//get reservoirs states from a module
		attrib_values = new double[values.length - 2*num_out - num_in];
		getModuleNodes(sub_idx, layer_type_E.LT_RES, attrib_values);
		for(i=0; i<attrib_values.length; i++)
		{
			values[idx_element] = attrib_values[i];
			idx_element++;
		}
		//setModuleNodes(sub_idx, layer_type_E.LT_RES, attrib_values);
		
		//get the input vector from the module
		attrib_values = new double[num_in];
		getModuleNodes(sub_idx, layer_type_E.LT_INPUT, attrib_values);
		for(i=0; i<attrib_values.length; i++)
		{
			values[idx_element] = attrib_values[i];
			idx_element++;
		}
	}
	
	/**
	 * In this function an echo-state network is run on a test sequence without teacher-forcing. The network
	 * output is collected in an array and saved into a file, if it is required.
	 * 
	 * @param seq_label: label of sequence ("test, "freerun" etc.) to be saved in the file at corresponding time steps
	 * @param test: object with test sequence
	 * @param max_sub_idx: maximum sub-reservoir index
	 * @param seq_target_sub: (only for saving) array of targets sequences for all active sub-reservoirs
	 * @param time_step: time step corresponding to the first sequence sample (for saving target and actual states)
	 * @param is_saving: is saving required for given sequence
	 * @param ofb_distort: object for simulating OFB distortion
	 * @param ciok_level: largest allowed MSE for successful component identification
	 * @return: object containing all performance statistics on the given sequence
	 */
	public stat_seq_C runFree(String seq_label, seq_C seq, seq_C[] seq_target_sub, int time_step,
			                  boolean is_saving, MathDistortion ofb_distort, double ciok_level)
	{
		int   i, j, k;
		int   size_in;//number of input neurons
		int   size_out;//number of output neurons
		int   num_sub;//number of all sub-reservoirs
		int   idx_sub;//temporary variable: sub-reservoir index is always 0 because method works only for 1 sub-reservoir
		int[] size_res;//number of reservoir neurons in each module
		int[] dummy_int;//dummy array of integers to call for saving the current state
		double[] sample_in;//temporary array keeping the current input vector 
		double[] sample_out;//temporary array keeping the current output vector		
		double[]   dummy_array;//dummy array to call for saving the current state
		double[]   input_current;//vector of the discovered input values at the current time step
		double[][] res_state;//states of reservoir neurons of each ESN module at the current time step
		double[][] responsibility;//responsibility of each module at each output node
		double[][] out_bias;//bias of output nodes in each module
		double[][] ofb_sub;//OFB of each sub-reservoir from each output neuron
		double[][] var_ofb;//variation of the OFB of each sub-reservoir from each output neuron
		double[][] output_collector;
		double[][] module_output;//for saving the ESN state: output of each module for each output neuron
		double[][] target_sub;//for saving the ESN state: target of each sub-reservoir for each output neuron
		boolean    comp_ident_ok;//result of component identification
		boolean    is_active_assigned;//indicator that values of at least one active were assigned
		stat_seq_C stat_seq;//set of statistics of the ESN performance on the given sequence
		Matrix tmp_matrix;//temporary matrix
		seq_parameter_C seq_param;//parameters of a sequence
		
		size_out = _module[0].getNumNodes(layer_type_E.LT_OUTPUT);//all modules have the same number of output neurons
		num_sub  = _module.length;
		output_collector = new double[seq.getSeqLen()][size_out];
		dummy_int   = new int[num_sub];
		dummy_array = new double[num_sub];
		target_sub = new double[num_sub][size_out];
		module_output = new double[num_sub][];
		stat_seq = new stat_seq_C(0);
		
		//assign 0's to all elements of the variation array
		//   because no variation of the OFB is considered in the current function
		tmp_matrix = new Matrix(num_sub, size_out);
		var_ofb    = tmp_matrix.getArray();
		
		//assign module specific values
		size_res  = new int[num_sub];
		res_state = new double[num_sub][];
		out_bias  = new double[num_sub][size_out];
		responsibility  = new double[num_sub][size_out];
		ofb_sub   = new double[num_sub][];
		for(i=0; i<num_sub; i++)
		{
			size_res[i]      = _module[i].getNumNodes(layer_type_E.LT_RES);
			res_state[i]     = new double[size_res[i]];
			module_output[i] = new double[size_out];
			ofb_sub[i]       = new double[size_out];
		}
		
		//initialize a dummy array
		for(i=0; i<num_sub; i++)
		{
			dummy_int[i] = 0;
			dummy_array[i] = 0;
		}
		
		//assign number of input neurons;
		//since an input layer is created or not created for all modules, it is enough to check it at one module
		size_in = _module[0].getNumNodes(layer_type_E.LT_INPUT);
		
		//assign states of reservoir neurons for the file output
		input_current = new double[size_in];
		is_active_assigned = false;
		for(j=0; j<num_sub; j++)
		{
			//extract input values from only from an active module;
			//reset the previously assigned values if there is more than one active module
			if(_module[j].getConfigured()==true)
			{
				if(is_active_assigned==false)//1st active module is detected
				{
					_module[j].getNodes(input_current, layer_type_E.LT_INPUT);
					is_active_assigned = true;
				}
				else//more than one active module is detected
				{
					for(i=0; i<size_in; i++)
					{
						input_current[i] = 0;
					}
				}
			}
		}
		
	
		//main-loop for the initial and the training-phase
		for(i=0; i < seq.getSeqLen(); i++)
		{
			//prepare a vector with input values, if input neurons exist
			if(size_in!=0)
			{
				sample_in = seq.getSampleIn(i);
			}
			else
			{
				sample_in = null;
			}
			//assign the target output (output neurons are always present in the ESN)
			sample_out = seq.getSampleOut(i);
			
			//assign current outputs of ESN modules whose values will be used as OFB in the next ESN computation
			for(j=0; j<num_sub; j++)
			{
				_module[j].getNodes(ofb_sub[j], layer_type_E.LT_OUTPUT);
			}
				
			//simulate OFB distortion, if it is necessary
			if(ofb_distort!=null)
			{
				for(idx_sub=0; idx_sub<num_sub; idx_sub++)
				{
					var_ofb[idx_sub] = ofb_distort.computeDisturbance(i);
					for(j=0; j<size_out; j++)
					{
						ofb_sub[idx_sub][j] += var_ofb[idx_sub][j];
					}
					//assign outputs of separate modules after distorting them
					_module[idx_sub].setNodes(ofb_sub[idx_sub], layer_type_E.LT_OUTPUT, false);
				}
			}
			
			//compute states of internal neurons;
			//compute an ESN output in the current step on a training sequence
			//"sample_in = null" because the input vector is a product of synchronization
			advance(null, false);
			calculateEsnOutput();
			
			for(j=0; j < size_out; j++)
			{
				output_collector[i][j] = _output[j];				
			}
			//save the reservoir state at the current sample only when the saving is required for given sequence
			if(_esn_state_save!=null && is_saving==true)
			{
				//initialization only at the very 1st sample
				if(i==0)
				{					
					for(j=0; j<num_sub; j++)
					{
						for(k=0; k<size_out; k++)
						{
							target_sub[j][k] = 0;
						}
					}
				}
				//assign target sub-reservoir values for the file output
				for(j=0; j<seq_target_sub.length; j++)
				{
					target_sub[j] = seq_target_sub[j].getSample(i)._out;
				}
				//assign states of reservoir neurons for the file output
				for(j=0; j<num_sub; j++)
				{
					//assign the states of the reservoir neurons for saving
					_module[j].getNodes(res_state[j], layer_type_E.LT_RES);
					_module[j].getOutputBias(out_bias[j]);
					_module[j].getResponsibility(responsibility[j]);
					//assign outputs of ESN modules for saving
					_module[j].getNodes(module_output[j], layer_type_E.LT_OUTPUT);
				}
				_esn_state_save.saveEsnState(time_step,seq_label, input_current, sample_in, res_state, ofb_sub, var_ofb,
						                     dummy_array, dummy_array, dummy_int, module_output, output_collector[i],
						                     target_sub, sample_out, out_bias, responsibility);
			}
			time_step++;
		}//for i
		
		//extract sequence parameters of active modules
		seq_param = new seq_parameter_C();
		for(i=0; i<_diff_evolution.length; i++)
		{
			if(_module[i].getConfigured()==true)
			{
				seq_param.addOscillatorParam(_module[i].getSeqParam());
			}
		}
		//compare parameters of config sequence and parameters of active modules
		comp_ident_ok = seq.getSeqParam().compareSeqParameter(seq_param);
		
		stat_seq.mse   = MathPerformance.computeMSE(output_collector, seq.getSeq());
		
		if(comp_ident_ok==true)
		{
			stat_seq.comp_ident_ok = 1;
			
			//result of component identification depends of the reached error 
			if(stat_seq.mse <= ciok_level)
			{
				stat_seq.comp_ident_ok_incl_error = 1;
			}
			else
			{
				stat_seq.comp_ident_ok_incl_error = 0;
			}
		}
		else
		{
			stat_seq.comp_ident_ok = 0;
			stat_seq.comp_ident_ok_incl_error = 0;
		}

		//compute network's performance indicators on the training sequence
		stat_seq.nrmse = MathPerformance.computeNRMSE(output_collector, seq.getSeq(), seq.computeVarianceOut());
		stat_seq.rmse  = MathPerformance.computeRMSE(output_collector, seq.getSeq());
		stat_seq.sel   = MathPerformance.computeSEL(output_collector, seq.getSeq(), _sel_thresh);
		stat_seq.lel   = MathPerformance.computeLEL(output_collector, seq.getSeq(), _lel_thresh);
		
		return stat_seq;
	}//end of function
	
	/**
	 * This function runs an mESN with teacher-forcing using a given array of sequences.
	 * A provided mode and a provided time step is used under saving ESN states.
	 * 
	 * @param mode: running mode (either "washout" or "train")
	 * @param seq: given array of sequences
	 * @param time_step: index of the very 1st sample of provided sequences
	 * @return: object containing all performance statistics on the given sequence
	 */
	public stat_seq_C runTeacher(String mode, seq_C[] seq, int time_step)
	{
		int   i, j;
		int   num_sub;//number of sub-reservoirs
		int   size_in;//number of input neurons
		int   size_out;//number of output neurons
		int   max_idx_seq;//maximum index of the provided sequences
		int[] size_res;//number of reservoir neurons in each module
		int[] dummy_int;//dummy array of integers to call for saving the current state
		double[]   sample_in;//temporary array keeping current input vector
		double[]   sample_out;//temporary array keeping current output vector
		double[]   dummy_array;//dummy array to call for saving the current state
		double[]   out_bias_seq;//output bias to add to every sample before use
		double[][] res_state;//states of reservoir neurons of each ESN module at the current time step
		double[][] responsibility;//responsibility of modules at each output node
		double[][] out_bias;//bias of output nodes in each module
		double[][] ofb_sub;//OFB of each ESN module from each output neuron
		double[][] var_ofb;//variation of the OFB of each sub-reservoir from each output neuron
		double[][] sample_out_sub;//temporary array keeping current targets of all sub-reservoirs
		double[][] output_collector;//array with the ESN output on the whole provided sequence with the max index
		stat_seq_C stat_seq;//set of statistics of the ESN performance on the given sequence
		Matrix tmp_matrix;//temporary matrix

		num_sub   = _module.length;
		size_out  = _module[0].getNumNodes(layer_type_E.LT_OUTPUT);//it is enough to address the 1st module because
		                                                           //all modules have the same number of output neurons
		max_idx_seq = seq.length-1;
		dummy_int   = new int[num_sub];
		dummy_array = new double[num_sub];
		output_collector = new double[seq[max_idx_seq].getSeqLen()][size_out];
		stat_seq = new stat_seq_C(0);
		
		//assign 0's to all elements of the variation array
		//   because no variation of the OFB is considered in the current function
		tmp_matrix = new Matrix(num_sub, size_out);
		var_ofb    = tmp_matrix.getArray();
		
		//assign module specific values
		size_res  = new int[num_sub];
		res_state = new double[num_sub][];
		out_bias  = new double[num_sub][size_out];
		responsibility = new double[num_sub][size_out];
		ofb_sub = new double[num_sub][];
		for(i=0; i<num_sub; i++)
		{
			size_res[i]    = _module[i].getNumNodes(layer_type_E.LT_RES);
			res_state[i]   = new double[size_res[i]];
			ofb_sub[i]     = new double[size_out];
			dummy_int[i]   = 0;
			dummy_array[i] = 0;
		}
		
		//assign number of input neurons;
		//since an input layer is created or not created for all modules, it is enough to check it at one module
		size_in = _module[0].getNumNodes(layer_type_E.LT_INPUT);
		
		//output bias to add to every sample before saving
		out_bias_seq = seq[max_idx_seq].getBias();
		
		//process samples of the provided sequences
		//(it is enough to get the length of the 1st sequence because all sequences have the same length)
		for(i=0; i < seq[0].getSeqLen(); i++)
		{
			//prepare a vector with input values, if input neurons exist;
			//it does not matter from which sequence to assign a sample, all sequences have the same input values
			if(size_in!=0)
			{
				sample_in = seq[0].getSampleIn(i);
			}
			else
			{
				sample_in = null;
			}
			//sample is assigned from sequence with largest index because the reservoir is configured for this sequence
			sample_out = seq[max_idx_seq].getSampleOut(i);

			//assign current outputs of ESN modules whose values will be used as OFB in the next ESN computation
			for(j=0; j<num_sub; j++)
			{
				_module[j].getNodes(ofb_sub[j], layer_type_E.LT_OUTPUT);
			}
			
			//compute an ESN output in the current step only in order to compute the performance statistics below
			advance(sample_in, false);
			calculateEsnOutput();
			
			//teacher-forcing of mESN's outputs: set them here (by now they cannot be computed (that is "0") because
			//                                                  all Wout=0)
			for(j=0; j < size_out; j++)
			{
				_output[j] = sample_out[j];
				output_collector[i][j] = _output[j];
			}
			
			//compute target values for every active sub-reservoir at current time step
			sample_out_sub = computeTargetSub(seq, i);
			
			//teacher-forcing of separate modules: set target outputs of all ESN modules
			for(j=0; j < num_sub; j++)
			{
				_module[j].setNodes(sample_out_sub[j], layer_type_E.LT_OUTPUT, false);
			}
			
			//save the reservoir state at the current sample
			if(_esn_state_save!=null)
			{
				//restore the output bias before saving in the output
				for(j=0; j<sample_out.length; j++)
				{
					sample_out[j] += out_bias_seq[j];
				}
				//restore the output bias in a module's output
				if(num_sub==1)
				{
					for(j=0; j<sample_out.length; j++)
					{
						sample_out_sub[0][j] = sample_out[j];
					}
				}
				else
				{
					System.err.println("mESN.runTeacher: currently, output bias can be restored only for single module");
					System.exit(1);
				}
				
				//assign the states of the reservoir neurons for the saving
				for(j=0; j<num_sub; j++)
				{
					_module[j].getNodes(res_state[j], layer_type_E.LT_RES);
					_module[j].getOutputBias(out_bias[j]);
					_module[j].getResponsibility(responsibility[j]);
				}
				_esn_state_save.saveEsnState(time_step, mode, sample_in, sample_in, res_state, ofb_sub, var_ofb,
						                     dummy_array, dummy_array, dummy_int, sample_out_sub, sample_out,
						                     sample_out_sub, sample_out, out_bias, responsibility);
			}
			time_step++;
		}//for i
		
		//compute network's performance indicators on the most complicated washout sequence
		stat_seq.comp_ident_ok_incl_error       = 0;//assign dummy because implementation is only for evolutionary config
		stat_seq.comp_ident_ok = 0;//assign dummy because implementation is only for evolutionary config
		stat_seq.mse   = MathPerformance.computeMSE(output_collector, seq[seq.length-1].getSeq());
		stat_seq.nrmse = MathPerformance.computeNRMSE(output_collector, seq[seq.length-1].getSeq(),
				                                                        seq[seq.length-1].computeVarianceOut());
		stat_seq.rmse  = MathPerformance.computeRMSE(output_collector, seq[seq.length-1].getSeq());
		stat_seq.sel   = MathPerformance.computeSEL(output_collector, seq[seq.length-1].getSeq(), _sel_thresh);
		stat_seq.lel   = MathPerformance.computeLEL(output_collector, seq[seq.length-1].getSeq(), _lel_thresh);
		
		return stat_seq;
	}
	
	/**
	 * The function converts a structure of the host ESN to an array of strings.
	 * Afterwards it calls an output object for their formatting.
	 * Finally it saves the prepared host ESNs in a file.
	 * 
	 * @param idx_run: index of the run where the saved ESN was processed
	 * 
	 * @return: none
	 */
	public void saveEsn(int idx_run)
	{
		int i;
		int size_in;//number of input nodes
		Vector<String> esn_part;
		
		if(_esn_save.isSavingRequired()==true)
		{
			_esn_save.saveEsnHeader(idx_run);

			for(i=0; i<_module.length; i++)
			{
				esn_part = _module[i].getModuleTypeAsStr();
				_esn_save.saveEsnPart(esn_part, network_part_save_E.MPS_MODULE_TYPE, i);
				
				esn_part = _module[i].getOutputBiasAsStr();
				_esn_save.saveEsnPart(esn_part, network_part_save_E.MPS_OUTPUT_BIAS, i);
				
				esn_part = _module[i].getOutputBiasRangeAsStr();
				_esn_save.saveEsnPart(esn_part, network_part_save_E.MPS_OUTPUT_BIAS_MIN_MAX, i);
				
				esn_part = _module[i].getResponsibilityAsStr();
				_esn_save.saveEsnPart(esn_part, network_part_save_E.MPS_RESPONSIBILITY, i);
				
				esn_part = _module[i].getResponsibilityRangeAsStr();
				_esn_save.saveEsnPart(esn_part, network_part_save_E.MPS_RESPONSIBILITY_MIN_MAX, i);
				
				esn_part = _module[i].getSeqParametersAsString();
				_esn_save.saveEsnPart(esn_part, network_part_save_E.MPS_SEQ_PARAM, i);

				switch(_module[i]._module_type)
				{
					case MT_ESN:
					case MT_FFANN:
						esn_part = _module[i].getSeedAsStr();
						_esn_save.saveEsnPart(esn_part, network_part_save_E.MPS_SEED, i);
						size_in = _module[i].getNumNodes(layer_type_E.LT_INPUT);
						if(size_in!=0)//it is always 0 for SIN modules
						{
							esn_part = _module[i].getInitWeightsAsStr(layer_type_E.LT_INPUT);
							_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_IN_W, i);
							esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_ACTIVATION, layer_type_E.LT_INPUT);
							_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_ACT, i);
							esn_part = _module[i].getValidRangeAsStr(layer_type_E.LT_INPUT);
							_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_IN_MIN_MAX, i);
						}				
						if(_module[i].ExistBackLayer()==true)
						{
							esn_part = _module[i].getInitWeightsAsStr(layer_type_E.LT_OFB);
							_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_OFB_W, i);
						}
						esn_part = _module[i].getInitWeightsAsStr(layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_W_INIT, i);
						esn_part = _module[i].getSpecificLayerParamAsStr(exp_param_E.EP_RES_CONNECT,layer_type_E.LT_RES);				
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_CONNECT, i);
						esn_part = _module[i].getConnectivityMatrixAsStr();
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_CONNECT_MTRX, i);
						esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_ACTIVATION, layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_ACT, i);
						esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_BIAS, layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_BIAS, i);
						esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_SIZE, layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_SIZE, i);
						esn_part = _module[i].getSpecificLayerParamAsStr(exp_param_E.EP_RES_SPECTR_RAD,layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_SR, i);
						esn_part = _module[i].getSpecificLayerParamAsStr(exp_param_E.EP_RES_TOPOLOGY,layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_TOPOLOGY, i);
						esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_LEAKAGE_RATE, layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_LEAKAGE_RATE, i);
						esn_part = _module[i].getValidRangeAsStr(layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_RES_MIN_MAX, i);
				
						esn_part = _module[i].getInitWeightsAsStr(layer_type_E.LT_OUTPUT);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_OUT_W, i);
						esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_ACTIVATION, layer_type_E.LT_OUTPUT);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_OUT_ACT, i);
						esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_BIAS, layer_type_E.LT_OUTPUT);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_OUT_BIAS, i);
						esn_part = _module[i].getValidRangeAsStr(layer_type_E.LT_OUTPUT);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.EPS_OUT_MIN_MAX, i);
						break;
					case MT_SIN:
						
						esn_part = _module[i].getCommonLayerParamAsStr(exp_param_E.EP_SIN_PARAM_SIZE, layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.SPS_PARAM_SIZE, i);
						esn_part = _module[i].getSpecificLayerParamAsStr(exp_param_E.EP_SIN_PARAM,layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.SPS_PARAM_VALUE, i);						
						esn_part = _module[i].getValidRangeAsStr(layer_type_E.LT_RES);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.SPS_PARAM_MIN_MAX, i);
						
						esn_part = _module[i].getValidRangeAsStr(layer_type_E.LT_OUTPUT);
						_esn_save.saveEsnPart(esn_part, network_part_save_E.SPS_OUT_MIN_MAX, i);
						break;
					default:
						System.err.println("mESN.saveEsn: unknown module type");
						System.exit(1);
						break;
				}
			}
			
			_esn_save.closeFile();
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
	 * The function computes deviations of the network's outputs from their provided target values
	 * It uses available ESN outputs.
	 * 
	 * @param sample_out: provided target output vector
	 * @return computed deviations
	 */
	public double[] computeError(double[] sample_out)
	{
		int i;
		double[] deviations;
		
		deviations = new double[_output.length];
		
		for(i=0; i<_output.length; i++)
		{
			deviations[i] = sample_out[i] - _output[i];
		}

		return deviations;
	}
	
	/**
	 * The function performs an update of an mESN under transition to the next time step.
	 * The provided array is an external ESN's input at the next time step.
	 * If it is required the function initiates storing a module output in its history.
	 * 
	 * @param sample_in: input sample at the next time step
	 * @param do_store: request to store a module output in its history
	 */
	public void advance(double[] sample_in, boolean do_store)
	{
		//advance all modules
		advanceModule(-1, sample_in, do_store);
	}
	
	/**
	 * The function computes an ESN output from the currently available outputs of separate modules.
	 */
	public void calculateEsnOutput()
	{
		int      i,j;
		int      num_sub;//number of modules
		int      size_out;//number of output neurons
		double[] responsibility;//module responsibility
		double[] module_output;//current output neurons of a module
		double[] output_bias;//output bias of the current module
		
		size_out = _output.length;
		
		//initialize outputs of mESN
		for(i=0; i<size_out; i++)
		{
			_output[i] = 0;
		}
		
		//advance the modules
		num_sub = _module.length;
		for(i=0; i<num_sub; i++)
		{
			//compute an mESN's output
			responsibility = _module[i].getResponsibility();
			module_output = _module[i].getNodes(layer_type_E.LT_OUTPUT);
			output_bias = _module[i].getOutputBias();
			for(j=0; j<size_out; j++)
			{
				_output[j] += (responsibility[j]*module_output[j]);
				_output[j] += output_bias[j];
			}
		}
	}
	
	/**
	 * The function performs an update of a specified ESN module under transition to the next time step.
	 * The module is specified by its index. If the provided index is "-1", the function updates all modules.
	 * The provided array is an external ESN's input at the next time step.
	 * If it is required the function initiates storing a module output in its history.
	 * 
	 * @param module_idx: index of a specified module
	 * @param sample_in: input sample at the next time step
	 * @param do_store: request to store a module output in its history
	 */
	public void advanceModule(int module_idx, double[] sample_in, boolean do_store)
	{
		int i;
		int num_sub;//number of modules
		
		//advance the modules
		num_sub = _module.length;
		if(module_idx==-1)
		{
			for(i=0; i<num_sub; i++)
			{
				_module[i].advance(sample_in, do_store);
			}
		}
		else
		{
			_module[module_idx].advance(sample_in, do_store);
		}
	}
	
	/**
	 * The function is an interface to prepare a history array of a specified module for storing the next elements.
	 * The module is specified by its index.
	 * 
	 * @param sub_idx: module index
	 */
	public void startModuleNodesHistory(int sub_idx)
	{
		_module[sub_idx].startNodesHistory();
	}
	
	/**
	 * The function is an interface to initiate a storing of nodes history of a specified module.
	 * The module is specified by its index.
	 * 
	 * @param sub_idx: module index
	 */
	public void storeModuleNodesHistory(int sub_idx)
	{
		_module[sub_idx].storeNodesHistory();
	}
	
	/**
	 * The function retrieves a history of a specified layer from a specified module.
	 * The layer is specified by its type.
	 * The module is specified by its index.
	 * 
	 * @param sub_idx: module index
	 * @param layer_type: required layer type
	 * @param storage: array where a retrieved history must be stored
	 */
	public void getModuleNodesHistory(int sub_idx, layer_type_E layer_type, double[][] storage)
	{
		_module[sub_idx].getNodesHistory(layer_type, storage);
	}
	
	/**
	 * The function computes an output of a specified module for all ESN's output neurons.
	 * The module is specified by a provided index.
	 * The computed output is stored in the provided array.
	 * 
	 * @param sub_idx: module index
	 * @param sub_output: storage for the computed module output
	 */
	public void computeSubOutput(int sub_idx, double[] sub_output)
	{
		int      i;
		int      size_out;//number of output neurons
		double[] output;//output of required module
		
		output = _module[sub_idx].calculateOutputVector();
		//assign values for output
		size_out = output.length;
		for(i=0; i<size_out; i++)
		{
			sub_output[i] = output[i];
		}
	}
	
	/**
	 * The function indicates a module specified by its index.
	 * 
	 * @param sub_idx: module index
	 * @return: requested module
	 */
	public Module getModule(int sub_idx)
	{
		return _module[sub_idx];
	}
	
	/**
	 * The function indicates a number of nodes of a specified type in a specified module.
	 * The module is specified by its index.
	 * 
	 * @param sub_idx: module index
	 * @param layer_type: specified layer type
	 * @return: number of nodes
	 */
	public int getNumNodesModule(int sub_idx, layer_type_E layer_type)
	{
		return _module[sub_idx].getNumNodes(layer_type);
	}
	
	/**
	 * The function calls a specified module to get nodes of a specified layer.
	 * The node states are stored in a provided storage.
	 * The layer is specified by its type.
	 * The module is specified by its index.
	 *  
	 * @param sub_idx: module index
	 * @param layer_type: layer type
	 * @param storage: array where values must be stored
	 */
	public void getModuleNodes(int sub_idx, layer_type_E layer_type, double[] storage)
	{
		_module[sub_idx].getNodes(storage, layer_type);
	}
	
	/**
	 * The function assigns provided values to neurons of a specified layer in a specified module.
	 * The layer is specified by its type.
	 * The module is specified by its index.
	 *  
	 * @param sub_idx: module index
	 * @param layer_type: layer type
	 * @param values: array of values to be assigned
	 */
	public void setModuleNodes(int sub_idx, layer_type_E layer_type, double[] values)
	{
		boolean apply_activation;//request to apply activation to assigned node values
		
		switch(layer_type)
		{
			case LT_INPUT:
				apply_activation = true;
				break;
			case LT_RES:
				apply_activation = false;
				break;
			case LT_OUTPUT:
				apply_activation = false;
				break;
			case LT_OFB:
				apply_activation = false;
				System.err.println("mESN.setModuleNodes: no node values are expected for OFB");
				System.exit(1);
				break;
			default:
				apply_activation = false;
				System.err.println("mESN.setModuleNodes: unknown layer type");
				System.exit(1);
				break;
		}
		
		_module[sub_idx].setNodes(values, layer_type, apply_activation);
	}
	
	/**
	 * The function gets a valid range of a node in a specified layer and module.
	 * The layer is specified by its type.
	 * The module is specified by its index.
	 *  
	 * @param sub_idx: module index
	 * @param node_idx: node index
	 * @param layer_type: layer type
	 * @param range: array of intervals to store valid intervals
	 */
	public void getModuleNodeRange(int sub_idx, int node_idx, layer_type_E layer_type, interval_C[] range)
	{
		_module[sub_idx].getNodeValidRange(node_idx, layer_type, range);
	}
	
	/**
	 * The function initiates a storing of all states of all modules in a specified storage.
	 * The storage is specified by its type.
	 * 
	 * @param type: provided storage type
	 */
	public void storeModuleNodes(storage_type_E type)
	{
		int i;
		int num_sub;//number of ESN modules
		
		num_sub = _module.length;
		for(i=0; i<num_sub; i++)
		{
			_module[i].storeNodes(type);
		}
	}
	
	/**
	 * The function initiates a storing of all states of a specified module in a specified storage.
	 * The module is specified by its index.
	 * The storage is specified by its type.
	 * 
	 * @param sub_idx: module index
	 * @param type: provided storage type
	 */
	public void storeModuleNodes(int sub_idx, storage_type_E type)
	{
		_module[sub_idx].storeNodes(type);
	}
	
	/**
	 * The function restores all previously preserved states of all active modules. The states are restored from
	 * a specified storage. The storage is specified by its type.
	 * States of inactive modules are set to 0. The function restores states of the ESN's output neurons accordingly
	 * to the stored states of its modules.
	 * 
	 * @param active_modules: indicator array of active modules
	 * @param type: storage type
	 */
	public void restoreModuleNodes(boolean[] active_modules, storage_type_E type)
	{
		int i;
		int num_sub;//number of ESN modules
		
		num_sub = _module.length;
		for(i=0; i<num_sub; i++)
		{
			if(active_modules[i]==true)
			{
				_module[i].restoreNodes(type);
			}
			else
			{
				_module[i].clearNodes();
			}
		}
	}
	
	/**
	 * The function restores all previously preserved states of a specified module from a specified storage.
	 * The module is specified by its index.
	 * The storage is specified by its type.
	 * 
	 * @param sub_idx: module index
	 * @param type: storage type
	 */
	public void restoreModuleNodes(int sub_idx, storage_type_E type)
	{
		_module[sub_idx].restoreNodes(type);
	}
	
	/**
	 * The function initiates a storing of output weights of all ESN modules for their possible activation later.
	 */
	public void storeModuleOutputWeights()
	{
		int i;
		
		for(i=0; i<_module.length; i++)
		{
			_module[i].storeWeightsInit();
		}
	}
	
	/**
	 * The function forwards provided values for assignment as responsibilities of a specified module.
	 * 
	 * @param sub_idx: index of a specified module
	 * @param responsibility: provided responsibility values
	 */
	public void setModuleResponsibility(int sub_idx, double[] responsibility)
	{
		_module[sub_idx].setResponsibility(responsibility);
	}
	
	/**
	 * The function calls a specified module to obtain its responsibilities.
	 * 
	 * @param sub_idx: index of a specified module
	 * @param storage: array to retrieved responsibilities
	 */
	public void getModuleResponsibility(int sub_idx, double[] storage)
	{
		_module[sub_idx].getResponsibility(storage);
	}
	
	/**
	 * The function assign a responsibility range of a specified module at the specified output.
	 * 
	 * @param sub_idx: index of module
	 * @param out_idx: index of module's output
	 * @param range: interval where retrieved responsibility range must be stored
	 */
	public void getModuleResponsibilityRange(int sub_idx, int out_idx, interval_C range)
	{
		_module[sub_idx].getResponsibilityRange(out_idx, range);
	}
	
	/**
	 * The function forwards provided values for assignment as an output bias of a specified module.
	 * 
	 * @param sub_idx: index of a specified module
	 * @param output_bias: provided output bias
	 */
	public void setModuleOutputBias(int sub_idx, double[] output_bias)
	{
		_module[sub_idx].setOutputBias(output_bias);
	}
	
	/**
	 * The function calls a specified module to obtain its output bias.
	 * 
	 * @param sub_idx: index of a specified module
	 * @param storage: array of the retrieved output bias
	 */
	public void getModuleOutputBias(int sub_idx, double[] storage)
	{
		_module[sub_idx].getOutputBias(storage);
	}
	
	/**
	 * The function indicates the current values of the whole modular network.
	 * 
	 * @return: current output of the whole modular network
	 */
	public double[] getOutput()
	{
		return _output;
	}
	
	/**
	 * The function stores the indicator of activity for each module in the provided storage. 
	 * 
	 * @param indicator: array to store the indicator of activity
	 */
	public void GetModuleActiveIndicator(boolean[] indicator)
	{
		int i;
		
		for(i=0; i<_module.length; i++)
		{
			indicator[i] = _module[i].getConfigured();
		}
	}
}