package experiment;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Object;
import java.nio.channels.FileChannel;
import java.util.StringTokenizer;
import java.util.Vector;

import adaptation.DiffEvolutionParam.config_ea_init_E;
import adaptation.DiffEvolutionParam.config_ea_order_method_E;

import types.interval_C;
import types.multi_val_C;
import types.vector_C;

import MathDiff.MathDistortion.distortion_E;
import MathDiff.MathNoise.noise_E;

import esn.Activation.activation_E;
import esn.EsnModule.config_method_E;
import esn.Layer.leakage_assign_E;
import esn.mESN.config_ea_mode_E;
import esn.ReservoirInitialization;
import experiment.ExpParam;
import experiment.ExpParam.exp_param_E;
import experiment.ExpParam.req_val_E;

/**
 * This class implements functions for saving the results of the experiment. 
 * @author Danil Koryakin
 *
 */
public class ExpOutput {
	
	public enum stat_exp_E
	{
		SE_MSE_AVG_TST,
		SE_DEV_MSE_TST,
		SE_MIN_MSE_TST,
		SE_IDX_MIN_MSE_TST,
		SE_MAX_MSE_TST,
		SE_IDX_MAX_MSE_TST,
		SE_MED_MSE_TST,
		SE_IDX_MED_MSE_TST,
		SE_NRMSE_AVG_TST,
		SE_DEV_NRMSE_TST,
		SE_RMSE_AVG_TST,
		SE_DEV_RMSE_TST,
		SE_MSE_AVG_TRN,
		SE_DEV_MSE_TRN,
		SE_NRMSE_AVG_TRN,
		SE_DEV_NRMSE_TRN,
		SE_RMSE_AVG_TRN,
		SE_DEV_RMSE_TRN,
		SE_MIN_NRMSE_TST,
		SE_IDX_MIN_NRMSE_TST,
		SE_MAX_NRMSE_TST,
		SE_IDX_MAX_NRMSE_TST,
		SE_MED_NRMSE_TST,
		SE_IDX_MED_NRMSE_TST,
		SE_RUN_REPEAT_TOTAL,//total number of repetitions of runs which were performed for current parameter vector
		SE_SEL_SUM_TST,
		SE_DEV_SEL_TST,    
		SE_BEST_SEL_TST,   
		SE_IDX_BEST_SEL_TST,
		SE_SEL_SUM_TRN,
		SE_DEV_SEL_TRN,    
		SE_BEST_SEL_TRN,   
		SE_IDX_BEST_SEL_TRN,
		SE_MSE_AVG_CFG,
		SE_MIN_MSE_CFG,
		SE_IDX_MIN_MSE_CFG,
		SE_SEL_AVG_CFG,
		SE_MAX_SEL_CFG,   
		SE_IDX_MAX_SEL_CFG,
		SE_LEL_SUM_CFG,
		SE_DEV_LEL_CFG,    
		SE_BEST_LEL_CFG,   
		SE_IDX_BEST_LEL_CFG,
		SE_CIOK_AVG_BY_CFG_ERR,//component identification rate (average of Component Identification OK) incl. MSE
		SE_CIOK_AVG_CFG,//average of Component Identification rate without considering the error
		SE_CIOK_AVG_BY_TST_ERR//config component identification rate is OK + small test MSE
	};
	
	public enum stat_run_E
	{
		SR_CIOK_INCL_ERR,//Component Identification is OK according to active modules and according to error
		SR_CIOK,//Component Identification is OK without considering the error
		SR_MSE,
		SR_NRMSE,
		SR_RMSE,
		SR_SEL,
		SR_LEL
	};
	
	public enum header_type_E
	{
		HT_EXP_STAT,//header for statistics for the whole experiment
		HT_RUN_STAT,//header for statistics for a single parameter vector
		HT_RUN_DATA //header for saving data of a single run
	};
	
	/**
	 * This is an enumeration of common mESN's attributes to be saved for the whole mESN.
	 * @author Danil Koryakin
	 */
	public enum esn_state_E
	{
		ES_STEP,//time step
		ES_MODE,//mode of applying the ESN (washout, train, test)
		ES_IN,//provided values of input neurons (they are common for all modules)
		ES_IN_TARGET,//input value which was discovered during synchronization
		ES_OUT,//computed ESN output
		ES_OUT_TARGET//target of the ESN output
	}
	
	/**
	 * This is an enumeration of attributes to be saved for each ESN module.
	 * @author Danil Koryakin
	 */
	public enum module_state_E
	{
		MS_RES,//computed states of the reservoir neurons
		MS_VAR_OFB,//variation of module's OFB
		MS_VAR_FIT,//fitness variance (applicable only for evolutionary synchronization)
		MS_VAR_IND,//variance of individuals in a population (applicable only for evolutionary synchronization)
		MS_NUM_SRT,//number of sorted individuals in a population (applicable only for evolutionary synchronization)
		MS_OFB,//output feedback value for computing an output of an ESN module
		MS_OUT,//computed outputs of modules
		MS_TARGET,//target values of a module
		MS_OUT_BIAS,//bias of the module's output
		MS_RESPONSIBILITY//responsibility of a module
	}
	
	public enum network_part_save_E
	{
		//*** COMMON parts for all types of modules
		
		MPS_MODULE_TYPE,//module types
		MPS_SEED,//seeding value for random generators
		MPS_RESPONSIBILITY,//module responsibilities
		MPS_RESPONSIBILITY_MIN_MAX,//minimum and maximum values of responsibilities
		MPS_OUTPUT_BIAS,//bias of module output
		MPS_OUTPUT_BIAS_MIN_MAX,//minimum and maximum values of output bias
		MPS_SEQ_PARAM,//parameters of a training sequence
		
		//*** specific parts of ESN module
		
		EPS_IN_W,//weight of connections between input and reservoir neurons 
		EPS_IN_ACT,//activation of input neurons
		EPS_IN_MIN_MAX,//minimum and maximum values of the input neurons
		EPS_RES_W_INIT,//initially generated reservoir weights
		EPS_RES_ACT,//activation of reservoir neurons
		EPS_RES_BIAS,//bias or reservoir neurons
		EPS_RES_SIZE,//size of reservoir modules
		EPS_RES_SR,//spectral radii of reservoir modules
		EPS_RES_CONNECT,//exclusive connectivity and connectivity in overlaps between modules
		EPS_RES_TOPOLOGY,//method of reservoir generation
		EPS_RES_LEAKAGE_RATE,//leakage rate of reservoir neurons
		EPS_RES_CONNECT_MTRX,//connectivity matrix of reservoir (usually used only for saving)
		EPS_RES_MIN_MAX,//minimum and maximum values of reservoir neurons
		EPS_OUT_W,//output weights
		EPS_OUT_ACT,//activation of output neurons
		EPS_OUT_BIAS,//bias of output neurons
		EPS_OUT_MIN_MAX,//minimum and maximum values of output neurons
		EPS_OFB_W,//output feedback weights
		
		//*** specific parts of sine module
		
		SPS_PARAM_SIZE,//sine module: number of parameters to be tuned
		SPS_PARAM_VALUE,//sine module: available values of module parameters: [0] - frequency, [1] - phase
		SPS_PARAM_MIN_MAX,//sine module: minimum and maximum values of sine parameters (frequency and phase)
		SPS_OUT_MIN_MAX,//sine module: minimum and maximum values of a sine output
		
		//*** specific parts of FF-ANN module
		FPS_IN_NORM_COEF,//FF-ANN: coefficient for normalization of values of the input vector 
		FPS_OUT_NORM_COEF//FF-ANN: coefficient for normalization of values of the output vector
	};
	
	/**
	 * class defines the beginning of an array which keeps all values of only one statistics
	 * through all experiments
	 * @author Danil Koryakin
	 */
	private class stat_data_C
	{
		private String   name = "";
		private boolean  save;//flag that the value should be saved
		private Object[] _stat_data;
		
		public stat_data_C(boolean is_integer, boolean is_string, boolean do_save, int num_elements)
		{
			save = do_save;
			if(is_string==true)
			{
				_stat_data = new String[num_elements];
			}
			else
			{
				if(is_integer)
				{
					_stat_data = new Integer[num_elements];
				}
				else
				{
					_stat_data = new Double[num_elements];
				}
			}
		}
	};
	
	/**
	 * class defines a storage to keep performance-independent parameters of all runs as well as all performance
	 * indicators for all intervals for all runs
	 * @author Danil Koryakin
	 */
	private class data_run_C
	{
		private stat_data_C _run_seed;//array with values that were used for seeding the successful repeat of the run
		private stat_data_C _run_repeat;//array with numbers of times to repeat a run till it is successfully performed
		private stat_data_C _run_seq;//array with sequence names that were used in each run
		private stat_data_C[][] _run_perform;//performance indicators on all intervals for all runs
                                     //1D: intervals
                                     //2D: performance indicators
		
		/**
		 * This class constructor builds an object to keep all performance statistics for all runs.
		 * Depending on the provided sequence purposes, it marks the performance indicators for saving.
		 * For example, the large-error length is saved only for the configuration interval.
		 * 
		 * @param num_runs: number of runs
		 * @param seq_purpose: vector with purposes for applying ESN on each interval
		 */
		public data_run_C(int num_runs, vector_C seq_purpose)
		{
			int i;
			int num_intervals;
			
			num_intervals = seq_purpose.getSize();
			
			_run_seed   = new stat_data_C(true, true,  true, num_runs);//"seed" is saved as a string "{..,..,..}"
			_run_repeat = new stat_data_C(true, false, true, num_runs);
			_run_seq    = new stat_data_C(true, true,  true, num_runs);
			_run_seed.name   = "rand_seed";//values for seeding random generators in the current run
			_run_repeat.name = "run_repeats";//number of repetitions of the current run
			_run_seq.name    = "sequence";//name of a sequence which was used in the current run
			
			_run_perform = new stat_data_C[num_intervals][stat_run_E.values().length];
			for(i=0; i<num_intervals; i++)
			{
				for(stat_run_E j : stat_run_E.values())
				{
					switch(j)
					{
						case SR_MSE:
							_run_perform[i][j.ordinal()] = new stat_data_C(false, false, true, num_runs);
							_run_perform[i][j.ordinal()].name = "mse_interval_" + i;
						break;
						case SR_NRMSE:
							_run_perform[i][j.ordinal()] = new stat_data_C(false, false, true, num_runs);
							_run_perform[i][j.ordinal()].name = "nrmse_interval_" + i;
						break;
						case SR_RMSE:
							_run_perform[i][j.ordinal()] = new stat_data_C(false, false, true, num_runs);
							_run_perform[i][j.ordinal()].name = "rmse_interval_" + i;
						break;
						case SR_SEL:
							_run_perform[i][j.ordinal()] = new stat_data_C(true,  false, true, num_runs);
							_run_perform[i][j.ordinal()].name = "SEL_interval_" + i;
						break;
						case SR_LEL:
							_run_perform[i][j.ordinal()] = new stat_data_C(true,  false, true, num_runs);
							_run_perform[i][j.ordinal()].name = "LEL_interval_" + i;
						break;
						case SR_CIOK_INCL_ERR:
							_run_perform[i][j.ordinal()] = new stat_data_C(true,  false, true, num_runs);
							_run_perform[i][j.ordinal()].name = "CiokInclError_interval_" + i;
						break;
						case SR_CIOK:
							_run_perform[i][j.ordinal()] = new stat_data_C(true,  false, true, num_runs);
							_run_perform[i][j.ordinal()].name = "CiokWithoutError_interval_" + i;
						break;
						default:
							System.err.println("data_run_C: no constructor call is defined for current indicator yet");
							System.exit(1);
						break;
					}//switch j
					//add the sequence purpose as a suffix to the name of the performance indicator on current interval
					_run_perform[i][j.ordinal()].name+= seq_purpose.getElement(i);
				}//for j
			}//for i
		}
		
		/**
		 * The function returns seeding values that were used for ESN modules in a specified run.
		 * The returned seeding values are formatted as a string.
		 * 
		 * @param idx_run: index of a specified run
		 * @return: string with seeding values
		 */
		public String getSeed(int idx_run)
		{
			return (String)_run_seed._stat_data[idx_run];
		}
		
		/**
		 * The function returns a number of runs for which the performance indicators were stored.
		 * 
		 * @return: number of stored runs
		 */
		public int getNumRuns()
		{
			return _run_seed._stat_data.length;
		}
		
		/**
		 * The functions returns a total number of performance indicators from all sequence intervals to be saved.
		 * 
		 * @return number of performance indicators to be saved
		 */
		public int getNumIndicatorsToSave()
		{
			int i, j;
			int num_save;//output variable
			
			num_save = 0;
			if(_run_seed.save==true)
			{
				num_save++;
			}
			if(_run_repeat.save==true)
			{
				num_save++;
			}
			if(_run_seq.save==true)
			{
				num_save++;
			}
			
			for(i=0; i<_run_perform.length; i++)
			{
				for(j=0; j<_run_perform[0].length; j++)
				{
					if(_run_perform[i][j].save==true)
					{
						num_save++;
					}
				}
			}
			
			return num_save;
		}
		
		/**
		 * The function returns an array with the names of the run data to be saved.
		 * 
		 * @return: array with the names to be saved
		 */
		public Vector<String> getNamesToSave()
		{
			int i, j;
			Vector<String> names;
			
			names = new Vector<String>(0, 1);
			
			//check seeding values, number of runs and sequence name
			if(_run_seed.save==true)
			{
				names.add(_run_seed.name);
			}
			if(_run_repeat.save==true)
			{
				names.add(_run_repeat.name);
			}
			if(_run_seq.save==true)
			{
				names.add(_run_seq.name);
			}
			
			//go over the performance indicators
			for(i=0; i<_run_perform.length; i++)
			{
				for(j=0; j<_run_perform[0].length; j++)
				{
					if(_run_perform[i][j].save==true)
					{
						names.add(_run_perform[i][j].name);
					}
				}
			}
			
			return names;
		}
		
		/**
		 * The function returns an array with values to be saved for the specified run.
		 * 
		 * @param idx_run: index of the specified run
		 * @return: array of values to be saved
		 */
		public Vector<stat_data_C> getValuesToSave(int idx_run)
		{
			int i, j;
			boolean is_int;//indicator that the value is an integer
			stat_data_C stat_data;//temporary object to be saved
			Vector<stat_data_C> data;//output array
			
			data = new Vector<stat_data_C>(0, 1);
			
			//check seeding values, number of runs and sequence name
			if(_run_seed.save==true)
			{
				stat_data = new stat_data_C(true, true, _run_seed.save, 1);
				stat_data._stat_data[0] = _run_seed._stat_data[idx_run];
				data.add(stat_data);
			}
			if(_run_repeat.save==true)
			{
				stat_data = new stat_data_C(true, false, _run_repeat.save, 1);
				stat_data._stat_data[0] = _run_repeat._stat_data[idx_run];
				data.add(stat_data);
			}
			if(_run_seq.save==true)
			{
				stat_data = new stat_data_C(true, true, _run_seq.save, 1);
				stat_data._stat_data[0] = _run_seq._stat_data[idx_run];
				data.add(stat_data);
			}
			
			//go over the performance indicators
			for(i=0; i<_run_perform.length; i++)
			{
				for(j=0; j<_run_perform[0].length; j++)
				{
					if(_run_perform[i][j].save==true)
					{
						if(_run_perform[i][j]._stat_data[idx_run].getClass()==Integer.class)
						{
							is_int = true;
						}
						else
						{
							is_int = false;
						}
						
						stat_data = new stat_data_C(is_int, false, true, 1);
						stat_data._stat_data[0] = _run_perform[i][j]._stat_data[idx_run];		
						data.add(stat_data);
					}//if to be saved
				}//for j
			}//for i
			
			return data;
		}
		
		/**
		 * The function returns an array of values for the given name of the performance indicator.
		 * 
		 * @param name: given name of the performance indicator
		 * @return: array of values for the specified indicator
		 */
		public Object[] getValuesByName(String name)
		{
			int i, j;
			Object[] data;//output array
			
			if(name.contentEquals(_run_seed.name)==true)
			{
				data = _run_seed._stat_data;
			}
			else if(name.contentEquals(_run_repeat.name)==true)
			{
				data = _run_repeat._stat_data;
			}
			else if(name.contentEquals(_run_seq.name)==true)
			{
				data = _run_seq._stat_data;
			}
			else
			{
				data = null;
				for(i=0; i<_run_perform.length; i++)
				{
					for(j=0; j<_run_perform[0].length; j++)
					{
						if(name.contentEquals(_run_perform[i][j].name)==true)
						{
							if(data==null)
							{
								data = _run_perform[i][j]._stat_data;
							}
							else
							{
								System.err.println("getValuesByName: more than 2 indicators with the same name");
								System.exit(1);
							}
						}//is names are equal
					}//for j
				}//for i
			}//_run_seed, _run_repeat or else
			
			return data;
		}
	};
	
	/**
	 * The class is responsible for formating a saved string with the states of all ESN neurons.
	 * All fields of the current string have the same width which equals to the number of symbols to save the longest
	 * of the provided values. This should avoid frequent jumps of the values over the consequently saved strings.
	 * @author Danil Koryakin
	 */
	public class esn_state_C
	{
		/**
		 * The class keeps information about a primary variable of the ESN state. It also includes an index of another
		 * state variable whose name is combined with the name of the primary variable.
		 * @author Danil Koryakin
		 */
		private class state_var_C
		{
			private String name;//name of the primary state variable
			private String short_name;//short name of the state variable which should be used in combined names
			private int    state_size;//number of elements in the vector of this state variable
			private int    min_width;//initial number of symbols in the column width; it can be increased later
			private int    idx_related;//index of the related state variable
		}

		private int[] _width_cur;//array with widths of all columns in the output file
		private int _num_sub_cols;//total number of sub-columns to be output
		private state_var_C[]   _esn_state;//array with names of ESN states from above-module level
		private state_var_C[][] _module_state;//array with names of variables to be saved for all ESN modules:1D-modules
		private File _file;//file object to save ESN states
		private BufferedWriter _bw;//buffered writer of a file to save ESN states
		
		/**
		 * The constructor of the class which formats the output of the ESN state. The input parameters indicate
		 * the number of elements in the corresponding ESN state. If a certain parameter value is "-1" then
		 * the corresponding ESN state should not be saved.
		 * @param num_step: "0 or -1"; "0" is when the time step should be saved, "1" - otherwise
         * @param num_mode: "0 or -1"; "0" is when the mode should be saved, "1" - otherwise
		 * @param num_in: number of sub-columns to save the input neurons
		 * @param num_res: numbers of reservoir neurons in each module: 1D - modules, 2D - numbers of neurons
		 * @param num_out: number of sub-columns to save the output neurons
		 */
		public esn_state_C(int num_step, int num_mode, int num_in, int[] num_res, int num_out)
		{
			int i,j;
			int num_sub;//number of reservoir neurons in each ESN module
			int idx_state;//index of the related state variable
			
			_width_cur = null;//array is allocated in the function where the names of the columns are created
			
			//check the input
			if(num_step!=0 && num_step!=-1)
			{
				System.err.println("esn_state_C: number of sub-columns for saving the time steps may be only 0 or -1");
				System.exit(1);
			}
			if(num_mode!=0 && num_mode!=1)
			{
				System.err.println("esn_state_C: number of sub-columns for saving the mode may be only 0 or -1");
				System.exit(1);
			}
			
			/*** mESN ***/
			
			//define names for all columns in the file with the ESN states
			_esn_state = new state_var_C[esn_state_E.values().length];
			for(i=0; i<esn_state_E.values().length; i++)
			{
				_esn_state[i] = new state_var_C();
			}
			idx_state = esn_state_E.ES_STEP.ordinal();
			_esn_state[idx_state].name = "TimeStep";
			_esn_state[idx_state].state_size  = num_step;
			_esn_state[idx_state].short_name = "_Step_";
			_esn_state[idx_state].idx_related = -1;
            _esn_state[idx_state].min_width = _esn_state[0].name.length();
            idx_state = esn_state_E.ES_MODE.ordinal();
			_esn_state[idx_state].name = "Mode";
			_esn_state[idx_state].state_size  = num_mode;
			_esn_state[idx_state].short_name = "_Mode_";
			_esn_state[idx_state].idx_related = -1;
			_esn_state[idx_state].min_width = 7;//length of the name "washout"
			idx_state = esn_state_E.ES_IN.ordinal();
			_esn_state[idx_state].name = "InputNeuron_";
			_esn_state[idx_state].state_size  = num_in;
			_esn_state[idx_state].short_name = "_In_";
			_esn_state[idx_state].idx_related = -1;
			_esn_state[idx_state].min_width = 25;//length of the name "VariationOfbSubreservoir_0_Out_0"
			idx_state = esn_state_E.ES_IN_TARGET.ordinal();
			_esn_state[idx_state].name = "InputTarget_";
			_esn_state[idx_state].state_size  = num_in;
			_esn_state[idx_state].short_name = "_InTarget_";
			_esn_state[idx_state].idx_related = -1;
			_esn_state[idx_state].min_width = 25;//length of the name "VariationOfbSubreservoir_0_Out_0"
			idx_state = esn_state_E.ES_OUT.ordinal();
			_esn_state[idx_state].name = "OutputNeuron_";
			_esn_state[idx_state].state_size  = num_out;
			_esn_state[idx_state].short_name = "_Out_";
			_esn_state[idx_state].idx_related = -1;
			_esn_state[idx_state].min_width = 25;//length of the name "VariationOfbSubreservoir_0_Out_0"
			idx_state = esn_state_E.ES_OUT_TARGET.ordinal();
			_esn_state[idx_state].name = "OutputTarget_";
			_esn_state[idx_state].state_size  = num_out;
			_esn_state[idx_state].short_name = "_OutTarget_";
			_esn_state[idx_state].idx_related = -1;
			_esn_state[idx_state].min_width = 25;//length of the name "VariationOfbSubreservoir_0_Out_0"
			
			/*** ESN modules ***/
			
			num_sub = num_res.length;
			_module_state = new state_var_C[num_sub][module_state_E.values().length];
			for(i=0; i<num_sub; i++)
			{
				for(j=0; j<module_state_E.values().length; j++)
				{
					_module_state[i][j] = new state_var_C();
				}//for j
			
				idx_state = module_state_E.MS_RES.ordinal();
				_module_state[i][idx_state].name = "ReservoirNeuron_";
				_module_state[i][idx_state].state_size  = num_res[i];
				_module_state[i][idx_state].short_name = "_Res_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width = 32;//length of the name "VariationOfbSubreservoir_0_Out_0"
				idx_state = module_state_E.MS_OFB.ordinal();
				_module_state[i][idx_state].name = "OfbFromOut_";
				_module_state[i][idx_state].state_size  = num_out;
				_module_state[i][idx_state].short_name  = "_OfbSub_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width  = 32;//length of the name "VariationOfbSubreservoir_0_Out_0"
			    idx_state = module_state_E.MS_VAR_OFB.ordinal();
			    _module_state[i][idx_state].name = "VariationOfbAtOut_";
			    _module_state[i][idx_state].state_size  = num_out;
			    _module_state[i][idx_state].short_name  = "_VarOfbSub_";
			    _module_state[i][idx_state].idx_related = -1;
			    _module_state[i][idx_state].min_width  = 32;//length of the name "VariationOfbSubreservoir_0_Out_0"
			    idx_state = module_state_E.MS_VAR_FIT.ordinal();
			    _module_state[i][idx_state].name = "VarianceOfFitness_";
			    _module_state[i][idx_state].state_size  = 1;//"1" because there is only one value per module
			    //_module_state[i][idx_state].state_size  = -1;//"-1" would avoid saving this performance statistics
			    _module_state[i][idx_state].short_name  = "_VarFitness_";
			    _module_state[i][idx_state].idx_related = -1;
			    _module_state[i][idx_state].min_width  = 19;//length of the name "VarianceOfFitness_0"
				idx_state = module_state_E.MS_VAR_IND.ordinal();
				_module_state[i][idx_state].name = "VarianceOfIndividuals_";
				_module_state[i][idx_state].state_size  = 1;//"1" because there is only one value per module
				//_module_state[i][idx_state].state_size  = -1;//"-1" would avoid saving this performance statistics
				_module_state[i][idx_state].short_name  = "_VarIndivid_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width  = 23;//length of the name "VarianceOfIndividuals_0"
				idx_state = module_state_E.MS_NUM_SRT.ordinal();
				_module_state[i][idx_state].name = "NumberOfSortedIndividuals";
				_module_state[i][idx_state].state_size  = 0;
				_module_state[i][idx_state].short_name  = "_NumSort_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width  = 25;//length of the name "NumberOfSortedIndividuals"
				idx_state = module_state_E.MS_OUT.ordinal();
				_module_state[i][idx_state].name = "Output_";
				_module_state[i][idx_state].state_size  = num_out;
				_module_state[i][idx_state].short_name = "_Out_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width = 32;//length of the name "VariationOfbSubreservoir_0_Out_0"
				idx_state = module_state_E.MS_TARGET.ordinal();
				_module_state[i][idx_state].name = "Target_";
				_module_state[i][idx_state].state_size  = num_out;
				_module_state[i][idx_state].short_name = "_Target_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width = 32;//length of the name "VariationOfbSubreservoir_0_Out_0"
				idx_state = module_state_E.MS_OUT_BIAS.ordinal();
				_module_state[i][idx_state].name = "OutputBias_";
				_module_state[i][idx_state].state_size  = num_out;
				_module_state[i][idx_state].short_name = "_OutBias_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width = 32;//length of the name "VariationOfbSubreservoir_0_Out_0"
				idx_state = module_state_E.MS_RESPONSIBILITY.ordinal();
				_module_state[i][idx_state].name = "Responsibility_";
				_module_state[i][idx_state].state_size  = num_out;
				_module_state[i][idx_state].short_name = "_Responsibility_";
				_module_state[i][idx_state].idx_related = -1;
				_module_state[i][idx_state].min_width = 32;//length of the name "VariationOfbSubreservoir_0_Out_0"
			}

			//compute a total number of columns to be saved
			_num_sub_cols = calculateNumCol(_esn_state);
			for(i=0; i<_module_state.length; i++)
			{
				_num_sub_cols += calculateNumCol(_module_state[i]);
			}//for module states
		}
		
		/**
		 * The function creates a set of column names for states of a provided array and adds these names to a provided
		 * array of column names.
		 * 
		 * @param field_names: array of column names
		 * @param state: array of states
		 */
		private void addColumnNames(Vector<Object> field_names, state_var_C[] state)
		{
			int i, j, k;
			int idx_related;//index of the related state variable
			int num_cols, num_cols_relate;//number of sub-columns of the primary and of the related state variable
			int cnt_obj;//counter of stored column headers
			int resulting_width;//resulting width of a column width
			String name;//name of a single column
			
			cnt_obj = field_names.size();
			for(i=0; i<state.length; i++)
			{
				//check the number of sub-fields
				num_cols = state[i].state_size;

				if(num_cols!=-1)//"-1" indicates not to save a certain column
				{
					if(num_cols==0)
					{
						name = state[i].name;
						field_names.add(name);
						//obtain a resulting width of a column
						if(state[i].min_width > name.length())
						{
							resulting_width = state[i].min_width;
						}
						else
						{
							resulting_width = name.length();
						}
						_width_cur [cnt_obj] = resulting_width;
						cnt_obj++;
					}
					else
					{
						//create column names for the state variables which do not have related variables
						if(state[i].idx_related==-1)
						{
							for(j=0; j<num_cols; j++)
							{
								name = state[i].name;
								name+= j;
								field_names.add(name);
								//obtain a resulting width of a column
								if(state[i].min_width > name.length())
								{
									resulting_width = state[i].min_width;
								}
								else
								{
									resulting_width = name.length();
								}
								_width_cur [cnt_obj] = resulting_width;
								cnt_obj++;
							}
						}
						else//create column names for the state variables which have related variables
						{
							idx_related = state[i].idx_related;
							num_cols_relate = state[idx_related].state_size;
							for(j=0; j<num_cols; j++)
							{
								for(k=0; k<num_cols_relate; k++)
								{
									name = state[i].name;
									name+= j;
									name+= state[idx_related].short_name;
									name+= k;
									field_names.add(name);
									//obtain a resulting width of a column
									if(state[i].min_width > name.length())
									{
										resulting_width = state[i].min_width;
									}
									else
									{
										resulting_width = name.length();
									}
									_width_cur [cnt_obj] = resulting_width;
									cnt_obj++;
								}
							}
						}
					}//number of columns for the current state
				}//must the current state be saved?
			}//for state length
		}
		
		/**
		 * The function adds the values of the provided LINEAR array as strings to the provided array of objects.
		 * The values are added starting with the provided index.
		 * The function also checks whether the number of values matches the number of available sub-columns.
		 * @param obj: array where the values should be added
		 * @param val: array of values
		 * @param idx_start: index of aray's element where the 1st value is saved
		 * @param state_var: state variable whose values are added
		 * @return: array of objects with the added values
		 */
		private Object[] addValuesToObjects(Object[] obj, double[] val, int idx_start, state_var_C state_var)
		{
			int i;
			int idx_str;//index where current string is saved 
			String str_val;//string from the current value
			
			//check the input
			if(val.length!=state_var.state_size)
			{
				System.err.println("addValuesToObjects: number of values does not match number of sub-columns");
				System.exit(1);
			}
			
			//assign the values as strings
			idx_str = idx_start;
			for(i=0; i<val.length; i++)
			{
				str_val = "" + val[i];
				obj[idx_str] = str_val; idx_str++;
			}
			
			return obj;
		}
		
		/**
		 * The function calculates a number of columns that are necessary to output all states of a provided array.
		 * 
		 * @param state: provided array of states
		 * @return: number of columns to save for the provided array
		 */
		private int calculateNumCol(state_var_C[] state)
		{
			int i;
			int idx_related;//index of a related attribute
			int num_cols_prime;//number of sections according to a prime attribute
			int num_cols_rel;//number of columns according to a related attribute
			int num_cols;//output variable
			
			num_cols = 0;
			for(i=0; i<state.length; i++)
			{
				if(state[i].state_size!=-1)
				{
					if(state[i].state_size==0)
					{
						num_cols_prime = 1;
					}
					else
					{
						num_cols_prime = state[i].state_size;
					}					
					if(state[i].idx_related==-1)
					{
						num_cols_rel = 1;
					}
					else
					{
						idx_related = state[i].idx_related;
						if(state[idx_related].state_size==0)
						{
							num_cols_rel = 1;
						}
						else
						{
							num_cols_rel = state[idx_related].state_size;
						}
					}
					num_cols+=(num_cols_prime*num_cols_rel);
				}//if a state is to be saved
			}//for i
			
			return num_cols;
		}
		
		/**
		 * The function produces a string for the submitted ESN state.
		 * @param time_step: time step where the ESN state was computed
		 * @param mode: aim of applying the ESN: "washout", "train", "test"
		 * @param input: states of the input neurons
		 * @param in_target: target values of the input neurons from the sequence
		 * @param res: states of reservoir neurons of each ESN module
		 * @param ofb_sub: OFB from all output neurons to compute states of the reservoir neurons in each sub-reservoir
		 * @param var_ofb_sub: variation of the OFB of the sub-reservoirs for all output neurons
		 * @param var_fit: variance of fitness in all populations
		 * @param var_individ: variance of individuals in all populations
		 * @param num_sort: number of sorted individuals in all popuations
		 * @param sub: squared array with the computed outputs of the sub-reservoirs for each output neuron
		 * @param output: states of the output neurons
		 * @param sub_target: squared array with the target outputs of the sub-reservoirs for each output neuron
		 * @param out_target: target states of the output neurons
		 * @param out_bias: squared array with the output bias of each output neuron in each module
		 * @param responsibility_sub: array with responsibility of modules at each output
		 * @return: string with the data of the provided ESN state
		 */
		private String convertEsnStateToString(int time_step, String mode, double[] input, double[] in_target, double[][] res,
				                               double[][] ofb_sub, double[][] var_ofb_sub, double[] var_fit,
				                               double[] var_individ, int[] num_sort, double[][] sub, double[] output,
				                               double[][] target_sub, double[] out_target, double[][] out_bias,
				                               double[][] responsibility_sub)
		{
			int i;
			int idx_str;//index of currently saved string
			int num_sub;//number of ESN modules
			String   str_val;//string from the current value
			Object[] obj_all_val;//array of strings of all provided values
			String   str_out;//output string
			state_var_C state;//temporary variable
			double[] tmp_state;//temporary array with values of currently saved state
			
			obj_all_val = new Object[_num_sub_cols];
			idx_str = 0;
			
			/*** mESN ***/
			
			//convert all provided values to the strings
			for(esn_state_E cur_esn_state : esn_state_E.values())
			{
				tmp_state = null;
				//assign an array to be saved
				switch(cur_esn_state)
				{
					case ES_STEP:
						str_val = "";
						str_val += time_step;//" "" + " is needed to convert "int" to "String" without the type Integer
						obj_all_val[idx_str] = str_val; idx_str++;
						break;
					case ES_MODE:
						obj_all_val[idx_str] = mode; idx_str++;
						break;
					case ES_IN:
						tmp_state = input;
						break;
					case ES_IN_TARGET:
						tmp_state = in_target;
						break;
					case ES_OUT:
						tmp_state = output;
						break;
					case ES_OUT_TARGET:
						tmp_state = out_target;
						break;
					default:
						System.err.println("ExpOutput.convertEsnStateToString: unknown ESN state");
						break;
				}
				
				//do not consider time step and mode, because they have non-double values
				if(cur_esn_state!=esn_state_E.ES_STEP && cur_esn_state!=esn_state_E.ES_MODE)
				{
					state = _esn_state[cur_esn_state.ordinal()];
					if(state.state_size!=-1)//"-1" shows that the column should not be saved
					{
						obj_all_val = addValuesToObjects(obj_all_val, tmp_state, idx_str, state);
						idx_str+=state.state_size;
					}
				}
			}
			
			/*** ESN modules ***/
			num_sub = res.length;
			for(i=0; i<num_sub; i++)
			{
				for(module_state_E cur_state : module_state_E.values())
				{
					tmp_state = null;
					//assign an array to be saved
					switch(cur_state)
					{
						case MS_RES:
							tmp_state = res[i];
							break;
						case MS_VAR_OFB:
							tmp_state = var_ofb_sub[i];
							break;
						case MS_VAR_FIT:
							//it is supposed that fitness is a scalar value
							tmp_state = new double[1];
							tmp_state[0] = var_fit[i];
							break;
						case MS_VAR_IND:
							//at the moment, variance of individuals is a scalar value
							//It is computed as an average over variances of all elements of an individual.
							tmp_state = new double[1];
							tmp_state[0] = var_individ[i];
							break;
						case MS_NUM_SRT:
							state = _module_state[i][cur_state.ordinal()];
							if(state.state_size!=-1)//"-1" shows that the column should not be saved
							{
								str_val = "";
								str_val += num_sort[i];//" "" + " is needed to convert "int" to "String" without the type Integer
								obj_all_val[idx_str] = str_val; idx_str++;
							}
							break;
						case MS_OFB:
							tmp_state = ofb_sub[i];
							break;
						case MS_OUT:
							tmp_state = sub[i];
							break;
						case MS_TARGET:
							tmp_state = target_sub[i];
							break;
						case MS_OUT_BIAS:
							tmp_state = out_bias[i];
							break;
						case MS_RESPONSIBILITY:
							tmp_state = responsibility_sub[i];
							break;
						default:
							System.err.println("ExpOutput.convertEsnStateToString: unknown module state");
							break;
					}
					
					//do not consider time step and mode, because they have non-double values
					if(cur_state!=module_state_E.MS_NUM_SRT)
					{
						state = _module_state[i][cur_state.ordinal()];
						if(state.state_size!=-1)//"-1" shows that the column should not be saved
						{
							obj_all_val = addValuesToObjects(obj_all_val, tmp_state, idx_str, state);
							idx_str+=state.state_size;
						}
					}
				}//for "cur_state"
			}//for "i"
			
			//check required widths of the strings to save at the current time step
			//increase the current widths of the columns, if it is necessary
			for(i=0; i<_width_cur.length; i++)
			{
				if(obj_all_val[i].toString().length() > _width_cur[i])
				{
					_width_cur[i] = obj_all_val[i].toString().length();
				}
			}
			
			//build a string with values of the provided ESN state
			str_out = "";
			for(i=0; i<obj_all_val.length; i++)
			{
				str_val  = convertObjectToString(obj_all_val[i]); 
				str_out += alignStr(str_val, 'l', _width_cur[i]+1, ' ');
			}
			str_out+="\n";
			
			return str_out;
		}
		
		/**
		 * The function creates a string of the file header for saving the ESN states.
		 * @return: string of the file header for saving the ESN states
		 */
		private String getStateHeader()
		{
			int i,j;
			String name;//name of a single column
			Vector<Object> field_names;//vector of the field names in the file with the ESN states
			String header;//output string
			
			field_names = new Vector<Object>(0, 1);
			_width_cur  = new int[_num_sub_cols];
			
			//create a list of column names
			addColumnNames(field_names, _esn_state);
			for(i=0; i<_module_state.length; i++)
			{
				//add a module index to each column name
				for(j=0; j<_module_state[i].length; j++)
				{
					name = "Module_" + i;
					name+= "_";
					name+= _module_state[i][j].name;
					_module_state[i][j].name = name;
				}
				addColumnNames(field_names, _module_state[i]);
			}

			//align the field names and add them to the header
			header = "";
			for(i=0; i<field_names.size(); i++)
			{
				//align and append one additional separating space
				name = convertObjectToString(field_names.get(i));
				name = alignStr(name, 'l', _width_cur[i]+1, ' ');
				header+= name;
			}
			header+="\n";
			
			return header;
		}
		
		/**
		 * The function saves the provided ESN states in the file which is specified in the provided object for the state
		 * output. This object specifies the attributes of the file format as well.
		 * 
		 * @param time_step: index of the time step where the ESN states were computed
		 * @param mode: mode of applying the ESN (washout, train, configuration or test)
		 * @param input: array with the states of the input neurons
		 * @param in_target: target values of the input neurons from the sequence
		 * @param res: array with the states of reservoir neurons for each ESN module
		 * @param ofb_sub: OFB from all output neurons to compute states of the reservoir neurons
		 * @param var_ofb_sub: variation of the OFB of the sub-reservoirs for all output neurons
		 * @param var_fit: variance of fitness in all populations
		 * @param var_individ: variance of individuals in all populations
		 * @param num_sort: number of sorted individuals in all populations
		 * @param sub: outputs of the sub-reservoirs for all output neurons
		 * @param output: ESN output
		 * @param target_sub: array of target outputs of the sub-reservoirs
		 * @param target: ESN target
		 * @param out_bias: array of output bias of output neurons in each module
		 * @param responsibility_sub: array with responsibility of modules at each output
		 */
		public void saveEsnState(int time_step, String mode, double[] input, double[] in_target, double[][] res, double[][] ofb_sub,
				                 double[][] var_ofb_sub, double[] var_fit, double[] var_individ, int[] num_sort,
				                 double[][] sub, double[] output, double[][] target_sub, double[] target,
				                 double[][] out_bias, double[][] responsibility)
		{
			String str_esn_state;//string with all ESN states to be saved
			
			str_esn_state = convertEsnStateToString(time_step, mode, input, in_target, res, ofb_sub, var_ofb_sub, var_fit,
					                                var_individ, num_sort, sub, output, target_sub, target, out_bias,
					                                responsibility);
			
			try{
				_bw.write(str_esn_state);
				_bw.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		/**
		 * The function terminates a saving of reservoir states.
		 */
		public void closeSavingEsnStates()
		{
			try{
				_bw.close();
				//assign a copy "ReadOnly" access
	            _file.setReadOnly();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	};
	
	/**
	 * This class is responsible for saving an ESN structure in a file.
	 * @author Danil Koryakin
	 *
	 */
	public class esn_output_C
	{
		/**
		 * The class keeps formating information for saving separate parts of an ESN.
		 * @author Danil Koryakin
		 */
		private class esn_part_C
		{
			private String name;//name of the primary state variable
			private String col_name;//array of column names
			private String row_name;//array of row names
			private Vector<String> comment;//comment to the ESN's part, for ex., explanation of structure of representation
		}
		
		private esn_part_C[] _esn_part;//array with formatting parameters of ESN parts to be saved
		private File _file;//file object to save ESN parts
		private BufferedWriter _bw;//buffered writer of a file to save ESN parts
		
		/**
		 * This is a constructor of the class "esn_output".
		 * It sets widths of fields for saving states of an ESN.
		 * The constructor opens a file for saving an ESN at provided path, only if the provided path is not "null".
		 * 
		 * @param filename: path to the file where an ESN should be saved;
		 *                  if it is "null" then no file should be opened for saving
		 */
		public esn_output_C(String filename)
		{
			int i;
			int idx_related;//index of the related state variable
			
			//define names for all ESN parts in the output file
			_esn_part = new esn_part_C[network_part_save_E.values().length];
			for(i=0; i<network_part_save_E.values().length; i++)
			{
				_esn_part[i] = new esn_part_C();
			}
			idx_related = network_part_save_E.MPS_OUTPUT_BIAS.ordinal();
			_esn_part[idx_related].name = "OUTPUT_BIAS";
			_esn_part[idx_related].col_name = "VALUE";
			_esn_part[idx_related].row_name = "OUTPUT_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("constant bias of the network's values");
			
			idx_related = network_part_save_E.MPS_MODULE_TYPE.ordinal();
			_esn_part[idx_related].name = "TYPE";
			_esn_part[idx_related].col_name = "VALUE";
			_esn_part[idx_related].row_name = "MODULE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("type of loaded module");
			
			idx_related = network_part_save_E.MPS_RESPONSIBILITY.ordinal();
			_esn_part[idx_related].name = "RESPONSIBILITY";
			_esn_part[idx_related].col_name = "VALUE";
			_esn_part[idx_related].row_name = "VALUE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			
			idx_related = network_part_save_E.MPS_RESPONSIBILITY_MIN_MAX.ordinal();
			_esn_part[idx_related].name = "MINIMUM_AND_MAXIMUM_OF_RESPONSIBILITY";
			_esn_part[idx_related].col_name = "RANGE";
			_esn_part[idx_related].row_name = "OUTPUT_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("allowed range to tune amplitudes of module's outputs");
			_esn_part[idx_related].comment.add("The left value is the smallest allowed amplitude.");
			_esn_part[idx_related].comment.add("The right value is the largest allowed amplitude.");

			idx_related = network_part_save_E.MPS_OUTPUT_BIAS.ordinal();
			_esn_part[idx_related].name = "OUTPUT_BIAS";
			_esn_part[idx_related].col_name = "VALUE";
			_esn_part[idx_related].row_name = "OUTPUT_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			
			idx_related = network_part_save_E.MPS_OUTPUT_BIAS_MIN_MAX.ordinal();
			_esn_part[idx_related].name = "MINIMUM_AND_MAXIMUM_OF_OUTPUT_BIAS";
			_esn_part[idx_related].col_name = "RANGE";
			_esn_part[idx_related].row_name = "OUTPUT_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("allowed range to tune amplitudes of module's output bias");
			_esn_part[idx_related].comment.add("The left value is the smallest allowed bias.");
			_esn_part[idx_related].comment.add("The right value is the largest allowed bias.");

			idx_related = network_part_save_E.MPS_SEED.ordinal();
			_esn_part[idx_related].name = "SEEDING_VALUE_FOR_RANDOM_GENERATORS";
			_esn_part[idx_related].col_name = "VALUES";
			_esn_part[idx_related].row_name = "MODULE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("value is used for evolutionary synchronization of mESN");
			
			idx_related = network_part_save_E.MPS_SEQ_PARAM.ordinal();
			_esn_part[idx_related].name = "PARAMETERS_OF_TRAINING_SEQUENCE";
			_esn_part[idx_related].col_name = "PARAMETERS";
			_esn_part[idx_related].row_name = "DYNAMICS_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			
			idx_related = network_part_save_E.EPS_IN_W.ordinal();
			_esn_part[idx_related].name = "INPUT_WEIGHTS";
			_esn_part[idx_related].col_name = "INPUT_NEURON";
			_esn_part[idx_related].row_name = "RES_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_IN_ACT.ordinal();
			_esn_part[idx_related].name = "ACTIVATION_OF_INPUT_NEURONS";
			_esn_part[idx_related].col_name = "ACTIVATIONS";
			_esn_part[idx_related].row_name = "IN_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_IN_MIN_MAX.ordinal();
			_esn_part[idx_related].name = "VALID_RANGES_OF_INPUT_NEURONS";
			_esn_part[idx_related].col_name = "INTERVAL";
			_esn_part[idx_related].row_name = "NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("Dummies (-,-) are placeholders to save valid ranges as a matrix.");
			
            idx_related = network_part_save_E.EPS_RES_W_INIT.ordinal();
            _esn_part[idx_related].name = "RESERVOIR_INIT_WEIGHTS";
            _esn_part[idx_related].col_name = "RESERVOIR_NEURON";
            _esn_part[idx_related].row_name = "RES_NEURON_";
            _esn_part[idx_related].comment  = new Vector<String>(0,1);
            _esn_part[idx_related].comment.add("none");
            idx_related = network_part_save_E.EPS_RES_ACT.ordinal();
			_esn_part[idx_related].name = "ACTIVATION_OF_RESERVOIR_NEURONS";
			_esn_part[idx_related].col_name = "ACTIVATIONS";
			_esn_part[idx_related].row_name = "RES_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_RES_BIAS.ordinal();
			_esn_part[idx_related].name = "BIAS_OF_RESERVOIR_NEURONS";
			_esn_part[idx_related].col_name = "BIASES";
			_esn_part[idx_related].row_name = "RES_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_RES_LEAKAGE_RATE.ordinal();
			_esn_part[idx_related].name = "LEAKAGE_RATE_OF_RESERVOIR_NEURONS";
			_esn_part[idx_related].col_name = "LEAKAGE_RATE";
			_esn_part[idx_related].row_name = "RES_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_RES_SIZE.ordinal();
			_esn_part[idx_related].name = "MODULE_SIZES";
			_esn_part[idx_related].col_name = "SIZES";
			_esn_part[idx_related].row_name = "MODULE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_RES_SR.ordinal();
			_esn_part[idx_related].name = "SPECTRAL_RADII_OF_RESERVOIR_MODULES";
			_esn_part[idx_related].col_name = "SPECTRAL_RADII";
			_esn_part[idx_related].row_name = "MODULE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_RES_CONNECT.ordinal();
			_esn_part[idx_related].name = "CONNECTIVITY_OF_MODULES";
			_esn_part[idx_related].col_name = "MODULE";
			_esn_part[idx_related].row_name = "MODULE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("On the main diagonal there are values of exclusive connectivity.");
			_esn_part[idx_related].comment.add("Outside the main diagonal there is connectivity in overlaps between the modules.");
			idx_related = network_part_save_E.EPS_RES_TOPOLOGY.ordinal();
			_esn_part[idx_related].name = "RESERVOIR_TOPOLOGY";
			_esn_part[idx_related].col_name = "VALUES";
			_esn_part[idx_related].row_name = "VALUE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_RES_CONNECT_MTRX.ordinal();
			_esn_part[idx_related].name = "CONNECTIVITY_MATRIX_OF_RESERVOIR";
			_esn_part[idx_related].col_name = "END_NEURON";
			_esn_part[idx_related].row_name = "BEGIN_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_RES_MIN_MAX.ordinal();
			_esn_part[idx_related].name = "VALID_RANGES_OF_RESERVOIR_NEURONS";
			_esn_part[idx_related].col_name = "INTERVAL";
			_esn_part[idx_related].row_name = "NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("Dummies (-,-) are placeholders to save valid ranges as a matrix.");
			
            idx_related = network_part_save_E.EPS_OUT_W.ordinal();
			_esn_part[idx_related].name = "OUTPUT_WEIGHTS";
			_esn_part[idx_related].col_name = "END_NEURON";
			_esn_part[idx_related].row_name = "OUT_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("first END neurons are reservoir neurons;");
			_esn_part[idx_related].comment.add("next END neurons are ESN's output neurons (self-recurrent connections);");
			_esn_part[idx_related].comment.add("final END neurons are input neurons");
			idx_related = network_part_save_E.EPS_OUT_ACT.ordinal();
			_esn_part[idx_related].name = "ACTIVATION_OF_OUTPUT_NEURONS";
			_esn_part[idx_related].col_name = "ACTIVATIONS";
			_esn_part[idx_related].row_name = "OUT_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_OUT_BIAS.ordinal();
			_esn_part[idx_related].name = "BIAS_OF_OUTPUT_NEURONS";
			_esn_part[idx_related].col_name = "BIAS";
			_esn_part[idx_related].row_name = "OUT_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			idx_related = network_part_save_E.EPS_OUT_MIN_MAX.ordinal();
			_esn_part[idx_related].name = "MINIMUM_AND_MAXIMUM_OF_OUTPUT_NEURONS";
			_esn_part[idx_related].col_name = "VALUES";
			_esn_part[idx_related].row_name = "VALUE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("The left value is the smallest value.");
			_esn_part[idx_related].comment.add("The right value is the largest value.");
			
			idx_related = network_part_save_E.EPS_OFB_W.ordinal();
			_esn_part[idx_related].name = "OFB_WEIGHTS";
			_esn_part[idx_related].col_name = "OUTPUT_NEURON";
			_esn_part[idx_related].row_name = "RES_NEURON_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			
			idx_related = network_part_save_E.SPS_PARAM_SIZE.ordinal();
			_esn_part[idx_related].name = "MODULE_NUM_PARAM";
			_esn_part[idx_related].col_name = "NUM_PARAM";
			_esn_part[idx_related].row_name = "MODULE_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("Module size has a meaning of a number of hidden states to be tuned. Here it is 2: a frequency and a phase.");
			
			idx_related = network_part_save_E.SPS_PARAM_VALUE.ordinal();
			_esn_part[idx_related].name = "PARAM_VALUE";
			_esn_part[idx_related].col_name = "VALUE";
			_esn_part[idx_related].row_name = "PARAM_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("VALUE_0 is for frequency, VALUE_1 is for phase");
			
			idx_related = network_part_save_E.SPS_PARAM_MIN_MAX.ordinal();
			_esn_part[idx_related].name = "MINIMUM_AND_MAXIMUM_OF_PARAM";
			_esn_part[idx_related].col_name = "VALUES";
			_esn_part[idx_related].row_name = "PARAM_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("The left value is the smallest value.");
			_esn_part[idx_related].comment.add("The right value is the largest value.");
			_esn_part[idx_related].comment.add("VALUE_0 is for frequency");
			_esn_part[idx_related].comment.add("VALUE_1 is for phase");
			_esn_part[idx_related].comment.add("MAX for phase is a dummy value because it depends on the current frequency as MAX = period = (2*PI/FREQ_cur)");
			
			idx_related = network_part_save_E.SPS_OUT_MIN_MAX.ordinal();
			_esn_part[idx_related].name = "MINIMUM_AND_MAXIMUM_OF_OUTPUT";
			_esn_part[idx_related].col_name = "INTERVAL";
			_esn_part[idx_related].row_name = "OUTPUT_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("smallest and largest values of a sine wave");
			_esn_part[idx_related].comment.add("The left value is the smallest value.");
			_esn_part[idx_related].comment.add("The right value is the largest value.");
			
			idx_related = network_part_save_E.FPS_IN_NORM_COEF.ordinal();
			_esn_part[idx_related].name = "INPUT_NORMALIZATION";
			_esn_part[idx_related].col_name = "VALUE";
			_esn_part[idx_related].row_name = "INPUT_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");
			
			idx_related = network_part_save_E.FPS_OUT_NORM_COEF.ordinal();
			_esn_part[idx_related].name = "OUTPUT_NORMALIZATION";
			_esn_part[idx_related].col_name = "VALUE";
			_esn_part[idx_related].row_name = "OUTPUT_";
			_esn_part[idx_related].comment  = new Vector<String>(0,1);
			_esn_part[idx_related].comment.add("none");

			if(filename!=null)
			{
				_file = new File(filename);
				try
				{
					_bw = new BufferedWriter(new FileWriter(_file));
				}
				catch(IOException e)
				{
					System.err.println("esn_output_C: cannot open a file for saving an ESN");
					System.exit(1);
				}
			}
			else
			{
				_bw = null;
			}
		}
		
		/**
		 * The function splits each string of the provided array into substrings. These substrings should be divided
		 * by spaces.
		 * The obtained substrings are stored as alone strings in a row of a rectangular array of strings.
		 * 
		 * @param str_in: input array of strings
		 * @return: rectangular array of strings
		 */
		private String[][] parseMatrix(Vector<String> str_in)
		{
			int i;
			String[][] str_out;//output array of strings
			
			str_out = new String[str_in.size()][];
			
			for(i=0; i<str_in.size(); i++)
			{
				str_out[i] = str_in.get(i).split(" ");
				
				//check a number of elements in each row of the output matrix
				if(i > 0)
				{
					if(str_out[i].length != str_out[i-1].length)
					{
						System.err.println("parseMatrix: unequal number of elements in different rows");
						System.exit(1);
					}
				}
			}
			
			return str_out;
		}
		
		public void closeFile()
		{
			try{
				_bw.close();
				//assign a copy "ReadOnly" access
	            _file.setReadOnly();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		/**
		 * The function indicates whether a file is opened for saving. 
		 * 
		 * @return: "true": if a file is opened for saving
		 *          "false": otherwise
		 */
		public boolean isSavingRequired()
		{
			boolean is_saving_required;//output variable
			
			if(_bw!=null)
			{
				is_saving_required = true;
			}
			else
			{
				is_saving_required = false;
			}
			
			return is_saving_required;
		}
		
		/**
		 * The function saves a header of an ESN file.
		 * 
		 * @param idx_run: index of a run where an ESN was processed.
		 */
		public void saveEsnHeader(int idx_run)
		{
			Vector<String> out_header;//output file header with added run specific parameters
			
			//save file header
			out_header = createFileHeader(header_type_E.HT_RUN_DATA, idx_run);
			saveFileHeader(_bw, out_header);
		}
		
		/**
		 * The function extracts a specified part of a specified ESN module from a provided set of strings.
		 * The ESN part is specified by its identifier.
		 * The ESN module is specified by its index.
		 * The extracted values are returned as an array of strings where values of the same row are separated
		 * with spaces.
		 * A row with the column names is not included in the output data.
		 * A column with the row names is not included in the output data either.
		 * 
		 * @param esn_data: loaded ESN data without comments
		 * @param esn_part: identifier of an ESN part
		 * @param module_idx: specified module index
		 * @return: set of strings with values of the specified part
		 */
		public Vector<String> getEsnPart(Vector<String> esn_data, network_part_save_E esn_part, int module_idx)
		{
			int i;
			int idx_name;//index of a string with a name of an ESN part
			int idx_space;//index of a space in a string
			String part_name;//name of an ESN part to be extracted
			String str_tmp;//temporary string
			Vector<String> str_vals;//output variables
			
			str_vals = new Vector<String>(0, 1);
			
			//find the beginning of the specified ESN part
			part_name = _esn_part[esn_part.ordinal()].name + "_MODULE_" + module_idx;
			idx_name = esn_data.indexOf(part_name);
			
			if(idx_name!=-1)
			{
				i = idx_name + 2;//"+2" in order to get rid of a name of the ESN part 
				str_tmp = esn_data.get(i);
				
				while(str_tmp.isEmpty()==false)//empty string must be at the end of each ESN part
				{
					//remove a name of the row
					idx_space = str_tmp.indexOf(" ");
					str_tmp = str_tmp.substring(idx_space+1);
					
					//remove possible spaces at the beginning and at the end
					str_tmp = str_tmp.trim();
					
					//store the string for the output
					str_vals.add(str_tmp);

					i++;
					//previously analyzed string could be the last one in the provided array
					if(i < esn_data.size())
					{
						str_tmp = esn_data.get(i);
					}
					else
					{
						str_tmp = "";
					}
				}
			}
			
			return str_vals;
		}
		
		/**
		 * The function formats and saves data of a provided ESN part for a specified ESN module.
		 * 
		 * The input data are provided in form of strings.
		 * The function splits each string into an array of values and stores them in a single row.
		 * All columns with columns of the output array are aligned to the same width.
		 * 
		 * NAME of OUTPUT MATRIX: After the formating the function adds a name to the output matrix.
		 * A module index is added to this name unless is "-1". "-1" must be provided for general parameters
		 * of an mESN like module responsibilities.
		 * The name is saved as the 1st string of the output array.
		 * 
		 * Comment strings are added between the name and an array of values.
		 * 
		 * @param part_data: values of the specified ESN part that are provided by strings
		 * @param esn_part: indicator that specifies a part for which the values are provided
		 * @param sub_idx: module index
		 */
		public void saveEsnPart(Vector<String> part_data, network_part_save_E esn_part, int sub_idx)
		{
			int i,j;
			int idx_row;//index of currently assigned row of the output matrix
			int width_col;//column width 
			int num_row_out;//number of output rows
			String     str_to_save;//string to be saved
		    String[][] str_vals;//rectangular array of values
		    String[][] str_part;//values of specified ESN part in a formatted form
		    Object[]   obj_val;//array of values to be aligned and then assigned to the output
			
		    str_vals = parseMatrix(part_data);

		    //"+1": one string with a name of the part
		    //"+1": one string with titles of the columns
		    //"+ number of comments": several comment strings
		    //"+ number of value rows": number of rows in the matrix with values
		    num_row_out = 1 + 1 + _esn_part[esn_part.ordinal()].comment.size() + str_vals.length;
		    
		    //prepare the output
		    str_part = new String[num_row_out][];
		    
		    //assign the name of the part
		    idx_row = 0;
			str_part[0] = new String[1];
			if(sub_idx!=-1)
			{
				str_part[0][0] = _esn_part[esn_part.ordinal()].name + "_MODULE_" + sub_idx;
			}
			else
			{
				str_part[0][0] = _esn_part[esn_part.ordinal()].name;
			}
			idx_row++;
			
			//assign the comment strings
			for(i=0; i<_esn_part[esn_part.ordinal()].comment.size(); i++)
			{
				str_part[idx_row] = new String[1];
				str_part[idx_row][0] = "# " + _esn_part[esn_part.ordinal()].comment.get(i);
				idx_row++;
			}
			
			//allocate a row with titles of columns
			//"+1" is for an additional column with names of rows
			str_part[idx_row] = new String[str_vals[0].length+1];

			//allocate an array for a value matrix by row
			for(i=0; i<str_vals.length; i++)
			{
				//left side: "+1" is because a row with column names is assigned above
				//right side: "+1" is for an additional column with names of rows
				str_part[idx_row+i+1] = new String[str_vals[i].length+1];
			}
			//assign a column with names of rows
			obj_val    = new Object[str_vals.length+1];
			obj_val[0] = "";
			for(i=0; i<str_vals.length; i++)
			{
				//"+1" because the 0-th element is already assigned
				obj_val[1+i] = _esn_part[esn_part.ordinal()].row_name + i;
			}
			width_col = findColWidth(convertObjectToString(obj_val[0]), obj_val);
			for(i=0; i<str_vals.length+1; i++)
			{
				str_part[idx_row+i][0] = alignStr(convertObjectToString(obj_val[i]), 'l', width_col, ' ');
			}
			//assign columns with values
			for(i=0; i<str_vals[0].length; i++)
			{
				//add a column index, if there are several columns
				if(str_vals[0].length > 1)
				{
					obj_val[0] = _esn_part[esn_part.ordinal()].col_name + "_" + i;
				}
				else
				{
					obj_val[0] = _esn_part[esn_part.ordinal()].col_name + ":";
				}
				for(j=0; j<str_vals.length; j++)
				{
					//"+1" because the 0-th element is already assigned
					obj_val[1+j] = str_vals[j][i];
				}
				width_col = findColWidth(convertObjectToString(obj_val[0]), obj_val);
				//"++" to get a space before values
				width_col++;
				for(j=0; j<str_vals.length+1; j++)
				{
					//"+1" because the 0-th column with names of rows is already assigned
					str_part[idx_row+j][1+i] = alignStr(convertObjectToString(obj_val[j]), 'r', width_col, ' ');
				}
			}
			
			try{
				_bw.write("\n");
				for(i=0; i<str_part.length; i++)
				{
					str_to_save = "";
					for(j=0; j<str_part[i].length; j++)
					{
						str_to_save += str_part[i][j];
					}
					str_to_save += "\n";
					_bw.write(str_to_save);
					_bw.flush();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	};

	private final String _gen_exp_param_dir_name = "param_vector";//general name of directory used when values are provided for more than 1 parameter
	
	private String _prj_name;//name of current project containing a current revision
	private String _java_revision;//Java revision which was installed as an experiment was running
	private ExpParam _exp_param;//object which keeps values of experiment parameters

	private String[] _out_name_param_val;//name of output directories and files for current parameter value
	private String[] _out_dir_param;//list of directories to save data after runs with corresponding parameter values
	private String _dir_out_main;//output directory
	private String _out_file_vectors;//name of the output file with a list of experiment parameter vectors 

	private stat_data_C[] _data_exp;//this object keeps statistics for all experiment parameters 
	private data_run_C    _data_run;//object with all performance indicators on all sequence intervals for all runs
	
	public ExpOutput(ExpParam exp_param)
	{
		int  i;
		int  num_runs;//number of runs per setting of experiment parameters
		File file_dir;//file object of working directory
		StringTokenizer st;
		String tmp_str;
		String path_out;
		vector_C seq_purpose;//array with purposes for applying ESN to the sequence on each interval
		
		_exp_param = exp_param;

		num_runs = (Integer)_exp_param.getParamVal(exp_param_E.EP_PERFORM_RUNS, req_val_E.RV_CUR);
		seq_purpose = (vector_C)_exp_param.getParamVal(exp_param_E.EP_PERFORM_SEQ_PURPOSE, req_val_E.RV_CUR);
		
		//allocate storage of the run statistics and assign its fields
		_data_run = new data_run_C(num_runs, seq_purpose); 
		
		//allocate storage of the experiment statistics and assign its fields
		_data_exp = new stat_data_C[stat_exp_E.values().length];
		
		//assign a flag which value is an integer
		_data_exp[stat_exp_E.SE_MSE_AVG_TST.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_MSE_TST.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MIN_MSE_TST.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_IDX_MIN_MSE_TST.ordinal()   ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());		
		_data_exp[stat_exp_E.SE_MAX_MSE_TST.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_IDX_MAX_MSE_TST.ordinal()   ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MED_MSE_TST.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_IDX_MED_MSE_TST.ordinal()   ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_NRMSE_AVG_TST.ordinal()     ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_NRMSE_TST.ordinal()     ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_RMSE_AVG_TST.ordinal()      ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_RMSE_TST.ordinal()      ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MSE_AVG_TRN.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_MSE_TRN.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_NRMSE_AVG_TRN.ordinal()     ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_NRMSE_TRN.ordinal()     ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_RMSE_AVG_TRN.ordinal()      ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_RMSE_TRN.ordinal()      ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MIN_NRMSE_TST.ordinal()     ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_IDX_MIN_NRMSE_TST.ordinal() ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MAX_NRMSE_TST.ordinal()     ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_IDX_MAX_NRMSE_TST.ordinal() ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MED_NRMSE_TST.ordinal()     ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_IDX_MED_NRMSE_TST.ordinal() ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_RUN_REPEAT_TOTAL.ordinal()  ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_SEL_SUM_TST.ordinal()       ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_SEL_TST.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());      
		_data_exp[stat_exp_E.SE_BEST_SEL_TST.ordinal()      ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());     
		_data_exp[stat_exp_E.SE_IDX_BEST_SEL_TST.ordinal()  ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_SEL_SUM_TRN.ordinal()       ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_SEL_TRN.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());      
		_data_exp[stat_exp_E.SE_BEST_SEL_TRN.ordinal()      ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());     
		_data_exp[stat_exp_E.SE_IDX_BEST_SEL_TRN.ordinal()  ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MSE_AVG_CFG.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MIN_MSE_CFG.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_IDX_MIN_MSE_CFG.ordinal()   ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_SEL_AVG_CFG.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_LEL_SUM_CFG.ordinal()       ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_DEV_LEL_CFG.ordinal()       ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());      
		_data_exp[stat_exp_E.SE_BEST_LEL_CFG.ordinal()      ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());     
		_data_exp[stat_exp_E.SE_IDX_BEST_LEL_CFG.ordinal()  ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_MAX_SEL_CFG.ordinal()       ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());     
		_data_exp[stat_exp_E.SE_IDX_MAX_SEL_CFG.ordinal()   ] = new stat_data_C(true,  false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_CIOK_AVG_BY_CFG_ERR.ordinal()      ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_CIOK_AVG_CFG.ordinal()] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		_data_exp[stat_exp_E.SE_CIOK_AVG_BY_TST_ERR.ordinal()      ] = new stat_data_C(false, false, true, _exp_param.getNumParamVectors());
		
		//assign names of separate values
		_data_exp[stat_exp_E.SE_MSE_AVG_TST.ordinal()       ].name = "avg_MSE_test";
		_data_exp[stat_exp_E.SE_DEV_MSE_TST.ordinal()       ].name = "Dev_mse_test";		
		_data_exp[stat_exp_E.SE_MIN_MSE_TST.ordinal()       ].name = "min_MSE_test";
		_data_exp[stat_exp_E.SE_IDX_MIN_MSE_TST.ordinal()   ].name = "Idx_min_MSE_test";//index of run where the smallest test MSE was obtained
		_data_exp[stat_exp_E.SE_MAX_MSE_TST.ordinal()       ].name = "max_MSE_test";
		_data_exp[stat_exp_E.SE_IDX_MAX_MSE_TST.ordinal()   ].name = "Idx_max_MSE_test";//index of run where the largest test MSE was obtained
		_data_exp[stat_exp_E.SE_MED_MSE_TST.ordinal()       ].name = "median_MSE_test";
		_data_exp[stat_exp_E.SE_IDX_MED_MSE_TST.ordinal()   ].name = "Idx_median_MSE_test";//index of run where a median test MSE was obtained
		_data_exp[stat_exp_E.SE_NRMSE_AVG_TST.ordinal()     ].name = "avg_NRMSE_test";
		_data_exp[stat_exp_E.SE_DEV_NRMSE_TST.ordinal()     ].name = "Dev_nrmse_test";
		_data_exp[stat_exp_E.SE_RMSE_AVG_TST.ordinal()      ].name = "avg_RMSE_test";
		_data_exp[stat_exp_E.SE_DEV_RMSE_TST.ordinal()      ].name = "Dev_rmse_test";
		_data_exp[stat_exp_E.SE_MSE_AVG_TRN.ordinal()       ].name = "avg_MSE_train";
		_data_exp[stat_exp_E.SE_DEV_MSE_TRN.ordinal()       ].name = "Dev_mse_train";
		_data_exp[stat_exp_E.SE_NRMSE_AVG_TRN.ordinal()     ].name = "avg_NRMSE_train";
		_data_exp[stat_exp_E.SE_DEV_NRMSE_TRN.ordinal()     ].name = "Dev_nrmse_train";
		_data_exp[stat_exp_E.SE_RMSE_AVG_TRN.ordinal()      ].name = "avg_RMSE_train";
		_data_exp[stat_exp_E.SE_DEV_RMSE_TRN.ordinal()      ].name = "Dev_rmse_train";
		_data_exp[stat_exp_E.SE_MIN_NRMSE_TST.ordinal()     ].name = "min_NRMSE_test";
		_data_exp[stat_exp_E.SE_IDX_MIN_NRMSE_TST.ordinal() ].name = "Idx_min_NRMSE_test";//index of run where the smallest test NRMSE was obtained
		_data_exp[stat_exp_E.SE_MAX_NRMSE_TST.ordinal()     ].name = "max_NRMSE_test";
		_data_exp[stat_exp_E.SE_IDX_MAX_NRMSE_TST.ordinal() ].name = "Idx_max_NRMSE_test";//index of run where the largest test NRMSE was obtained
		_data_exp[stat_exp_E.SE_MED_NRMSE_TST.ordinal()     ].name = "median_NRMSE_test";
		_data_exp[stat_exp_E.SE_IDX_MED_NRMSE_TST.ordinal() ].name = "Idx_median_NRMSE_test";//index of run where a median test NRMSE was obtained
		_data_exp[stat_exp_E.SE_RUN_REPEAT_TOTAL.ordinal()  ].name = "Run_repeats_total";
		_data_exp[stat_exp_E.SE_SEL_SUM_TST.ordinal()       ].name = "sum_SEL_test";
		_data_exp[stat_exp_E.SE_DEV_SEL_TST.ordinal()       ].name = "Dev_SEL_test";
		_data_exp[stat_exp_E.SE_BEST_SEL_TST.ordinal()      ].name = "Best_SEL_test";
		_data_exp[stat_exp_E.SE_IDX_BEST_SEL_TST.ordinal()  ].name = "Idx_best_SEL_test";//index of run where the best test SEL was obtained
		_data_exp[stat_exp_E.SE_SEL_SUM_TRN.ordinal()       ].name = "sum_SEL_train";
		_data_exp[stat_exp_E.SE_DEV_SEL_TRN.ordinal()       ].name = "Dev_SEL_train";
		_data_exp[stat_exp_E.SE_BEST_SEL_TRN.ordinal()      ].name = "Best_SEL_train";
		_data_exp[stat_exp_E.SE_IDX_BEST_SEL_TRN.ordinal()  ].name = "Idx_best_SEL_train";//index of run where the best training SEL was obtained
		_data_exp[stat_exp_E.SE_MSE_AVG_CFG.ordinal()       ].name = "avg_MSE_config";
		_data_exp[stat_exp_E.SE_MIN_MSE_CFG.ordinal()       ].name = "min_MSE_config";
		_data_exp[stat_exp_E.SE_IDX_MIN_MSE_CFG.ordinal()   ].name = "Idx_min_MSE_config";//index of run where the smallest test MSE was obtained		
		_data_exp[stat_exp_E.SE_SEL_AVG_CFG.ordinal()       ].name = "avg_SEL_config";
		_data_exp[stat_exp_E.SE_LEL_SUM_CFG.ordinal()       ].name = "sum_LEL_config";
		_data_exp[stat_exp_E.SE_DEV_LEL_CFG.ordinal()       ].name = "Dev_LEL_config";
		_data_exp[stat_exp_E.SE_BEST_LEL_CFG.ordinal()      ].name = "Best_LEL_config";
		_data_exp[stat_exp_E.SE_IDX_BEST_LEL_CFG.ordinal()  ].name = "Idx_best_LEL_config";//index of run where the best config LEL was obtained
		_data_exp[stat_exp_E.SE_MAX_SEL_CFG.ordinal()       ].name = "max_SEL_config";
		_data_exp[stat_exp_E.SE_IDX_MAX_SEL_CFG.ordinal()   ].name = "Idx_max_SEL_config";//index of run where the largest config SEL was obtained
		_data_exp[stat_exp_E.SE_CIOK_AVG_BY_CFG_ERR.ordinal()].name = "Config_Identification_Rate_incl_Error(%)";//percentage of runs where target components were identified correctly according to both, the config error and the osci param list 
		_data_exp[stat_exp_E.SE_CIOK_AVG_CFG.ordinal()].name = "Config_Identification_Rate_without_Error(%)";//percentage of runs where target components were identified correctly only according to the osci parameter list
		_data_exp[stat_exp_E.SE_CIOK_AVG_BY_TST_ERR.ordinal()].name = "Amount_NotOutlier_Runs(%)";//percentage of runs where target components were identified correctly and the test error was small
		
		file_dir = new File(".");
		path_out = file_dir.getAbsolutePath();
		st = new StringTokenizer(path_out, File.separator);
		//extract name of current directory;
		//it is one level above "bin" if the binary has been started from the command line 
		tmp_str = "";
		while(st.hasMoreTokens() && tmp_str.matches("bin")==false)
		{
			_prj_name = tmp_str;
			tmp_str  = st.nextToken();
		}
		//extract a Java name
		_java_revision = System.getProperty("java.version");
		_out_dir_param = new String[_exp_param.getNumParamVectors()];
		_out_file_vectors = "param_vectors.dat";
		_out_name_param_val = new String[_exp_param.getNumParamVectors()];
		_out_name_param_val = createOutNameParamVal();
		
		//set parameters for the output
		_dir_out_main  = "." + File.separator + "output";
		//create separate output directories
		_exp_param.setFirstParamVector();
		for(i = 0; i < _out_name_param_val.length; i++)
		{
			//create a directory to save the output for current experiment parameter value
			_out_dir_param[i] = _dir_out_main + File.separator + _out_name_param_val[i];
			//set the next combination of values of experiment parameters
			_exp_param.setNextParamVector();
		}
	}
	
	/**
	 * add necessary number of specified symbols from the right, if the input string must be aligned to the left;
	 * otherwise; add a necessary number of specified symbols from the right 
	 * @param str, a string to be aligned
	 * @param lr, alignment request: 'l' for left alignment, 'r' for right alignment 
	 * @param total_len, total length of resulting string
	 * @param symbol, a symbol to be added
	 * @return
	 */
	private String alignStr(String str, char lr, int total_len, char symbol)
	{
		String str_out;
		
		if(str.length() < total_len)
		{
			int i;
			str_out = "";
			for(i=0; i<(total_len-str.length()); i++)
			{
				str_out += symbol;
			}
			//realize the given alignment
			if(lr=='r')
			{
				str_out += str;
			}
			else if(lr=='l')
			{
				str_out = str + str_out; 
			}
			else//do not realize an unknown alignment
			{
				str_out = str;
			}
		}
		else
		{
			str_out = str;
		}
		
		return str_out;
	}
	
	/**
	 * convert a specified value of a used type to a string
	 * @param obj, specified value
	 * @return, specified value as a string
	 */
	private String convertObjectToString(Object obj)
	{
		String   str_out;
		Class<?> param_class;
		
		//convert a parameter value to a text form
		param_class = obj.getClass();
		if(param_class==Double.class                   ||
		   param_class==Integer.class                  ||
		   param_class==Boolean.class                  ||
		   param_class==String.class                   ||
		   param_class==activation_E.class               ||
		   param_class==ReservoirInitialization.class  ||
		   param_class==noise_E.class                  ||
		   param_class==config_method_E.class          ||
		   param_class==distortion_E.class			   ||
		   param_class==config_ea_mode_E.class         ||
		   param_class==config_ea_init_E.class         ||
		   param_class==config_ea_order_method_E.class ||
		   param_class==leakage_assign_E.class)
		{
			str_out = obj.toString();
		}
		else if(param_class==interval_C.class)
		{
			str_out = ((interval_C)obj).toString();
		}
		else if(param_class==vector_C.class)
		{
			str_out = ((vector_C)obj).toString();
		}
		else if(param_class==multi_val_C.class)
		{
			if(((multi_val_C)obj).isInterval()==true)
			{
				str_out = ((multi_val_C)obj).getInterval().toString();
			}
			else
			{
				str_out = ((multi_val_C)obj).toString();
			}
		}
		else
		{
			System.err.println("ExpOutput.convertObjectToString: submitted type of parameter values is not supported");
			System.exit(1);
			str_out = null;
		}
		
		return str_out;
	}
	
	/**
	 * create an output file header with values of used parameters as an array of strings
	 * @param header_type, type of the header to be created
	 * @param idx_run, used only, if run-specific data is added to the header
	 * @return, array of strings with the header information
	 */
	private Vector<String> createFileHeader(header_type_E header_type, Integer idx_run)
	{
		int i;
		int max_len;//maximum length of a parameter string
		Vector<String> param_all;//array with names of all parameters
		Vector<String> param_provided;//array with names of only provided parameters
		Vector<String> header;//created file header
		Vector<Object> param_val;//array of parameter values
		String param_name;//current parameter name
		String tmp_str;
		
		param_all      = _exp_param.getAllParamNames();
		param_provided = _exp_param.getProvidedParamNames();
		header         = new Vector<String>(0, 1);
		param_val      = new Vector<Object>(0, 1);
		
		//create the file header with only parameter names (without parameter values)
		for(i=0; i<param_all.size(); i++)
		{
			param_name = param_all.get(i);
			header.add("# " + param_name + ": ");
			//this is not a provided parameter
			if(param_provided.contains(param_name)==false)
			{
				param_val.add(_exp_param.getParamVal(param_name, req_val_E.RV_DEF));
			}
			else
			{
				switch(header_type)
				{
					case HT_EXP_STAT:
						//add a reference to the parameter values, only if there are several provided parameters
						//no reference is needed in the case of a single provided parameter, its value appears as
						//a column in the file with the statistics for the whole experiment and in the names of output
						//directories 
						if(param_provided.size() > 1)
						{
							tmp_str = "see in the file " + _out_file_vectors;
							param_val.add(tmp_str);
						}
						else if(param_provided.size()==1)
						{
							param_val.add(_exp_param.getParamVal(param_name, req_val_E.RV_CUR));
						}
						else
						{
							System.err.println("createFileHeader: no provided parameters");
							System.exit(1);
						}
					break;
					case HT_RUN_STAT:
					case HT_RUN_DATA:
						param_val.add(_exp_param.getParamVal(param_name, req_val_E.RV_CUR));
					break;
					default:
						System.err.println("createFileHeader: unknown header type");
						System.exit(1);
					break;
				}
			}
		}//for i
		//fields with additional parameters independent on the header type
		header.add   ("# PROJECT NAME: ");
		param_val.add(_prj_name);
		header.add   ("# JAVA REVISION: ");
		param_val.add(_java_revision);
		header.add   ("# USER COMMENT: ");
		param_val.add(_exp_param.getUsrComment());
		//fields with additional parameters which are dependent on the header type
		switch(header_type)
		{
			case HT_EXP_STAT:
				//no additional parameters 
			break;
			case HT_RUN_STAT:
				//no additional parameters 
			break;
			case HT_RUN_DATA:
				header.add   ("# RANDOM SEED: ");
				tmp_str = _data_run.getSeed(idx_run);
				param_val.add(tmp_str);
			break;
			default:
				System.err.println("createFileHeader: unknown header type");
				System.exit(1);
			break;
		}
		//find the maximum length of a string with a parameter name
		max_len = 0;
		for(i=0; i<header.size(); i++)
		{
			if(max_len < header.get(i).length())
			{
				max_len = header.get(i).length();
			}
		}
		//align the parameter names and add parameter values
		for(i=0; i<header.size(); i++)
		{
			tmp_str = alignStr(header.get(i), 'l', max_len, ' ');
			tmp_str+= convertObjectToString(param_val.get(i));
			header.set(i, tmp_str);
		}
		
		return header;
	}
	
	/**
	 * create a specified directory
	 * @param path_dir, path to a directory to be created 
	 */
	private void createDir(String path_dir)
	{
		File path = new File(path_dir);//object with output file
		
		//create the output directory		
		if(path.exists()==false)//if there is no instance with this name then create a directory
		{
			path.mkdirs();
		}
		else
		{
			//rename a file with the same name as a name of the directory
			while(path.isFile()==true)
			{
				System.err.println("createDir: remove or rename file with the same name as output directory");
				System.exit(1);
			}
			path.mkdirs();
		}
	}
	
	/**
	 * create a common name for all output directories and files for current parameter value
	 * @return: list of common output file names for all combinations of experiment parameter values
	 */
	private String[] createOutNameParamVal()
	{
		int i;
		String[] out_list;//output array
		String tmp_str;
		Integer num_param_comb;//total number of experiment parameter combinations
		Vector<String> param_names;//names of parameters where values were provided
		Object param_val;//parameter value
		Class<?> param_class;//class of parameter value
		
		param_names = _exp_param.getProvidedParamNames();
		num_param_comb = _exp_param.getNumParamVectors();
		out_list = new String[num_param_comb];
		
		for(i=0; i<num_param_comb; i++)
		{
			//assign part of name without the parameter value
			if(param_names.size()==1)//is there a single parameter?
			{
				out_list[i] = param_names.firstElement();
			}
			else
			{
				out_list[i] = _gen_exp_param_dir_name;
			}
			out_list[i]+= "_";
			
			//add parameter value in case of a single loaded parameter;
			//add index of parameter combination in case of multiple loaded parameters
			if(param_names.size()==1)//is there a single parameter?
			{
				param_val = _exp_param.getParamValByName(param_names.firstElement(), i);
			}
			else
			{
				param_val = i;//use variable of parameter value to store the index
			}
			
			//convert a parameter value to a text form
			tmp_str = convertObjectToString(param_val);
			param_class = param_val.getClass();
			if(param_class==Integer.class)
			{
				out_list[i]+=alignStr(tmp_str, 'r', num_param_comb.toString().length(), '0');
			}
			else if(param_class==Double.class || param_class==interval_C.class)
			{
				//leave only a value of the right bound 
				if(param_class==interval_C.class)
				{
					tmp_str  = "";
					tmp_str += ((interval_C)param_val).getUpperLimitAsDouble();
				}
				
				//replace comma with underscore
				out_list[i]+= tmp_str.replace(".", "_");
			}
			else
			{
				System.err.println("createOutNameParamVal: submitted type of parameter values is not supported");
				System.exit(1);
			}
		}
		return out_list;
	}
	
	/**
	 * The function finds a width of a column with data for an output file.
	 * @param col_name: name of the column
	 * @param array: array of data to be output in the column
	 * @return: column width
	 */
	private int findColWidth(String col_name, Object[] array)
	{
		int i;
		int max_len;//maximum string length
		String tmp_str;//temporary string
		
		if(col_name!=null)
		{
			max_len = col_name.length();
		}
		else
		{
			max_len = 0;
		}
		for(i=0; i<array.length; i++)
		{
			tmp_str = convertObjectToString(array[i]).toString();
			if(tmp_str.length() > max_len)
			{
				max_len = tmp_str.length();
			}
		}
		return max_len;
	}

	/**
	 * save parameters used in the experiment
	 * @param bw, pointer to a BufferedWriter object
	 * @param param, array of strings each of them keeping parameter values to be saved
	 */
	private void saveFileHeader(BufferedWriter bw, Vector<String> param)
	{
		int i;
		try {
			for(i=0; i<param.size(); i++)
			{
				bw.write(param.get(i) + "\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * The function calls to create an object which will be used to format the output of an mESN state.
	 * @param filename: path to the file where the ESN states should be saved
	 * @param is_step: "true": if the time step should be saved, "false" - otherwise
     * @param is_mode: "true" is when the ESN application mode should be saved, "false" - otherwise
	 * @param num_in: number of the input neurons
	 * @param num_res: number of reservoir neurons in each module 
	 * @param num_out: number of the output neurons
	 * @return: object which is used to save the states of a single ESN in a file
	 */
	public esn_state_C prepareSavingEsnState(String filename, boolean is_step, boolean is_mode, int num_in,
			                                 int[] num_res, int num_out)
	{
		int num_step;//needed number of sub-columns for the output of the time steps
		int num_mode;//needed number of sub-columns for the output of the ESN application mode
		String state_header;//string with the names of columns in the file with the ESN states
		esn_state_C esn_state;//output object
		
		//assign a number of sub-columns for the output of the time step
		if(is_step==true)
		{
			num_step = 0;
		}
		else
		{
			num_step = -1;
		}
		//assign a number of sub-columns for the output of the ESN application mode
		if(is_mode==true)
		{
			num_mode = 0;
		}
		else
		{
			num_mode = -1;
		}
		
		esn_state = new esn_state_C(num_step, num_mode, num_in, num_res, num_out);
		
		esn_state._file = new File(filename);
		try
		{
			esn_state._bw = new BufferedWriter(new FileWriter(esn_state._file));
			state_header   = esn_state.getStateHeader();
			esn_state._bw.write(state_header);
			esn_state._bw.flush();
		}
		catch(IOException e)
		{
			System.err.println("prepareSavingEsnState: cannot open and save a header of ESN state in specified file");
			System.exit(1);
		}
		
		return esn_state;
	}
	
	/**
	 * The function calls to create an object which will be used to save an ESN. A file is opened for saving, only if
	 * the provided path is not "null".
	 * 
	 * @param filename: path to the file where an ESN should be saved
	 * @return: object which is used to save the states of a single ESN in a file
	 */
	public esn_output_C prepareSavingEsn(String filename)
	{
		esn_output_C esn_output;//output object
		
		esn_output = new esn_output_C(filename);
		
		return esn_output;
	}
	
	/**
	 * create main output directory 
	 */
	public void createDirMain()
	{
		createDir(_dir_out_main);
	}
	
	/**
	 * create output directory for the given parameter 
	 * @param idx_param_vect, index of parameter vector whose directory must be created 
	 */
	public void createDirParam(int idx_val)
	{
		createDir(_out_dir_param[idx_val]);
	}
	
	/**
	 * return output directory for the parameter vector specified by its index 
	 * @param idx, index of parameter vector 
	 * @return output directory
	 */
	public String getOutDirVectorIdx(int idx)
	{
		return _out_dir_param[idx];
	}
	
	/**
	 * return a common path to file where specific information from the given run is saved
	 * The calling class must add an identifier of the saved information as a suffix to the
	 * returned name by itself. For example, the common name is "d:\output_dir\run_123_".
	 * The calling algorithm should add the suffix "connect.dat" to save the connectivity
	 * matrix. In the shown example an index of the current run is 123.
	 * @return, common path to the files with run specific information  
	 */
	public String getCommonPathRun(Integer idx_exp, Integer idx_run)
	{
		return _out_dir_param[idx_exp] + File.separator + "run_" + idx_run.toString();
	}
	
	/**
	 * output the experiment parameter and its current value to the console  
	 * @param idx_param, index of the current vector of experiment parameters  
	 * @return none
	 */
	public void printParamVal(Integer idx_param)
	{
		String str_out;//string for data output
		Object param_val;//parameter value
		Vector<String> param_names;//names of parameters with provided values
		
		param_names = _exp_param.getProvidedParamNames(); 
		
		if(param_names.size()==1)
		{
			str_out   = param_names.firstElement() + "=";
			param_val = _exp_param.getParamValByName(param_names.firstElement(), idx_param);
			str_out  += convertObjectToString(param_val);
		}
		else
		{
			str_out = _gen_exp_param_dir_name + "=";
			str_out+= idx_param.toString();
		}		
		System.out.println(str_out);
	}
	
	/**
	 * The function copies a specified parameter file from the main project directory to the output directory.
	 * 
	 * @param path_param_file: name of parameter file
	 */
	public void saveParamFile(String name_param_file)
	{
		File        out_dir = new File(_dir_out_main);//file object of output directory
		File        param_file;//file object of parameter file
        File        param_file_new;
        FileChannel inChannel = null; 
        FileChannel outChannel = null;

		if(out_dir.exists()==false)//check existence of the output directory
		{
			System.err.println("saveParamFile: output directory does not exist yet");
			System.exit(1);
		}
		else
		{
			//create objects for a source and destination files
			param_file     = new File("." + File.separator + name_param_file);//file object of parameter file
			param_file_new = new File(_dir_out_main + File.separator + name_param_file);
			
			//rename a file with the same name as a name of the directory
			if(param_file_new.exists()==true)
			{
				System.err.println("saveParamFile: there is already a parameter file in the output directory");
				System.exit(1);
			}
			
			//copy the parameter file
			try { 
	            inChannel  = new FileInputStream(param_file).getChannel(); 
	            outChannel = new FileOutputStream(param_file_new).getChannel(); 
	            inChannel.transferTo(0, inChannel.size(), outChannel); 
	        } catch (IOException e) { 
	        	e.printStackTrace(); 
	        } finally { 
	            try { 
	                if (inChannel != null) 
	                    inChannel.close(); 
	                if (outChannel != null) 
	                    outChannel.close(); 
	            } catch (IOException e) {}
	            //assign a copy "ReadOnly" access
	            param_file_new.setReadOnly();
	        }
		}
	}
	
	/**
	 * save vectors of provided experiment parameters
	 */
	public void saveParamVectors()
	{
		Integer  i, j;
		File file;//object of output file
		Vector<String> provided_param;//array of names of the provided parameters
		String   str_out;//string ready for output 
		int      num_param_vect;//number of parameter vectors
		int[]    col_width;//widths of columns with the parameter values 
		int      col_width_idx;//width of column with the indices of parameter vectors
		Object[] tmp_array;//temporary array needed to call a function
		Object   tmp_val;//temporary value of experiment parameter
		
		num_param_vect = _exp_param.getNumParamVectors();

		provided_param = _exp_param.getProvidedParamNames();
		col_width = new int[provided_param.size()];
		//get width of the column with index of run in the output file
		tmp_array = new Object[1];
		tmp_array[0]  = num_param_vect-1;
		col_width_idx = findColWidth("Vector", tmp_array);
		//get widths of columns with statistics in the output file
		for(i=0; i<provided_param.size(); i++)
		{
			col_width[i] = findColWidth(provided_param.get(i),
					                    _exp_param.getParamValues(provided_param.get(i)));
		}
		
		try {
			file = new File(_dir_out_main, _out_file_vectors);
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			
			str_out = alignStr("Vector", 'l', col_width_idx, ' ');
			str_out+= " ";

			//go over all provided parameters and save their names in a file
			for(i=0; i < provided_param.size(); i++)
			{
				str_out += alignStr(provided_param.get(i), 'l', col_width[i], ' ');
				str_out += " ";
			}
			str_out += "\n";
			bw.write(str_out);
			
			//go over all values of the experiment parameters
			_exp_param.setFirstParamVector();
			for(i=0; i<num_param_vect; i++)
			{
				str_out = i.toString();
				str_out = alignStr(str_out, 'l', col_width_idx, ' ');
				str_out+= " ";
				
				for(j=0; j<provided_param.size(); j++)
				{
					tmp_val  = _exp_param.getParamVal(provided_param.get(j), req_val_E.RV_CUR);
					str_out += alignStr(convertObjectToString(tmp_val), 'l', col_width[j], ' ');
					str_out += " ";
				}
				str_out += "\n";
				bw.write(str_out);
				
				//set next parameter vector
				_exp_param.setNextParamVector();
			}
			
			bw.close();
			bw = null;
			//assign a copy "ReadOnly" access
            file.setReadOnly();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * save statistics from all experiments 
	 * @param dataCollector: array with statistics from each experimental parameter
	 * @param param: array of strings with parameters to be saved
	 * @param exp_param_header: array of strings with all values of all experiment parameters
	 */
	public void saveStatExp()
	{
		Integer  i, j;
		File     file;
		Vector<String> out_header;//output file header with values of used parameters
		String   str_out;//string for data output
		String   filename;//name of output file
		Vector<String> param_names;//number of parameter names
		Object   param_val;//value of current parameter
		Object[] tmp_array;//temporary array needed to call a function
		int      col_width_exp;//width of column with values of experiment parameter
		int[]    col_width;//array with widths of columns in output file
		
		//get a list of parameter names
		param_names = _exp_param.getProvidedParamNames();
		
		col_width = new int[_data_exp.length];
		//get width of the column with values of experiment parameter in the output file
		if(param_names.size()==1)//specify parameter value in case of a single loaded parameter
		{
			col_width_exp = findColWidth(param_names.firstElement(),
					                     _exp_param.getParamValues(param_names.firstElement()));
		}
		else//specify only index of parameter vector, if values were provided for several parameters
		{
			tmp_array = new Object[1];
			tmp_array[0]  = _exp_param.getNumParamVectors()-1;
			col_width_exp = findColWidth(_gen_exp_param_dir_name, tmp_array);
		}
		//get widths of columns with statistics in the output file
		for(i=0; i<_data_exp.length; i++)
		{
			if(_data_exp[i].save)
			{
				col_width[i] = findColWidth(_data_exp[i].name, _data_exp[i]._stat_data);
			}
		}
		
		try {
			if(param_names.size()==1)
			{
				filename = "statistics_"+ param_names.firstElement() + ".dat";
				str_out  = param_names.firstElement();
			}
			else
			{
				filename = "statistics_" + _gen_exp_param_dir_name + ".dat";
				str_out  = _gen_exp_param_dir_name;
			}
			str_out = alignStr(str_out, 'l', col_width_exp, ' ');
			str_out+= " ";
			file = new File(_dir_out_main, filename);
			
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			//save file header
			out_header = createFileHeader(header_type_E.HT_EXP_STAT, null);
			saveFileHeader(bw, out_header);
			
			//go over all output statistics and save their names in a file
			for(i=0; i < _data_exp.length; i++)
			{
				if(_data_exp[i].save)
				{
					str_out += alignStr(_data_exp[i].name, 'l', col_width[i], ' ');
					str_out += " ";
				}
			}
			str_out += "\n";
			bw.write(str_out);
			
			//go over all values of the experiment parameters 
			for(i=0; i < _exp_param.getNumParamVectors(); i++)
			{
				if(param_names.size()==1)//specify parameter value in case of a single loaded parameter
				{
					param_val = _exp_param.getParamValByName(param_names.firstElement(), i);
					str_out = convertObjectToString(param_val);
				}
				else//specify only index of parameter vector, if values were provided for several parameters
				{
					str_out = i.toString();
				}
				str_out = alignStr(str_out, 'l', col_width_exp, ' ');
				str_out+= " ";
				
				for(j=0; j<_data_exp.length; j++)
				{
					if(_data_exp[j].save)
					{
						str_out += alignStr(convertObjectToString(_data_exp[j]._stat_data[i]), 'l', col_width[j], ' ');
						str_out += " ";
					}
				}
				str_out += "\n";
				bw.write(str_out);
			}
			
			bw.close();
			bw = null;
			//assign a copy "ReadOnly" access
            file.setReadOnly();
		} catch(FileNotFoundException fnf) {
			fnf.printStackTrace();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	/**
	 * save statistics from all runs for the current value of experiment parameter
	 * @param dataCollector: array of output data from all runs
	 * @param idx_param: index of current parameter value
	 */
	public void saveStatRun(int idx_param)
	{
		Integer i, j;
		File   file;
		Vector<String> out_header;//output file header with values of used parameters
		Vector<String> names_to_save;//names of indicators to save
		Vector<stat_data_C> val_run;//values from one run
		String str_out;//string for data output 
		String filename;//file for saving errors from all runs of current parameter value
		String tmp_str;//temporary string
		int   num_runs;//number of values to be saved in one column
		int   col_width_run;//width of column with values of experiment parameter
		int[] col_width;//array with widths of columns in output file
		Object[] tmp_array;//temporary array needed to call a function
		
		filename = _out_dir_param[idx_param] + File.separator + "error_" + _out_name_param_val[idx_param] + ".dat";
		
		col_width = new int[_data_run.getNumIndicatorsToSave()];
		//get width of a column with the run index
		tmp_array = new Object[1];
		tmp_array[0]  = _data_run.getNumRuns() - 1;
		col_width_run = findColWidth("Run", tmp_array);
		//get widths of columns with statistics in the output file
		names_to_save = _data_run.getNamesToSave();
		for(i=0; i<names_to_save.size(); i++)
		{
			tmp_str = names_to_save.elementAt(i);
			tmp_array = _data_run.getValuesByName(tmp_str);
			col_width[i] = findColWidth(tmp_str, tmp_array);
		}
		
		//save data
		try {
			file = new File(filename);
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			out_header = createFileHeader(header_type_E.HT_RUN_STAT, null);
			saveFileHeader(bw, out_header);
			
			str_out = alignStr("Run", 'l', col_width_run, ' ');
			str_out+= " ";

			//go over all output statistics and save their names in a file
			for(i=0; i < col_width.length; i++)
			{
				str_out += alignStr(names_to_save.elementAt(i), 'l', col_width[i], ' ');
				str_out += " ";
			}
			str_out += "\n";
			bw.write(str_out);
			
			//go over all values of the experiment parameters
			num_runs = _data_run.getNumRuns();
			for(i=0; i < num_runs; i++)
			{
				str_out = i.toString();
				str_out = alignStr(str_out, 'l', col_width_run, ' ');
				str_out+= " ";
				
				val_run = _data_run.getValuesToSave(i);
				for(j=0; j<names_to_save.size(); j++)
				{
					str_out +=alignStr(convertObjectToString(val_run.elementAt(j)._stat_data[0]),'l',col_width[j],' ');
					str_out +=" ";
				}
				str_out += "\n";
				bw.write(str_out);
			}
			
			bw.close();
			bw = null;
			//assign a copy "ReadOnly" access
            file.setReadOnly();
		} catch(FileNotFoundException fnf) {
			fnf.printStackTrace();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	/**
	 * The function sets a value of the specified statistics in the given experiment. 
	 * @param idx_stat: index of statistics
	 * @param idx_exp: index of value of the experiment parameter
	 * @param value: value of statistics to be assigned
	 */
	public void setStatExp(stat_exp_E idx_stat, int idx_exp, Number value)
	{
		_data_exp[idx_stat.ordinal()]._stat_data[idx_exp] = value;
	}
	
	/**
	 * The function returns an indicator whether the specified statistics will be saved.
	 * @param idx_stat: index of specified statistics
	 * @return TRUE: if statistics will be saved; FALSE: otherwise
	 */
	public boolean getStatExpSave(stat_exp_E idx_stat)
	{
		return _data_exp[idx_stat.ordinal()].save;
	}
	
	/**
	 * The function assigns an indicator whether the specified statistics should be saved in a file.
	 * @param idx_stat: index of specified statistics
	 * @param do_save: indicator whether the statistics should be saved
	 */
	public void setStatExpSave(stat_exp_E idx_stat, boolean do_save)
	{
		_data_exp[idx_stat.ordinal()].save = do_save;
	}
	
	/**
	 * The function returns an array with the required performance indicator from all runs.
	 * The required indicator is returned for the specified interval of the sequence.
	 * 
	 * @param indicator: required performance indicator
	 * @param idx_interval: index of the sequence interval
	 * @return array with the performance indicator
	 */
	public Number[] getRunPerform(stat_run_E indicator, int idx_interval)
	{
		int      i;
		int      num;//number of performance values to be returned
		Number[] performance;
		
		//prepare to return performance indicators as an array of numbers
		num = _data_run._run_perform[idx_interval][indicator.ordinal()]._stat_data.length;
		performance = new Number[num];
		for(i=0; i<num; i++)
		{
			performance[i] = (Number)_data_run._run_perform[idx_interval][indicator.ordinal()]._stat_data[i];
		}
		
		return performance;
	}
	
	/**
	 * The function stores a provided value of the specified performance indicator for saving in a file.
	 * The provided value is given for the specified sequence interval and for the specified run.
	 * 
	 * @param indicator: performance indicator
	 * @param idx_interval: index of sequence interval
	 * @param idx_run: index of run
	 * @param value: value of the performance indicator to be stored
	 */
	public void setRunPerform(stat_run_E indicator, int idx_interval, int idx_run, Number value)
	{
		_data_run._run_perform[idx_interval][indicator.ordinal()]._stat_data[idx_run] = value;
	}
	
	/**
	 * The function stores the provided value which was used for seeding the random generators in the specified run.
	 * The function store it as a string which contains seeding values for all modles.
	 * 
	 * @param idx_run: index of run
	 * @param value: provided array of seeding values
	 */
	public void setRunSeed(int idx_run, int[] value)
	{
		int     i;
		Integer tmp_int;//temporary value
		String  seed_str;
		
		seed_str = "{";
		for(i=0; i<value.length; i++)
		{
			tmp_int = value[i];
			seed_str += tmp_int.toString();
			if(i!=value.length-1)//no comma is needed after the last element
			{
				seed_str += ",";
			}
		}
		seed_str += "}";
		
		_data_run._run_seed._stat_data[idx_run] = seed_str;
	}
	
	/**
	 * The function returns an array with numbers of run repetitions.
	 * 
	 * @return: array with numbers of run repetitions
	 */
	public Number[] getRunRepeat()
	{
		int      i;
		int      num;//number of performance values to be returned
		Number[] repeat;
		
		//prepare to return numbers of repetitions as an array of numbers
		num = _data_run._run_repeat._stat_data.length;
		repeat = new Number[num];
		for(i=0; i<num; i++)
		{
			repeat[i] = (Number)_data_run._run_repeat._stat_data[i];
		}
		
		return repeat;
	}
	
	/**
	 * The function stores the provided number of repetitions of the specified run.
	 * The number of repetitions was performed until the run was successfully completed.
	 * 
	 * @param idx_run: index of run
	 * @param value: provided number of repetitions
	 */
	public void setRunRepeat(int idx_run, Number value)
	{
		_data_run._run_repeat._stat_data[idx_run] = value;
	}
	
	/**
	 * The function stores a provided string as a sequence name of the specified run.
	 * 
	 * @param idx_run: index of run
	 * @param seq_name: provided sequence name
	 */
	public void setRunSeq(int idx_run, String seq_name)
	{
		_data_run._run_seq._stat_data[idx_run] = seq_name;
	}
}
