package experiment;

import java.io.File;

import adaptation.DiffEvolutionParam;
import adaptation.DiffEvolutionParam.config_ea_init_E;
import adaptation.DiffEvolutionParam.config_ea_order_method_E;
import types.stat_seq_C;
import types.vector_C;
import types.seq_C;
import MathDiff.MathDistortion;
import MathDiff.MathDistortion.distortion_E;
import MathStat.StatCommon;
import esn.EsnModule;
import esn.EsnModule.config_method_E;
import esn.Module.storage_type_E;
import esn.mESN;
import esn.mESN.config_ea_mode_E;
import experiment.ExpOutput.stat_run_E;
import experiment.ExpParam.exp_param_E;
import experiment.ExpParam.req_val_E;

/**
 * This class provides functions to perform one experiment run on a separate processor core.
 * @author Danil Koryakin
 *
 */
public class ExpRun extends Thread
{
	ExpParam  _exp_param;//object with parameters for the experiment
	ExpSeq    _exp_seq;//object containing loaded sequences
	ExpOutput _exp_output;//object keeping output statistics
	StatCommon  _math_stat;//object providing functions to compute output statistics
	
	Integer  _idx_run;//index of currently processed run
	Integer  _idx_exp;//index of current vector of experiment parameters
	int[]    _seed_init;//initial value for seeding random generators in the current run
	int[]    _seed_curr;//value for seeding random generators for each ESN module
	boolean[] _seed_load;//request to use a loaded seeding value
	boolean[] _active_modules;//indicators of active modules
	vector_C _seq_purpose;//sequence purposes
	vector_C _seq_start;//indices of the first samples
	vector_C _seq_stop;//indices of the last samples
	vector_C _seq_active;//indices of active sequences
	vector_C _seq_distort;//request for OFB distortion
	seq_C[]  _train_seq;//array of sequences which were used for training the output weights of the sub-reservoirs
	
	String[] _file_esn_load;//array of paths, including the file names, to ESN modules to be loaded
	
	//statistics: 1D<->output neurons; [][0]<->training statistics; [][1]<->test statistics
	Integer  _run_repeats;//number of repetitions of the current run
	Integer  _run_repeats_max;//maximum number of repetition of current run
	stat_seq_C[] _stat_seq;//statistics which were computed for all sequence intervals
	
	boolean _do_run;//indicator to start the next run
	boolean _do_finish;//indicator to finish thread
	
	/**
	 * constructor of the thread for processing a run of the experiment
	 * @param exp_param, object with parameters for the experiment 
	 * @param exp_output, object keeping output statistics
	 * @param math_stat, object providing functions to compute output statistics
	 */
	public ExpRun(ExpParam  exp_param,
	              ExpOutput exp_output,
	              StatCommon  math_stat)
	{
		_exp_param  = exp_param;
		_exp_output = exp_output;
		_math_stat  = math_stat;
		
		_do_run    = false;//no run must be processed before setting the necessary parameters 
		_do_finish = false;//in the beginning the thread must not be finished
	}
	
	/**
	 * The function creates an object for simulation of OFB distortion.
	 * 
	 * @param seq_len: length of the configuration sequence
	 * @param max_abs: maximum absolute values for each element of the configuration sample
	 * @param seed: value which should be used for seeding the generator of random numbers
	 * @return: object for simulation of disturbances
	 */
	private MathDistortion createOfbDistort(int seq_len, double[] max_abs, int seed)
	{
		int i;
		int num_out;//number of the elements of the disturbance vector
		vector_C period_vec;//period of the disturbance as a vector
		int[]    period;//period of the disturbance
		vector_C decay_vec;//decay parameter as a vector
		double[] decay;//coefficient of exponential decay or variance at the last index of the sequence
		vector_C strength_rel;//initial strength as a percentage of the maximum amplitude of configuration sequence
		double[] strength_abs;//computed absolute value of the initial strength
		distortion_E   type;//required type of the disturbance
		MathDistortion disturb;//output variable
		
		num_out      = max_abs.length;
		type         = (distortion_E)_exp_param.getParamVal(exp_param_E.EP_DISTORT_METHOD, req_val_E.RV_CUR);
		period_vec   = (vector_C)_exp_param.getParamVal(exp_param_E.EP_DISTORT_PERIOD, req_val_E.RV_CUR);
		strength_rel = (vector_C)_exp_param.getParamVal(exp_param_E.EP_DISTORT_STRENGTH, req_val_E.RV_CUR);
		decay_vec    = (vector_C)_exp_param.getParamVal(exp_param_E.EP_DISTORT_DECAY, req_val_E.RV_CUR);
		
		//allocate to call the constructor of the disturbance object
		period       = new int[num_out];
		decay        = new double[num_out];
		strength_abs = new double[num_out];
		
		switch(type)
		{
			case DISTORT_CONST:
				//compute the absolute values of the disturbance strength and,
				//convert the period into the corresponding arrays
				for(i=0; i<num_out; i++)
				{
					strength_abs[i] = max_abs[i] * strength_rel.getElementAsDouble(i);
					period[i] = period_vec.getElementAsInt(i);
				}
				disturb = new MathDistortion(type, strength_abs, period);
				break;
			case DISTORT_GAUSS:
				System.err.println("createConfigDisturbance: disturbance with a static Gaussian is not available");
				System.exit(1);
				disturb = null;
				break;
			case DISTORT_GAUSS_EXP:
				//compute the absolute values of the disturbance strength and,
				//convert the period and decay coefficient into the corresponding arrays
				for(i=0; i<num_out; i++)
				{
					strength_abs[i] = max_abs[i] * strength_rel.getElementAsDouble(i);
					period[i] = period_vec.getElementAsInt(i);
					decay[i]  = decay_vec.getElementAsDouble(i);
				}
				disturb = new MathDistortion(type, strength_abs, decay, seq_len, period, seed);
				break;
			case DISTORT_NONE:
				disturb = new MathDistortion(type, num_out);
				break;
			default:
				System.err.println("createConfigDisturbance: unknown type of disturbance");
				System.exit(1);
				disturb = null;
				break;
		}
		return disturb;
	}
	
	/**
	 * The function initializes the global variables of the class after starting a run. The initialized variables
	 * are those which depend on the values of the experimental parameters for this run.
	 * For example, these are variables that depend on the ESN structure or on a number of different sequence
	 * intervals.
	 * @param num_out: number of the output neurons
	 * @param num_sub: total number of the sub-reservoirs
	 * @param num_intervals: number of intervals where the statistics are computed
	 */
	private void initRunAfterStart(int num_out, int num_sub, int num_intervals)
	{
		int i,j;
		
		/*initialize the performance statistics*/
		_stat_seq = new stat_seq_C[num_intervals];
		
		for(i=0; i<num_intervals; i++)
		{
			_stat_seq[i] = new stat_seq_C(num_out, 0);
			
			_stat_seq[i].comp_ident_ok_incl_error = 0;
			_stat_seq[i].comp_ident_ok            = 0;
			for(j=0; j<num_out; j++)
			{
				_stat_seq[i].mse  [j] = Double.NaN;
				_stat_seq[i].nrmse[j] = Double.NaN;
				_stat_seq[i].rmse [j] = Double.NaN;
				_stat_seq[i].sel  [j] = Integer.MAX_VALUE;
				_stat_seq[i].lel  [j] = Integer.MAX_VALUE;
			}
		}

		//initialize an array of the training sequences
		_train_seq = new seq_C[num_sub];
		for(i=0; i<num_sub; i++)
		{
			_train_seq[i] = null;
		}
		
		//allocate an indicator array for activation of ESN modules
		_active_modules = new boolean[num_sub];
	}
	
	/**
	 * The function applies the provided ESN to the loaded sequences. At every time step the ESN is applied to
	 * the currently active sequence according to the current purpose.
	 * 
	 * @param esn: provided echo-state network
	 * @return: array with indicators for each module whether a training of the corresponding module succeeded 
	 *          "false" is if the training was required and it could not be finished because of problems,
	 *          "true" is if the training was not required or it was successfully finished
	 */
	private boolean switchRun(mESN esn)
	{
		final int module_idx = 0;//always seed of the 1st module is used to create a distortion object  
		
		int       out_idx;//index of the output vector's element to get the performance indicator
		int       idx_interval;//index of the current interval of the sequence
		int       idx_first, idx_last;//indices of 1st and last samples of the active sequence
		int       do_ofb_distort;//request to perform OFB disturbance
		int       max_idx_seq;//maximum index of a sequence to be used in the current ESN configuration
		int       max_idx_module;//maximum index of a module to be activated
		boolean   is_multifile;//indicator that data were loaded from a multi-file sequence
		boolean   train_ok;//indicator that the training was OK on all sequences
		boolean   train_seq_ok;//indicator that the training was OK on the current sequence
		Integer   seq_active_int;//index of currently active sequence
		Character seq_active_chr;//index of currently active sequence can be the character '*'
		Character seq_purpose;//current purpose
		MathDistortion ofb_distort;//object with parameters of required OFB distortion 
		seq_C     seq;//sequence to create an object for OFB distortion
		
		is_multifile = _exp_seq.isMiltifileSeq();
		
		//"out_idx" is always 0 because currently the performance indicators are saved only for the 1st output neuron
		out_idx = 0;
		
		//go over sequences which must be activated
		train_ok = true;
		for(idx_interval=0; idx_interval<_seq_active.getSize(); idx_interval++)
		{
			//get maximum index of a sequence which will be used in the current ESN configuration
			seq_active_int = _seq_active.getElementAsInt (idx_interval);
			seq_active_chr = _seq_active.getElementAsChar(idx_interval);
			if(seq_active_int!=null)
			{
				max_idx_module = seq_active_int;
				//assign a sequence either from multi-file or from single-file sequence
				if(is_multifile==false)
				{
					max_idx_seq = seq_active_int;
				}
				else
				{
					max_idx_seq = 0;//for multi-file sequences, data from only one file is provided
				}
			}
			else
			{
				if(seq_active_chr=='*')
				{
					//assign a sequence either from multi-file or from single-file sequence
					if(is_multifile==false)
					{
						max_idx_module = _exp_seq.getSeqNum() - 1;
						max_idx_seq    = _exp_seq.getSeqNum() - 1;
					}
					else
					{
						//An mESN must already be present (loaded), if a multi-file sequence is provided.
						max_idx_module = esn.getNumModules() - 1;
						max_idx_seq    = 0;//for multi-file sequences, data from only one file is provided
					}
				}
				else
				{
					max_idx_module = 0;
					max_idx_seq    = 0;
					System.err.println("switchSeq: unknown index of the active sequence");
					System.exit(1);
				}
			}
			markActiveModules(max_idx_module);
			
			//assign 1st and last sample of sequence which will be processed with the ESN according to current purpose
			idx_first = _seq_start.getElementAsInt(idx_interval);
			idx_last  = _seq_stop.getElementAsInt(idx_interval);
			//apply the ESN according to the current purpose
			seq_purpose = _seq_purpose.getElementAsChar(idx_interval);
			//define an object to compute OFB distortion
			seq = _exp_seq.getSeq(max_idx_seq, idx_first, idx_last);
			do_ofb_distort = _seq_distort.getElementAsInt(idx_interval);
			if(do_ofb_distort==1)
			{
				ofb_distort = createOfbDistort(seq.getSeqLen(), seq.getMaxAbsOut(), esn.getSeedModule(module_idx));
			}
			else
			{
				ofb_distort = null;
			}
			//process the ESN according to the purpose
			switch(seq_purpose)
			{
				case 'w':
					doTeacherForcedRun(false, esn, max_idx_seq, idx_first, idx_last, ofb_distort, _stat_seq[idx_interval]);
					//reset indicator that an mESN is configured
					esn.resetConfigured();
				break;
				case 't':
					if(seq_active_chr!='*')
					{
						System.err.println("switchSeq: only \"*\" is accepted as a sequence index for training");
						System.exit(1);
					}
					train_seq_ok = doTrainRun(esn, max_idx_seq, idx_first, idx_last, _stat_seq[idx_interval]);
					//reset indicator that an mESN is configured
					esn.resetConfigured();
					
					if(train_seq_ok==false)
					{
						train_ok = false;
					}
				break;
				case 'f':
					if(seq_active_chr!='*')
					{
						System.err.println("switchSeq: only \"*\" is accepted as a sequence index for the free-run");
						System.exit(1);
					}
					doFreeRun("freerun", esn, max_idx_seq, idx_first, idx_last, ofb_distort, _stat_seq[idx_interval]);
					//reset indicator that an mESN is configured
					esn.resetConfigured();
				break;
				case 'c':
					doTeacherForcedRun(true, esn, max_idx_seq, idx_first, idx_last, ofb_distort, _stat_seq[idx_interval]);
				break;
				case 'e':
					doFreeRun("test", esn, max_idx_seq, idx_first, idx_last, ofb_distort, _stat_seq[idx_interval]);
					//reset indicator that an mESN is configured
					esn.resetConfigured();
				break;
				default:
					System.err.println("switchSeq: unknown sequence purpose");
					System.exit(1);
				break;
			}//switch "seq_purpose"
			_exp_output.setRunPerform(stat_run_E.SR_CIOK_INCL_ERR, idx_interval, _idx_run, _stat_seq[idx_interval].comp_ident_ok_incl_error);
			_exp_output.setRunPerform(stat_run_E.SR_CIOK, idx_interval, _idx_run, _stat_seq[idx_interval].comp_ident_ok);
			_exp_output.setRunPerform(stat_run_E.SR_MSE,   idx_interval, _idx_run, _stat_seq[idx_interval].mse  [out_idx]);
			_exp_output.setRunPerform(stat_run_E.SR_NRMSE, idx_interval, _idx_run, _stat_seq[idx_interval].nrmse[out_idx]);
			_exp_output.setRunPerform(stat_run_E.SR_RMSE,  idx_interval, _idx_run, _stat_seq[idx_interval].rmse [out_idx]);
			_exp_output.setRunPerform(stat_run_E.SR_SEL,   idx_interval, _idx_run, _stat_seq[idx_interval].sel  [out_idx]);
			_exp_output.setRunPerform(stat_run_E.SR_LEL,   idx_interval, _idx_run, _stat_seq[idx_interval].lel  [out_idx]);
		}//for "idx_interval"
		return train_ok;
	}
	
	/**
	 * The function prepares a sequence for the free-run and configures the ESN for this sequence.
	 * @param seq_label: label to be saved at the time steps of the free-run: "test" or "freerun"
	 * @param esn: echo-state network
	 * @param max_idx_seq: max index of sequence to be used for the free-run
	 * @param idx_first: index of the first sample
	 * @param idx_last: index of the last sample
	 * @param ofb_distort: object for simulating OFB distortion
	 * @param stat_seq_store: object with the statistics which are to be computed during the free-run
	 */
	private void doFreeRun(String seq_label, mESN esn, int max_idx_seq, int idx_first, int idx_last,
			               MathDistortion ofb_distort, stat_seq_C stat_seq_store)
	{
		int i;
		boolean    is_saving;//indicator whether the saving is required
		double     ciok_level;//largest allowed error for successful component identification
		seq_C      test_seq;//sequence where the ESN will be applied
		seq_C[]    seq_target_sub;//array of sequences as target values for separate sub-reservoirs
		stat_seq_C stat_seq;//set of statistics computed when the ESN is run on the given test sequence
		
		//extract a single sequence for the testing
		test_seq = _exp_seq.getSeq(max_idx_seq, idx_first, idx_last);
		
		is_saving = esn.isSavingRequired();
		if(is_saving==true)
		{
			//prepare several sequences of sub-reservoirs' target values for their saving together with test ESN states
			seq_target_sub = new seq_C[max_idx_seq+1];
			for(i=0; i<=max_idx_seq; i++)
			{
				seq_target_sub[i] = _exp_seq.getSeq(i, idx_first, idx_last);
			}
			//obtain the differences between the sequences
			seq_target_sub = _exp_seq.convertSeqToDiff(seq_target_sub);
		}
		else
		{
			seq_target_sub = null;
		}

		//activate modules according to a target dynamics if they were not configured on a preceding interval
		if(esn.isConfigured()==false)
		{
			//activate the sub-reservoirs with an index up to the provided index;
			//ESN states MUST BE restored
			esn.activateModules(_active_modules, true);
		}
		
		ciok_level = (Double)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_CIOK_LEVEL, req_val_E.RV_CUR);
		
		//run the ESN on the test sequence
		stat_seq = esn.runFree(seq_label, test_seq, seq_target_sub, idx_first, is_saving, ofb_distort, ciok_level);
		stat_seq_store.assignData(stat_seq);
		
		//store the output of the sub-reservoirs for their possible activation later
		esn.storeModuleNodes(storage_type_E.ST_INIT);
	}

	/**
	 * The function prepares sequences for a teacher-forced run and runs the ESN on the sequence which is specified by
	 * its index.
	 * The teacher-forced run can be either a washout run or a configuration run. It is specified with the provided
	 * boolean indicator. In the washout run the loaded sequence is a source of the teaching values.
	 * In the configuration run the ESN is adapted to the teacher sequence either by forcing it with computed teaching
	 * values or by a search for suitable reservoir states.
	 * @param config: if "true": specified configuration method should be applied; "false": washout run
	 * @param esn: echo-state network
	 * @param max_idx_seq: largest required index of sequence where the ESN is to be run in the teacher-forced mode
	 * @param idx_first: index of the first sample
	 * @param idx_last: index of the last sample
	 * @param ofb_distort: object for simulating OFB distortion
	 * @param stat_seq_store: object with the statistics which are to be computed during the teacher-forced run
	 */
	private void doTeacherForcedRun(boolean is_config, mESN esn, int max_idx_seq, int idx_first, int idx_last,
			                        MathDistortion ofb_distort, stat_seq_C stat_seq_store)
	{
		final int module_idx = 0;//parameters of diff evolution always uses a seed of the first module;
		                         //ATTENTION: this generator is different from those used under evolution
		
		int i;
		seq_C[]    seq;//array of sequences which will be used for the differential driving
		stat_seq_C stat_seq;//set of statistics computed when the ESN is run on the sequence of provided largest index
		config_method_E config_method;//specified configuration method to be applied
		DiffEvolutionParam de_param;//parameters of differential evolution

		if(is_config==true)//configuration run
		{
			//activate the sub-reservoirs with an index up to the provided index;
			//states of the ESN must NOT be restored
			esn.activateModules(_active_modules, false);

			config_method = (config_method_E)_exp_param.getParamVal(exp_param_E.EP_CONFIG_METHOD, req_val_E.RV_CUR);
			switch(config_method)
			{
				case CM_SubTargetEstimate:
					//activate the sub-reservoirs with an index up to the provided index;
					//states of the ESN must NOT be restored;
					//synchronization requires disabling of unneeded modules
					esn.activateModules(_active_modules, false);
					
					//prepare a sequence with the provided index as a single sequence for the configuration
					seq = new seq_C[1];
					seq[0] = _exp_seq.getSeq(max_idx_seq, idx_first, idx_last);
					stat_seq = esn.configSubTargetEstimate(seq[0], _train_seq);
				break;
				case CM_DirectTeacherForce:
					//activate the sub-reservoirs with an index up to the provided index;
					//states of the ESN must NOT be restored;
					//synchronization requires disabling of unneeded modules
					esn.activateModules(_active_modules, false);
					
					//prepare a sequence with the provided index as a single sequence for the configuration
					seq = new seq_C[1];
					seq[0] = _exp_seq.getSeq(max_idx_seq, idx_first, idx_last);
					stat_seq = esn.configDirectTeacherForce(seq[0], idx_first, ofb_distort);
				break;
				case CM_DiffEvolution:
					//prepare a sequence with the provided index as a single sequence for the configuration
					seq = new seq_C[max_idx_seq+1];
					for(i=0; i<=max_idx_seq; i++)
					{
						seq[i] = _exp_seq.getSeq(i, idx_first, idx_last);
					}
					de_param = new DiffEvolutionParam(esn.getSeedModule(module_idx));
					de_param.force_mode = (config_ea_mode_E)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_MODE, req_val_E.RV_CUR);
					de_param.init_mode  = (config_ea_init_E)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_INIT, req_val_E.RV_CUR);
					de_param.coevol_method   = (config_ea_order_method_E)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_COEVOL_ORDER_METHOD, req_val_E.RV_CUR);
					de_param.coevol_order    = (vector_C)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_COEVOL_ORDER, req_val_E.RV_CUR);
					de_param.sort_min_num    = (Integer)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_COEVOL_SORT_MIN_NUM, req_val_E.RV_CUR);
					de_param.validation_len  = (Integer)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_VALIDATION_LENGTH, req_val_E.RV_CUR);
					de_param.F  = (Double)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_DIFF_WEIGHT, req_val_E.RV_CUR);
					de_param.CR = (Double)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_CROSS_PROB, req_val_E.RV_CUR);
					de_param.margin_module = (vector_C)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_MARGIN_MODULE, req_val_E.RV_CUR);
					de_param.pop_size = (Integer)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_POP_SIZE, req_val_E.RV_CUR);
					de_param.num_gen  = (Integer)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_NUM_GEN, req_val_E.RV_CUR);
					de_param.fitness_length = (Integer)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_FIT_LEN, req_val_E.RV_CUR);
					de_param.ciok_level = (Double)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_CIOK_LEVEL, req_val_E.RV_CUR);
					de_param.activ_thresh = (Double)_exp_param.getParamVal(exp_param_E.EP_CONFIG_EA_ACTIV_THRESH, req_val_E.RV_CUR);

					stat_seq = esn.configDiffEvolution(seq, idx_first, ofb_distort, de_param, _seed_curr);
			    break;
				default:
					System.err.println("doTeacherForcedRun: unknown configuration method is required");
					System.exit(1);
					stat_seq = null;
				break;
			}
		}
		else//washout run
		{
			//activate specified modules;
			//states of the ESN must NOT be restored
			esn.activateModules(_active_modules, false);
			
			//prepare a set of sequences for the washout
			seq = new seq_C[max_idx_seq+1];
			for(i=0; i<=max_idx_seq; i++)
			{
				seq[i] = _exp_seq.getSeq(i, idx_first, idx_last);
			}
			
			//washout the ESN with these sequences
			stat_seq = esn.runTeacher("washout", seq, idx_first);
		}
		stat_seq_store.assignData(stat_seq);

		//store the states of the modules to restore them later
		esn.storeModuleNodes(storage_type_E.ST_INIT);
	}

	/**
	 * The function prepares sequences for the training and configures the ESN for these sequences.
	 * @param esn: echo-state network
	 * @param max_idx_seq: largest required index of sequence to be used for the training
	 * @param idx_first: index of the first sample
	 * @param idx_last: index of the last sample
	 * @param stat_seq_store: object with the statistics which were computed on the training sequence in the free-run
	 * @return: "true" is if the training was OK for all sub-reservoirs, "false" is if the training of the whole ESN
	 *          was not finished because it was not OK for at least one sub-reservoir
	 */
	private boolean doTrainRun(mESN esn, int max_idx_seq, int idx_first, int idx_last, stat_seq_C stat_seq_store)
	{
		int        i;
		boolean    train_ok;//indicator that the training was OK
		EsnModule  esn_sub;//currently considered ESN module
		seq_C[]    seq;//array of sequences which will be used for the differential driving
		seq_C[]    seq_init;//array of last samples for initialization of the output neurons
		seq_C      seq_test;//sequence with the maximum index for the run without reacher forcing
		stat_seq_C stat_seq;//set of statistics computed when the ESN is run on the given training sequence
		
		//extract intervals of the sequences
		seq      = new seq_C[max_idx_seq+1];
		seq_init = new seq_C[max_idx_seq+1];
		for(i=0; i<=max_idx_seq; i++)
		{
			seq[i]        = _exp_seq.getSeq(i, idx_first,   idx_last);
			_train_seq[i] = seq[i];//store the training sequences for using them under the configuration
			seq_init[i]   = _exp_seq.getSeq(i, idx_first-1, idx_first-1);
		}
		seq_test = seq[max_idx_seq];
		//obtain the differences between the sequences
		seq      = _exp_seq.convertSeqToDiff(seq);
		seq_init = _exp_seq.convertSeqToDiff(seq_init);
		
		//activate specified modules;
		//ESN states MUST BE restored
		esn.activateModules(_active_modules, true);
		
		//train output weights of each sub-reservoir separately
		train_ok = true;
		for(i=0; i<=max_idx_seq && train_ok==true; i++)
		{
			//create a small temporary ESN to train the output weights of the current sub-reservoir
			esn_sub = (EsnModule)esn.getModule(i);
			
			//initialize the output neuron with the last value of the corresponding target
			if(idx_first-1 < 0)
			{
				esn_sub.getOutputLayer().assignInitState();
			}
			else
			{
				//here index equals "0" because there is a single sample for each sequence
				esn_sub.getOutputLayer().setNodes(seq_init[i].getSampleOut(0), false);
			}
			
			//train the weights of the current sub-reservoir on its sequence
			esn_sub.training(seq[i]);
			train_ok &= esn_sub.getOutputLayer().isTrained();
			
			//- store reached states of the current module to restore them later
			//- store collected statistics about reservoir states
			esn.storeModuleNodes(i, storage_type_E.ST_TMP);
		}//for i
		
		if(train_ok==true)
		{
			//store the trained weights after the training of the output weights of the sub-reservoir with largest index 
			esn.storeModuleOutputWeights();
			
			//configure the trained ESN for the free run on the training sequence;
			//ESN states MUST BE restored
			esn.activateModules(_active_modules, true);
			
			//make a free run on the training sequence
			stat_seq = esn.runFree("", seq_test, null, 0, false, null, 0);
			stat_seq_store.assignData(stat_seq);
			
			//extract sequences for running the ESN on the largest of them in a teacher-forced run
			seq = new seq_C[max_idx_seq+1];
			for(i=0; i<=max_idx_seq; i++)
			{
				seq[i] = _exp_seq.getSeq(i, idx_first, idx_last);
			}
			
			//activate modules for a teacher-forced run;
			//ESN states MUST BE restored
			esn.activateModules(_active_modules, true);
	
			//Save the ESN states when it is teacher-forced with the most complicated sequence. In this case all
			//sub-reservoirs are active and their states are saved.
			esn.runTeacher("train", seq, idx_first);
			
			//store that neuron states that were reached in the training
			//(swap weights from temporary storages to initial ones)
			//It should be done after previous free- and teacher-forced run.
			esn.restoreModuleNodes(_active_modules, storage_type_E.ST_TMP);
			esn.storeModuleNodes(storage_type_E.ST_INIT);
		}//was training successful?

		return train_ok;
	}
	
	/**
	 * The function resets a run-specific data.
	 */
	private void resetRunData()
	{
		_idx_run = -1;
		_idx_exp = -1;
	}
	
	/**
	 * The function resets requests for loading seeding values.
	 */
	private void resetSeedLoad()
	{
		int i;
		
		for(i=0; i<_seed_load.length; i++)
		{
			_seed_load[i] = false;
		}
	}
	
	/**
	 * The function increments current seeds for all ESN modules which were not trainable.
	 * 
	 * @param esn: modular echo-state network
	 */
	private void updateSeed(mESN esn)
	{
		int i;
		int num_sub;//number of ESN modules
		
		num_sub = esn.getNumModules();
		for(i=0; i<num_sub; i++)
		{
			if(esn.isTrainedModule(i)==false)
			{
				_seed_curr[i]++;
			}
		}
	}
	
	/**
	 * The function sets indicators of modules that must be active in the run.
	 * The active modules are all modules with indices from "0" up to a provided one, inclusive.
	 * 
	 * @param max_idx_active: index of maximum provided module
	 */
	private void markActiveModules(int max_idx_active)
	{
		int i;
		int num_modules;//total number of modules
		
		num_modules = _active_modules.length;
		for(i=0; i<num_modules; i++)
		{
			if(i<=max_idx_active)
			{
				_active_modules[i] = true;
			}
			else
			{
				_active_modules[i] = false;
			}
		}
	}
	
	/**
	 * set the indicator to finish the thread
	 */
	public void finishThread()
	{
		_do_finish = true;
	}
	
	/**
	 * get an indicator whether the run is finished and ready to accept parameters for
	 * the next run
	 * @return, "true": if the run has already finished the processing; "false": otherwise  
	 */
	public boolean isReady()
	{
		boolean is_ready;
		
		if(_do_run)
		{
			is_ready = false;
		}
		else
		{
			is_ready = true;
		}
		return is_ready;
	}
	
	/**
	 * process the run according the set parameter values
	 */
	public void run()
	{
		boolean   f_save;//intermediate variable with a value of the indicator of request to save the run data in files
		boolean   train_ok;//indicator whether the training was OK
		mESN       esn;//object with the generated echo-state network
		
		do{//main loop of the thread
			try{sleep(1);}
			catch(InterruptedException e)
			{//simply continue the thread
			}
			//run shall be started only after setting of its parameters
			if(_do_run)
			{
				do{
					//generate an ESN or load it
					esn = new mESN(_file_esn_load, _exp_param, _exp_output, _exp_output.getCommonPathRun(_idx_exp, _idx_run), _seed_curr, _seed_load);

					//store the statistics and run-specific parameters for the output
					_exp_output.setRunSeed(_idx_run, _seed_curr);
					
					/*store initially used seeding values*/
					if(_run_repeats==0)
					{
						_seed_init = _seed_curr.clone();
						//initialize an array to keep the statistics for each interval of the sequence
						initRunAfterStart(esn.getNumOutputNeurons(), esn.getNumModules(), _seq_active.getSize());
						resetSeedLoad();//reset requests for loading seeding values
						                //=> seeding values will not be loaded in ESN again
						                //=> under repetitions, the seeding values will only be changed by increments 
					}

					//proceed with the training of ESN, only if its reservoir was successfully scaled
					train_ok = true;
					if(esn.getNonZeroSR()==true)
					{
						//apply the ESN to the loaded sequences
						train_ok = switchRun(esn);
						
						if(train_ok==true)
						{	
							//save data generated in one run
							f_save = (Boolean)_exp_param.getParamVal(exp_param_E.EP_IO_SAVE_DATA, req_val_E.RV_CUR);
							if(f_save)
							{
								//close saving of reservoir states obtained while running on TRAIN, TEST and CONFIG sequences
								esn.closeSavingEsnState();
								
								//save an ESN
								esn.saveEsn(_idx_run);
							}
							
							//store the statistics for the output
							if(esn.getNumOutputNeurons() > 1)
							{
								System.out.println("ExpRun.run: currently performance indicators are saved only for 1st output element");
								System.exit(1);
							}
							_exp_output.setRunRepeat(_idx_run, _run_repeats);
						}
						else
						{
							//take another seeding value and try to perform the experiment again
							if(_run_repeats < _run_repeats_max)
							{
								updateSeed(esn);
								_run_repeats++;
							}
							else
							{
								System.err.println("run: impossible to train the randomly generated ESN");
								System.exit(1);
							}
						}
					}
					else
					{
						//take another seeding value and try to perform the experiment again
						if(_run_repeats < _run_repeats_max)
						{
							updateSeed(esn);
							_run_repeats++;
						}
						else
						{
							System.err.println("run: impossible to get random reservoir with non-zero spectral rad");
							System.exit(1);
						}
					}
				}while((esn.getNonZeroSR()==false || train_ok==false) && _run_repeats <= _run_repeats_max);
				
				resetRunData();//It must be called before "_do_run = false;" because a state of "_do_run" is checked
	                           //in functions "startRun()", "setIdxExp()" and "setIdxRun()" that are called
				               //from the class "ExpMain".
	                           //A call to "resetRunData()" after "_do_run = false" would sometimes overwrite
	                           //the variables "_idx_exp" and "_idx_run" that were set in calls to "setIdxExp()" and
	                           //"setIdxRun()".
				_do_run = false;
			}
		}while(_do_finish==false);//loop until it is required to be finished
	}
	
	/**
	 * The function stores initial seeding values of the current run in the provided array.
	 * The function issues an error message if the provided array and a run's storage of initial values have
	 * different sizes. 
	 * 
	 * @param seed_init: provided array to store initial seeding values
	 */
	public void getInitialSeed(int[] seed_init)
	{
		int i;
		
		if(seed_init.length!=_seed_init.length)
		{
			System.err.println("ExpRun.getInitialSeed: incorrect size of array for initial seeding values");
			System.exit(1);
		}
		
		for(i=0; i<_seed_init.length; i++)
		{
			seed_init[i] = _seed_init[i];
		}
	}
	
	/**
	 * set a required index of the processed run
	 * @param idx_run, index of processed run
	 */
	public void setIdxRun(int idx_run)
	{
		if(_do_run==false)
		{
			_idx_run = idx_run;
		}
		else
		{
			System.err.println("setIdxRun: previous run is not finished yet");
			System.exit(1);
		}
	}
	
	/**
	 * set a required index of the processed vector of the experiment parameters
	 * @param idx_exp, index of parameter vector
	 */
	public void setIdxExp(int idx_exp)
	{
		if(_do_run==false)
		{
			_idx_exp = idx_exp;
		}
		else
		{
			System.err.println("setIdxExp: previous run is not finished yet");
			System.exit(1);
		}
	}
	
	/**
	 * set sequence data which should be used in the next experiment run
	 * @param exp_seq: objects containing loaded sequences
	 */
	public void setSeq(ExpSeq exp_seq)
	{
		String seq_names;//string containing sequence names
		
		if(_do_run==false)
		{
			//retrieve a sequence from a certain file if there is a multi-file sequence
			if(exp_seq.isMiltifileSeq()==false)
			{
				_exp_seq = exp_seq;
				seq_names = "{sequences_are_the_same_in_all_runs}";
			}
			else
			{
				_exp_seq = new ExpSeq(exp_seq, _idx_run);
				seq_names = _exp_seq.getAllSeqNamesAsStr();
			}
			_exp_output.setRunSeq(_idx_run, seq_names);
		}
		else
		{
			System.err.println("setSeq: previous run is not finished yet");
			System.exit(1);
		}
	}
	
	/**
	 * The function assigns required indices of the active sequences for the next run.
	 * @param seq_active: required indices of the active sequences
	 */
	public void setSeqActive(vector_C seq_active)
	{
		_seq_active = seq_active;
	}
	
	/**
	 * The function assigns required sequence purposes for the next run.
	 * @param seq_purpose: required sequence purposes
	 */
	public void setSeqPurpose(vector_C seq_purpose)
	{
		_seq_purpose = seq_purpose;
	}
	
	/**
	 * The function assigns a vector of requests for OFB distortion on sequences in the next run.
	 * @param seq_purpose: vector of requests for OFB distortion
	 */
	public void setSeqDistort(vector_C seq_distort)
	{
		_seq_distort = seq_distort;
	}
	
	/**
	 * The function assigns required indices of the first and last samples for the next run.
	 * @param seq_start: required indices of the first samples
	 * @param seq_stop:  required indices of the last samples
	 */
	public void setSeqBorder(vector_C seq_start, vector_C seq_stop)
	{
		_seq_start = seq_start;
		_seq_stop  = seq_stop;
	}

	/**
	 * This function sets border values for seeding the random generators in all tries of the next run. All values
	 * between the initial and maximum values can be used for seeding the random generators. A single value is used for
	 * seeding the random generators in a single try to perform current run of the experiment. Several tries of
	 * the current run can be performed. If in the current run the randomly generated ESN does not satisfy some
	 * criteria then this run can be started once more. To generate an ESN with other properties, the random generators
	 * must be seeded with another value.
	 * Normally another seeding value is obtained through incrementing the current seeding value until the seeding
	 * value reaches the maximum one. In the latter case all available seeding values were used. If the properties of
	 * none of the generated ESNs satisfied some criteria then it does not make sense to continue the experiment
	 * because ESNs with the desired properties cannot be generated.
	 * An example of such properties can be that the spectral radius of the randomly generated reservoir is larger than
	 * "0". Such reservoirs are difficult to generate, if the reservoir is small and has a very low connectivity.
	 * The seeding values will be only used when a loaded seeding value should not be used. The loaded seeding value is
	 * used if the provided request is set (equals true).
	 * 
	 * @param seed_init: array with initial values for seeding random generators of all ESN modules
	 * @param load_seed: array of requests for all ESN modules to ignore provided seeding values and to use loaded ones
	 * @param run_repeats_max: largest allowed number of repetitions of the current run
	 */
	public void setSeedRand(int seed_init[], boolean[] load_seed, int run_repeats_max)
	{
		int i;
		int num_sub;//number of modules in the current run
		
		if(_do_run==false)
		{
			num_sub = load_seed.length;
			
			//set a seeding value for each module
			_seed_curr = new int[num_sub];
			for(i=0; i<num_sub; i++)
			{
				_seed_curr[i] = seed_init[i];
			}
			
			_seed_load = load_seed.clone();
			_run_repeats_max = run_repeats_max;
			//initialize a counter of run repetitions
			_run_repeats = 0;
		}
		else
		{
			System.err.println("setSeedRand: previous run is not finished yet");
			System.exit(1);
		}
	}
	
	/**
	 * The function uses a provided directory path and provided file names to set paths to files with ESN modules
	 * to be loaded.
	 * The function reset a path if there is no request to load the corresponding module.
	 * 
	 * @param path_dir: path to a directory with ESN modules to be loaded
	 * @param esn_files: array of file names with ESN modules to be loaded
	 * @param load_esn: array of requests to load ESN modules
	 */
	public void setFileEsnLoad(String path_dir, String[] esn_files, boolean[] load_esn)
	{
		int i;
		int num_files;//number of ESN files to be loaded 
		
		num_files = esn_files.length;
		_file_esn_load = new String[num_files];
		
		for(i=0; i<num_files; i++)
		{
			if(load_esn[i]==true)
			{
				_file_esn_load[i] = path_dir + File.separator + esn_files[i];
			}
			else
			{
				_file_esn_load[i] = path_dir + File.separator + "*";
			}
		}
	}
	
	/**
	 * set an indicator that the object can start processing the next run;
	 * This function should be called only after the parameters for the next run have been provided. 
	 */
	public void startRun()
	{
		if(_do_run==false)
		{			
			_do_run = true;			
		}
		else
		{
			System.err.println("startRun: previous run is not finished yet");
			System.exit(1);
		}
	}
}
