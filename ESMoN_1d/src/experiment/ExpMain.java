package experiment;

import MathStat.StatCommon;

import types.vector_C;

import experiment.ExpOutput.stat_exp_E;
import experiment.ExpOutput.stat_run_E;
import experiment.ExpParam.exp_param_E;
import experiment.ExpParam.req_val_E;

/**
 * This class is the main class of RC_Experimen. It distributes the runs among available processors and prepares
 * the statistics of the whole experiment for saving.  
 * @author Danil Koryakin
 */
public class ExpMain
{
	int num_of_processors;//number of available processors for parallelization
	int[] _seed;//array of current seeding values for all ESN modules
	boolean _seed_confirmed;//indicator that an initial seed has been confirmed in the first run
	
	final int _seed_initial = 0;//initial seeding value for all ESN modules
	
	ExpRun[]  _exp_run;
	ExpParam  _exp_param;
	ExpOutput _exp_output;
	StatCommon  _math_stat;
	
	public ExpMain()
	{
		int i;
		Runtime runtime = Runtime.getRuntime();//variable to get a number of available processors

		//TODO: multi-processor option must be restored
	    num_of_processors = runtime.availableProcessors();
		//num_of_processors = 1;
		
		_seed_confirmed = false;
		
		_exp_param  = new ExpParam();
		_math_stat  = new StatCommon();
		_exp_output = new ExpOutput(_exp_param);
		_exp_run    = new ExpRun[num_of_processors];
		for(i=0; i<num_of_processors; i++)
		{
			_exp_run[i] = new ExpRun(_exp_param, _exp_output, _math_stat);
			_exp_run[i].start();
		}
	}
	
	/**
	 * check whether all runs are ready
	 * @return, "true", is all runs are ready; "false" is otherwise
	 */
	private boolean checkAllRunsReady()
	{
		int i;
		boolean all_ready;
		
		all_ready = true;
		for(i=0; i<_exp_run.length && all_ready==true; i++)
		{
			all_ready = _exp_run[i].isReady();
		}
		return all_ready; 
	}
	
	/**
	 * find a run object which is ready to perform the next runs
	 * @return, pointer to ready object of run
	 */
	private ExpRun getNextRunReady()
	{
		int i;
		ExpRun exp_run;
		
		exp_run = null;
		for(i=0; i<_exp_run.length && exp_run==null; i++)
		{
			//allow further runs only after confirming initial seeding values in the first run
			if(_seed_confirmed==true || i==0)
			{
				if(_exp_run[i].isReady())
				{
					exp_run = _exp_run[i];
				}
			}
		}
		return exp_run; 
	}
	
	/**
	 * The function initializes initial seeding values with a predefined value.
	 * Currently the initial value is set to "0".
	 * 
	 * @param num_sub: number of required ESN modules
	 */
	private void prepareInitialSeed(int num_sub)
	{
		int i;
		
		_seed = new int[num_sub];
		for(i=0; i<num_sub; i++)
		{
			_seed[i] = _seed_initial;
		}
	}
	
	/**
	 * The function assigns initial seeding values. They are either "0" (as defined in ExpMain)
	 * or loaded with ESN modules in the very first run.
	 * Seeding values of further runs are simply increments of initial values regardless their origins.
	 * The function increments the seeding  values that were obtained in the 1st run. The increment is done using
	 * a provided max number of repetitions.
	 * The function resets requests for loading seeding values after confirmation of initial seeds.
	 * 
	 * This function is needed because ESNs are loaded in Runs, not in ExpMain. 
	 * 
	 * ATTENTION: This function checks for run index "1" because this is the next index after the very 1st run.
	 * 
	 * Precondition: The function must be called in the current run is finished. The current run is always the 1st one
	 * because "1" is an index of the 2nd run to be started.
	 * 
	 * @param run_idx: index of a run to be assigned
	 * @param run_repeat_max: max number of run repetitions
	 * @param load_seed: requests to load seeding values
	 */
	private void confirmInitialSeed(int run_idx, int run_repeat_max, boolean[] load_seed)
	{
		int i;
		
		if(run_idx==1)//is the 2nd run waiting?
		{
			_exp_run[0].getInitialSeed(_seed);
			updateInitialSeed(run_repeat_max);
			_seed_confirmed = true;

			//reset requests to load seeds in the next runs
			for(i=0; i<load_seed.length; i++)
			{
				load_seed[i] = false;
			}
		}
	}
	
	/**
	 * The function computes seeding values for the next run.
	 * The new values are computed by adding a provided value to them.
	 * 
	 * @param add_value: value for adding to each seeding value
	 */
	private void updateInitialSeed(int add_value)
	{
		int i;
		
		for(i=0; i<_seed.length; i++)
		{
			_seed[i] += (add_value + 1);
		}
	}
	
	/*
	 * load experiment parameters; run experiment for each experiment parameter value  
	 */
	private void doExperiment()
	{
		Integer  i, j;
		double   tmp_double;
		int      tmp_int;
		int      idx_interv_trn;//index of train sequence interval
		int      idx_interv_tst;//index of test sequence interval
		int      idx_interv_cfg;//index of configuration sequence interval
		int      num_runs;//number of runs per combination of experiment parameters
		int      run_repeat_max;//max number of repetitions of a single run
		int      run_repeat_total;//total number of run repetitions for current parameter vector
		int      num_vect;//number of different vectors of experiment parameters
		boolean  all_runs_done;//indicator that all runs of the current parameter vector are done
		boolean[] do_save;//array with indicators whether the corresponding performance statistics should be saved
		boolean[] load_seed;//request to use a loaded seeding value for each ESN module
		boolean[] load_esn;//requests for loading ESN modules
		ExpSeq   exp_seq;//object with loaded sequences for the experiment
		ExpRun   exp_run;//object of run which is ready
		String   name_param_file;//path to a parameter file to be saved
		String[] seq_files;//array of names of sequence files
		String[] esn_files;//array of paths, including filenames, to ESN files to be loaded
		vector_C seq_purpose;//vector of sequence purposes in the current parameter vector
		vector_C seq_start;//vector of the indices of the first samples
		vector_C seq_stop;//vector of the indices of the last samples
		vector_C seq_active;//indices of sequences to be activated when it is required
		vector_C seq_distort;//vector of requests for OFB distortions on all sequences
		
		//variable used to get a value of an indexed statistics (value and its array index)
		StatCommon.idx_stat_C idx_stat;
		
		//create main output directory
		_exp_output.createDirMain();
		name_param_file = _exp_param.getNameParamFile();
		_exp_output.saveParamFile(name_param_file);
		
		//save vectors of provided experiment parameters
		_exp_output.saveParamVectors();
		
		//initialize indicators for saving
		do_save = new boolean[stat_exp_E.values().length];
		for(stat_exp_E stat_exp : stat_exp_E.values())
		{
			do_save[stat_exp.ordinal()] = _exp_output.getStatExpSave(stat_exp);
		}
		
		num_vect = _exp_param.getNumParamVectors();
		//go over all settings of the experiment parameters
		_exp_param.setFirstParamVector();
		
		for(i = 0; i < num_vect; i++)
		{
			//create directory for the current vector of experiment parameters
			_exp_output.createDirParam(i);
			
			seq_files= ((vector_C)_exp_param.getParamVal(exp_param_E.EP_PERFORM_DATA, req_val_E.RV_CUR)).getArrayStr();
			seq_purpose = (vector_C)_exp_param.getParamVal(exp_param_E.EP_PERFORM_SEQ_PURPOSE, req_val_E.RV_CUR);
			seq_start   = (vector_C)_exp_param.getParamVal(exp_param_E.EP_PERFORM_SEQ_START, req_val_E.RV_CUR);
			seq_stop    = (vector_C)_exp_param.getParamVal(exp_param_E.EP_PERFORM_SEQ_STOP, req_val_E.RV_CUR);
			seq_active  = (vector_C)_exp_param.getParamVal(exp_param_E.EP_PERFORM_SEQ_TARGET_NUM, req_val_E.RV_CUR);
			seq_distort = (vector_C)_exp_param.getParamVal(exp_param_E.EP_DISTORT_SEQ, req_val_E.RV_CUR);
			
			//every sequence is loaded under the construction of the corresponding ExpData object
			exp_seq = new ExpSeq(_exp_param.getPathSeq(), seq_files);
			
			esn_files = ((vector_C)_exp_param.getParamVal(exp_param_E.EP_IO_LOAD_ESN_FILE, req_val_E.RV_CUR)).getArrayStr();
			load_esn  = ((vector_C)_exp_param.getParamVal(exp_param_E.EP_IO_LOAD_ESN, req_val_E.RV_CUR)).getArrayBoolean();
			load_seed = ((vector_C)_exp_param.getParamVal(exp_param_E.EP_IO_LOAD_SEED, req_val_E.RV_CUR)).getArrayBoolean();
			prepareInitialSeed(load_seed.length);
			
			//make defined number of runs for current setting of experiment parameters
			num_runs       = (Integer)_exp_param.getParamVal(exp_param_E.EP_PERFORM_RUNS      , req_val_E.RV_CUR);
			run_repeat_max = (Integer)_exp_param.getParamVal(exp_param_E.EP_PERFORM_RUN_REPEAT, req_val_E.RV_CUR);
			idx_interv_trn = _exp_param.getIdxInterv('t');
			idx_interv_tst = _exp_param.getIdxInterv('e');
			idx_interv_cfg = _exp_param.getIdxInterv('c');
			j = 0;
			all_runs_done = false;
			//seeding values must be confirmed for each new parameter vector because a new seeding value can be loaded
			_seed_confirmed = false;
			while(all_runs_done==false)
			{
				if(j<num_runs)
				{
					exp_run = getNextRunReady();
					if(exp_run!=null)
					{
						confirmInitialSeed(j, run_repeat_max, load_seed);
						
						exp_run.setIdxExp(i);
						exp_run.setIdxRun(j);
						exp_run.setSeq(exp_seq);
						exp_run.setSeqActive(seq_active);
						exp_run.setSeqPurpose(seq_purpose);
						exp_run.setSeqDistort(seq_distort);
						exp_run.setSeqBorder(seq_start, seq_stop);
						//all values from "seed" to "seed + num_run_repeat" are allowed seeding values
						exp_run.setSeedRand(_seed, load_seed, run_repeat_max);
						exp_run.setFileEsnLoad(_exp_param.getPathEsn(), esn_files, load_esn);
						//compute initial seeding value for the next run
						//ATTENTION: This call has no effect in 1st run because after this run confirmed initial values
						//           will be updated from "confirmInitialSeed()".
						updateInitialSeed(run_repeat_max);
						
						exp_run.startRun();
						j++;
					}
				}
				else
				{
					//wait until all runs for the current parameter vector are done
					all_runs_done = checkAllRunsReady();
				}
			}//for "j" (experiment runs)
			
			//assign performance on the test sequence for the whole experiment
			if(idx_interv_tst!=-1)
			{
				//average and sum of values
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_MSE_AVG_TST,       i, tmp_double);			
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_NRMSE_AVG_TST,     i, tmp_double);
				tmp_int = (int)_math_stat.computeSum(_exp_output.getRunPerform(stat_run_E.SR_SEL, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_SEL_SUM_TST,       i, tmp_int);
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_RMSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_RMSE_AVG_TST,     i, tmp_double);
				
				//standard deviations
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_MSE_TST,   i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_NRMSE_TST, i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_SEL, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_SEL_TST,   i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_RMSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_RMSE_TST, i, tmp_double);
				
				//best values
				idx_stat = _math_stat.findMin(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_MIN_MSE_TST,     i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_MIN_MSE_TST, i, idx_stat.idx);
				idx_stat = _math_stat.findMin(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_MIN_NRMSE_TST,     i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_MIN_NRMSE_TST, i, idx_stat.idx);
				idx_stat = _math_stat.findMax(_exp_output.getRunPerform(stat_run_E.SR_SEL, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_BEST_SEL_TST,       i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_BEST_SEL_TST,   i, idx_stat.idx);
				//worst values
				idx_stat = _math_stat.findMax(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_MAX_MSE_TST,     i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_MAX_MSE_TST, i, idx_stat.idx);
				idx_stat = _math_stat.findMax(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_MAX_NRMSE_TST,     i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_MAX_NRMSE_TST, i, idx_stat.idx);
				//median values
				idx_stat = _math_stat.findMedian(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_MED_MSE_TST,     i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_MED_MSE_TST, i, idx_stat.idx);
				idx_stat = _math_stat.findMedian(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_tst));
				_exp_output.setStatExp(stat_exp_E.SE_MED_NRMSE_TST,     i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_MED_NRMSE_TST, i, idx_stat.idx);
				
				//number of not outlier runs
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_CIOK_INCL_ERR, idx_interv_tst));
				tmp_double*= 100;//to save identification rate in %
				_exp_output.setStatExp(stat_exp_E.SE_CIOK_AVG_BY_TST_ERR, i, tmp_double);
			}
			else
			{
				do_save[stat_exp_E.SE_MSE_AVG_TST.ordinal()]          = false;
				do_save[stat_exp_E.SE_NRMSE_AVG_TST.ordinal()]        = false;
				do_save[stat_exp_E.SE_SEL_SUM_TST.ordinal()]          = false;
				do_save[stat_exp_E.SE_RMSE_AVG_TST.ordinal()]         = false;
				do_save[stat_exp_E.SE_DEV_MSE_TST.ordinal()]          = false;
				do_save[stat_exp_E.SE_DEV_NRMSE_TST.ordinal()]        = false;
				do_save[stat_exp_E.SE_DEV_SEL_TST.ordinal()]          = false;
				do_save[stat_exp_E.SE_DEV_RMSE_TST.ordinal()]         = false;
				do_save[stat_exp_E.SE_MIN_MSE_TST.ordinal()]          = false;
				do_save[stat_exp_E.SE_IDX_MIN_MSE_TST.ordinal()]      = false;
				do_save[stat_exp_E.SE_MAX_MSE_TST.ordinal()]          = false;
				do_save[stat_exp_E.SE_IDX_MAX_MSE_TST.ordinal()]      = false;
				do_save[stat_exp_E.SE_MED_MSE_TST.ordinal()]          = false;
				do_save[stat_exp_E.SE_IDX_MED_MSE_TST.ordinal()]      = false;
				do_save[stat_exp_E.SE_MIN_NRMSE_TST.ordinal()]        = false;
				do_save[stat_exp_E.SE_IDX_MIN_NRMSE_TST.ordinal()]    = false;
				do_save[stat_exp_E.SE_MAX_NRMSE_TST.ordinal()]        = false;
				do_save[stat_exp_E.SE_IDX_MAX_NRMSE_TST.ordinal()]    = false;
				do_save[stat_exp_E.SE_MED_NRMSE_TST.ordinal()]        = false;
				do_save[stat_exp_E.SE_IDX_MED_NRMSE_TST.ordinal()]    = false;
				do_save[stat_exp_E.SE_BEST_SEL_TST.ordinal()]         = false;
				do_save[stat_exp_E.SE_IDX_BEST_SEL_TST.ordinal()]     = false;
				do_save[stat_exp_E.SE_CIOK_AVG_BY_TST_ERR .ordinal()] = false;
			}
			
			//assign performance on the training sequence for the whole experiment
			if(idx_interv_trn!=-1)
			{
				//average and sum of values
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_NRMSE_AVG_TRN,     i, tmp_double);
				tmp_int = (int)_math_stat.computeSum(_exp_output.getRunPerform(stat_run_E.SR_SEL, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_SEL_SUM_TRN,       i, tmp_int);
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_MSE_AVG_TRN,       i, tmp_double);
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_RMSE, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_RMSE_AVG_TRN,     i, tmp_double);
				
				//standard deviations
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_NRMSE_TRN, i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_SEL, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_SEL_TRN,   i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_MSE_TRN,   i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_RMSE, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_RMSE_TRN, i, tmp_double);
				
				//best values
				idx_stat = _math_stat.findMax(_exp_output.getRunPerform(stat_run_E.SR_SEL, idx_interv_trn));
				_exp_output.setStatExp(stat_exp_E.SE_BEST_SEL_TRN,       i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_BEST_SEL_TRN,   i, idx_stat.idx);
			}
			else
			{
				do_save[stat_exp_E.SE_NRMSE_AVG_TRN.ordinal()]     = false;
				do_save[stat_exp_E.SE_SEL_SUM_TRN.ordinal()]       = false;
				do_save[stat_exp_E.SE_MSE_AVG_TRN.ordinal()]       = false;
				do_save[stat_exp_E.SE_RMSE_AVG_TRN.ordinal()]      = false;
				do_save[stat_exp_E.SE_DEV_NRMSE_TRN.ordinal()]     = false;
				do_save[stat_exp_E.SE_DEV_SEL_TRN.ordinal()]       = false;
				do_save[stat_exp_E.SE_DEV_MSE_TRN.ordinal()]       = false;
				do_save[stat_exp_E.SE_DEV_RMSE_TRN.ordinal()]      = false;
				do_save[stat_exp_E.SE_BEST_SEL_TRN.ordinal()]      = false;
				do_save[stat_exp_E.SE_IDX_BEST_SEL_TRN.ordinal()]  = false;
			}
			
			//assign performance indicators from the configuration sequence for the whole experiment
			if(idx_interv_cfg!=-1)
			{
				//average values
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_MSE_AVG_CFG,       i, tmp_double);
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_NRMSE_AVG_CFG,     i, tmp_double);
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_RMSE, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_RMSE_AVG_CFG,     i, tmp_double);
				
				//standard deviations
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_MSE, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_MSE_CFG,   i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_NRMSE, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_NRMSE_CFG, i, tmp_double);
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_RMSE, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_RMSE_CFG, i, tmp_double);
				
				//sum of values
				tmp_int = (int)_math_stat.computeSum(_exp_output.getRunPerform(stat_run_E.SR_LEL, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_LEL_SUM_CFG,       i, tmp_int);
			    //standard deviations
				tmp_double = _math_stat.computeStdDev(_exp_output.getRunPerform(stat_run_E.SR_LEL, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_DEV_LEL_CFG,   i, tmp_double);
			    //best values
				idx_stat = _math_stat.findMin(_exp_output.getRunPerform(stat_run_E.SR_LEL, idx_interv_cfg));
				_exp_output.setStatExp(stat_exp_E.SE_BEST_LEL_CFG,       i, idx_stat.val);
				_exp_output.setStatExp(stat_exp_E.SE_IDX_BEST_LEL_CFG,   i, idx_stat.idx);
				
				//component identification rate
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_CIOK_INCL_ERR, idx_interv_cfg));
				tmp_double*= 100;//to save identification rate in %
				_exp_output.setStatExp(stat_exp_E.SE_CIOK_AVG_BY_CFG_ERR, i, tmp_double);
				tmp_double = StatCommon.computeMean(_exp_output.getRunPerform(stat_run_E.SR_CIOK, idx_interv_cfg));
				tmp_double*= 100;//to save identification rate in %
				_exp_output.setStatExp(stat_exp_E.SE_CIOK_AVG_CFG, i, tmp_double);
			}
			else
			{
				do_save[stat_exp_E.SE_MSE_AVG_CFG.ordinal()]      = false;
				do_save[stat_exp_E.SE_NRMSE_AVG_CFG.ordinal()]    = false;
				do_save[stat_exp_E.SE_RMSE_AVG_CFG.ordinal()]     = false;
				do_save[stat_exp_E.SE_DEV_MSE_CFG.ordinal()]      = false;
				do_save[stat_exp_E.SE_DEV_NRMSE_CFG.ordinal()]    = false;
				do_save[stat_exp_E.SE_DEV_RMSE_CFG.ordinal()]     = false;
				do_save[stat_exp_E.SE_LEL_SUM_CFG.ordinal()]      = false;
				do_save[stat_exp_E.SE_DEV_LEL_CFG.ordinal()]      = false;
				do_save[stat_exp_E.SE_BEST_LEL_CFG.ordinal()]     = false;
				do_save[stat_exp_E.SE_IDX_BEST_LEL_CFG.ordinal()] = false;
				do_save[stat_exp_E.SE_CIOK_AVG_BY_CFG_ERR.ordinal()] = false;
				do_save[stat_exp_E.SE_CIOK_AVG_CFG.ordinal()]        = false;
			}

			//compute and assign a total number of run repetitions
			run_repeat_total = (int)_math_stat.computeSum(_exp_output.getRunRepeat());
			_exp_output.setStatExp(stat_exp_E.SE_RUN_REPEAT_TOTAL,   i, run_repeat_total);
			
			//Command-Line info
			_exp_output.printParamVal(i);
			
			//save data from all runs for current value of the experiment parameter
			_exp_output.saveStatRun(i);
			
			_exp_param.setNextParamVector();
		}//for "i" (experiment parameters)
		
		//set indicators for saving the statistics of the experiment
		for(stat_exp_E stat_exp : stat_exp_E.values())
		{
			_exp_output.setStatExpSave(stat_exp, do_save[stat_exp.ordinal()]);
		}
		
		//save the statistics
        _exp_output.saveStatExp();
        
        System.out.println("experiment is successfully completed");
	}
	
	/**
	 * close allocated runs
	 */
	private void terminateRuns()
	{
		int i;
		//delete separate runs
		for(i=0; i<_exp_run.length; i++)
		{
			_exp_run[i].finishThread();
		}
	}
	
	public static void main(String[] args)
	{
		ExpMain experiment;//instance of experiment
		
		experiment = new ExpMain();
		experiment.doExperiment();
		experiment.terminateRuns();
	}
}