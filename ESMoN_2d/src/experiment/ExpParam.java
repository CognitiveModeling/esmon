package experiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;

import adaptation.DiffEvolutionParam.config_ea_init_E;
import adaptation.DiffEvolutionParam.config_ea_order_method_E;

import MathDiff.MathNoise;
import MathDiff.MathDistortion.distortion_E;
import MathDiff.MathNoise.noise_E;

import types.interval_C;
import types.multi_val_C;
import types.vector_C;

import esn.Activation.activation_E;
import esn.EsnModule.config_method_E;
import esn.Layer.leakage_assign_E;
import esn.mESN.config_ea_mode_E;
import esn.ReservoirInitialization;

/**
 * This class implements functions to load the provided parameter values from a file.
 * @author Danil Koryakin
 *
 */
public class ExpParam
{	
	public enum req_val_E 
	{
		RV_DEF,//default value
		RV_CUR //current value
	};
	
	public enum exp_param_E
	{
		//*** parameters of ESN module
		
		EP_ACTIVATION,//attribute of all layers
		EP_ACTIVATION_LOGISTIC_PARAM,//parameter of logistic function as activation
		EP_SIZE,      //attribute of all layers
		EP_BIAS,      //attribute of all layers
		EP_LEAKAGE_RATE,  //attribute of all layers
		EP_LEAKAGE_ASSIGN,//attribute of all layers; type of leakage assignment: the same for all or random
		EP_RES_CONNECT,
		EP_RES_SPECTR_RAD,
		EP_RES_ACTIVATION_RATIO,//(%) amount of reservoir neurons with specified activation
		EP_RES_TOPOLOGY,//choice of method for generation of reservoir
		EP_RES_W,
		EP_SUBRES_CONNECT_OUTSIDE,
		EP_OFB_USE,
		EP_OFB_W,
		EP_INPUT_USE,
		EP_INPUT_W,
		EP_INPUT_SIZE,//number of input neurons
		EP_OUTPUT_SIZE,//number of output neurons
		EP_OUTPUT_ACTIVATION,
		EP_OUTPUT_RECURRENCE,//recurrent connections of output neurons
		
		//*** parameters of SIN module
		
		EP_SIN_PARAM_SIZE, //attribute of all layers
		EP_SIN_PARAM,      //frequency and phase
		
		//*** module-type independent parameters
		
		EP_PERFORM_RUNS,//how many times the experiment should be run for the same combination of experiment parameters
		EP_PERFORM_RUN_REPEAT,//max number of times a run can be repeated until ESN fulfills some predefined conditions
		EP_PERFORM_SEL_THRESH,//threshold to find small error length (SEL)
		EP_PERFORM_LEL_THRESH,//threshold to find large error length (LEL)
		EP_PERFORM_DATA,
		EP_PERFORM_SEQ_TARGET_NUM,//number of waves in a target dynamics
		EP_PERFORM_SEQ_START,
		EP_PERFORM_SEQ_STOP,
		EP_PERFORM_SEQ_PURPOSE,
		EP_PERFORM_NOISE_TYPE,
		EP_PERFORM_NOISE_BOUNDS,
		EP_CONFIG_METHOD,//method for configuration of mESN
		EP_CONFIG_EA_MODE,//mode of evolutionary configuration (CONST force, INTERLEAVE of forcing and free-run)
		EP_CONFIG_EA_POP_SIZE,//population size
		EP_CONFIG_EA_DIFF_WEIGHT,//differential weight
		EP_CONFIG_EA_CROSS_PROB,//crossover probability
		EP_CONFIG_EA_NUM_GEN,//number of generations
		EP_CONFIG_EA_INIT,//initialization at every time step (none, random, hyper-uniform, hyper-norm)
		EP_CONFIG_EA_FIT_LEN,//number of time steps to compute fitness of an individual
		EP_CONFIG_EA_MARGIN_MODULE,//maximum exaggeration of a valid range of a module states and outputs
		EP_CONFIG_EA_COEVOL_ORDER_METHOD,//method to arrange an order of modules under co-evolution: random or normal
		EP_CONFIG_EA_COEVOL_ORDER,//order of modules under co-evolution, e.g. {0,1,2,...}
		EP_CONFIG_EA_COEVOL_SORT_MIN_NUM,//minimum number of individuals to be sorted
		EP_CONFIG_EA_VALIDATION_LENGTH,//min number of time steps for a genotype to stay in the maturity pool
		EP_CONFIG_EA_CIOK_LEVEL,//largest config MSE to decide that component identification is really correct
		EP_CONFIG_EA_ACTIV_THRESH,//magnitude of module output to reach to start active suppression of that module
		EP_DISTORT_METHOD,//method to vary distribution of OFB among the sub-reservoirs
		EP_DISTORT_PERIOD,//number of time steps to repeat the OFB distortion
		EP_DISTORT_STRENGTH,//(%)strength of constant distortion oder initial variance of Gaussian distribution
		EP_DISTORT_DECAY,//coefficient of exponential variance decay or variance at end of a sequence
		EP_DISTORT_SEQ,//OFB distortion requests for all sequence
		EP_IO_SAVE_DATA,//indicator that ESN states should be saved
		EP_IO_LOAD_ESN,//indicator that ESNs should be loaded
		EP_IO_LOAD_SEED,//request to use a seed value from a loaded ESN
		EP_IO_LOAD_ESN_FILE,//files with ESNs to be loaded
		EP_IO_LOAD_FFANN_FILE_OUT_0,//files with FF-ANNs to model the output weights at ESN's output 0
		EP_IO_LOAD_FFANN_FILE_OUT_1,//files with FF-ANNs to model the output weights at ESN's output 1
	};
	
	private class param_C
	{
		String   name = "";		
		int      idx_value_cur = 0;//index of current parameter value; its value is meaningless, if no values were provided
		Object[] value = null;//array of parameter values
		Object   value_def;//default parameter value
		Object   value_cur;//current value of the experiment parameter; it equals to default, if no values were loaded
	};
	
	//array of continuous parameter values 
	private param_C[] _param = null;
	
	//experiment attributes and parameters which cannot be loaded	
	private String  _exp_user_comment = null;//additional user comment that can be inserted in the output file header
	private String  _dir_seq = null;//path to directory with sequences
	private String  _dir_esn = null;//path to a directory with ESN files
	private int     _num_param_vectors;//total number of combinations of the loaded parameter values
	
	final private String _name_param_file = "ExpParam.dat";//name of a file with parameters of experiment

	public ExpParam()
	{
		int i;
		Vector<String> filedata;//data loaded from the file
		
		//allocate arrays for parameter values
		_param = new param_C[exp_param_E.values().length];
		for(i=0; i<_param.length; i++)
		{
			_param[i] = new param_C();
		}

		//assign names of experiment parameters
		_param[exp_param_E.EP_ACTIVATION.ordinal()].name                = "TRANSFER_FUNCTION";
		_param[exp_param_E.EP_ACTIVATION_LOGISTIC_PARAM.ordinal()].name = "TRANSFER_FUNCTION_LOGISTIC_PARAM";
		_param[exp_param_E.EP_SIZE.ordinal()].name                    = "RES_SIZE";
		_param[exp_param_E.EP_RES_CONNECT.ordinal()].name             = "CONNECTIVITY";
		_param[exp_param_E.EP_RES_SPECTR_RAD.ordinal()].name          = "SPECTRAL_RADIUS";
		_param[exp_param_E.EP_RES_ACTIVATION_RATIO.ordinal()].name    = "TRANSFER_FUNCTION_RATIO";
		_param[exp_param_E.EP_RES_TOPOLOGY.ordinal()].name            = "TOPOLOGY";
		_param[exp_param_E.EP_RES_W.ordinal()].name                   = "RES_W_BOUNDS";
		_param[exp_param_E.EP_LEAKAGE_RATE.ordinal()].name            = "LEAKAGE_RATE";
		_param[exp_param_E.EP_LEAKAGE_ASSIGN.ordinal()].name          = "LEAKAGE_ASSIGN";
		_param[exp_param_E.EP_BIAS.ordinal()].name                    = "RESERVOIR_BIAS";
		_param[exp_param_E.EP_SUBRES_CONNECT_OUTSIDE.ordinal()].name  = "CONNECTIVITY_OUTSUB";
		_param[exp_param_E.EP_OFB_USE.ordinal()].name                 = "USE_W_BACK";
		_param[exp_param_E.EP_OFB_W.ordinal()].name                   = "BACK_BOUNDS";
		_param[exp_param_E.EP_INPUT_USE.ordinal()].name               = "USE_INPUT";
		_param[exp_param_E.EP_INPUT_W.ordinal()].name                 = "INPUT_BOUNDS";
		_param[exp_param_E.EP_INPUT_SIZE.ordinal()].name              = "INPUT_SIZE";
		_param[exp_param_E.EP_OUTPUT_SIZE.ordinal()].name             = "OUTPUT_SIZE";
		_param[exp_param_E.EP_OUTPUT_ACTIVATION.ordinal()].name       = "TRANSFER_FUNCTION_OUT";
		_param[exp_param_E.EP_OUTPUT_RECURRENCE.ordinal()].name       = "OUTPUT_LAYER_RECURRENCE";
		_param[exp_param_E.EP_SIN_PARAM_SIZE.ordinal()].name          = "SIN_PARAM_SIZE";
		_param[exp_param_E.EP_SIN_PARAM.ordinal()].name               = "SIN_PARAM";
		_param[exp_param_E.EP_PERFORM_RUNS.ordinal()].name            = "NUMBER_OF_RUNS";
		_param[exp_param_E.EP_PERFORM_RUN_REPEAT.ordinal()].name      = "NUMBER_OF_RUN_REPEAT";
		_param[exp_param_E.EP_PERFORM_SEL_THRESH.ordinal()].name      = "SEL_THRESH";
		_param[exp_param_E.EP_PERFORM_LEL_THRESH.ordinal()].name      = "LEL_THRESH";
		_param[exp_param_E.EP_PERFORM_DATA.ordinal()].name            = "FILE_WITH_SAMPLES";
		_param[exp_param_E.EP_PERFORM_SEQ_TARGET_NUM.ordinal()].name  = "SEQ_TARGET_NUM";
		_param[exp_param_E.EP_PERFORM_SEQ_START.ordinal()].name       = "SEQ_START";
		_param[exp_param_E.EP_PERFORM_SEQ_STOP.ordinal()].name        = "SEQ_STOP";
		_param[exp_param_E.EP_PERFORM_SEQ_PURPOSE.ordinal()].name     = "SEQ_PURPOSE";
		_param[exp_param_E.EP_PERFORM_NOISE_TYPE.ordinal()].name      = "NOISE_TYPE";
		_param[exp_param_E.EP_PERFORM_NOISE_BOUNDS.ordinal()].name    = "NOISE_PARAM";
		_param[exp_param_E.EP_CONFIG_METHOD.ordinal()].name           = "CONFIG_METHOD";
		_param[exp_param_E.EP_CONFIG_EA_MODE.ordinal()].name          = "CONFIG_EA_MODE";
		_param[exp_param_E.EP_CONFIG_EA_POP_SIZE.ordinal()].name      = "CONFIG_EA_POP_SIZE";
		_param[exp_param_E.EP_CONFIG_EA_DIFF_WEIGHT.ordinal()].name   = "CONFIG_EA_DIFF_WEIGHT";
		_param[exp_param_E.EP_CONFIG_EA_CROSS_PROB.ordinal()].name    = "CONFIG_EA_CROSS_PROB";
		_param[exp_param_E.EP_CONFIG_EA_NUM_GEN.ordinal()].name       = "CONFIG_EA_NUM_GEN";
		_param[exp_param_E.EP_CONFIG_EA_INIT.ordinal()].name          = "CONFIG_EA_INIT";
		_param[exp_param_E.EP_CONFIG_EA_FIT_LEN.ordinal()].name       = "CONFIG_EA_FIT_LEN";
		_param[exp_param_E.EP_CONFIG_EA_MARGIN_MODULE.ordinal()].name = "CONFIG_EA_MARGIN_MODULE";
		_param[exp_param_E.EP_CONFIG_EA_COEVOL_ORDER_METHOD.ordinal()].name    = "CONFIG_EA_COEVOL_ORDER_METHOD";
		_param[exp_param_E.EP_CONFIG_EA_COEVOL_ORDER.ordinal()].name           = "CONFIG_EA_COEVOL_ORDER";
		_param[exp_param_E.EP_CONFIG_EA_COEVOL_SORT_MIN_NUM.ordinal()].name    = "CONFIG_EA_COEVOL_SORT_MIN_NUM";
		_param[exp_param_E.EP_CONFIG_EA_VALIDATION_LENGTH.ordinal()].name      = "CONFIG_EA_VALIDATION_LENGTH";
		_param[exp_param_E.EP_CONFIG_EA_CIOK_LEVEL.ordinal()].name    = "CONFIG_EA_COMP_IDENT_OK_LEVEL";
		_param[exp_param_E.EP_CONFIG_EA_ACTIV_THRESH.ordinal()].name   = "CONFIG_EA_SUPPRESSION_LEVEL";
		_param[exp_param_E.EP_DISTORT_METHOD.ordinal()].name          = "DISTORT_METHOD";
		_param[exp_param_E.EP_DISTORT_PERIOD.ordinal()].name          = "DISTORT_PERIOD";
		_param[exp_param_E.EP_DISTORT_STRENGTH.ordinal()].name        = "DISTORT_STRENGTH";
		_param[exp_param_E.EP_DISTORT_DECAY.ordinal()].name           = "DISTORT_DECAY";
		_param[exp_param_E.EP_DISTORT_SEQ.ordinal()].name             = "DISTORT_SEQ";
		_param[exp_param_E.EP_IO_SAVE_DATA.ordinal()].name            = "SAVE_RUN_DATA";
		_param[exp_param_E.EP_IO_LOAD_ESN.ordinal()].name             = "LOAD_ESN";
		_param[exp_param_E.EP_IO_LOAD_SEED.ordinal()].name            = "LOAD_SEED";
		_param[exp_param_E.EP_IO_LOAD_ESN_FILE.ordinal()].name        = "LOAD_ESN_FILE";
		_param[exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_0.ordinal()].name= "LOAD_FFANN_FILE_OUT_0";
		_param[exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_1.ordinal()].name= "LOAD_FFANN_FILE_OUT_1";
		
		
		//assign default values of the experiment parameters
		_param[exp_param_E.EP_ACTIVATION.ordinal()].value_def                = activation_E.TANH;
		_param[exp_param_E.EP_ACTIVATION_LOGISTIC_PARAM.ordinal()].value_def = (Double)1.0;
		_param[exp_param_E.EP_SIZE.ordinal()].value_def                    = new multi_val_C(5);
		_param[exp_param_E.EP_RES_CONNECT.ordinal()].value_def             = new multi_val_C(0.4);
		_param[exp_param_E.EP_RES_SPECTR_RAD.ordinal()].value_def          = new multi_val_C(0.8);
		_param[exp_param_E.EP_RES_ACTIVATION_RATIO.ordinal()].value_def    = (Double)1.0;
		_param[exp_param_E.EP_RES_TOPOLOGY.ordinal()].value_def            = ReservoirInitialization.RANDOM;
		_param[exp_param_E.EP_RES_W.ordinal()].value_def                   = new multi_val_C(-1.0,+1.0);
		_param[exp_param_E.EP_LEAKAGE_RATE.ordinal()].value_def            = new multi_val_C(1.0);
		_param[exp_param_E.EP_LEAKAGE_ASSIGN.ordinal()].value_def          = leakage_assign_E.LA_NONE;
		_param[exp_param_E.EP_BIAS.ordinal()].value_def                    = new multi_val_C(0.0, 0.0);
		_param[exp_param_E.EP_SUBRES_CONNECT_OUTSIDE.ordinal()].value_def  = (Double)0.00;
		_param[exp_param_E.EP_OFB_USE.ordinal()].value_def                 = true;
		_param[exp_param_E.EP_OFB_W.ordinal()].value_def                   = new multi_val_C(-0.1,+0.1);
		_param[exp_param_E.EP_INPUT_USE.ordinal()].value_def               = false;
		_param[exp_param_E.EP_INPUT_W.ordinal()].value_def                 = new multi_val_C(-0.01,+0.01);
		_param[exp_param_E.EP_INPUT_SIZE.ordinal()].value_def              = (Integer)0;
		_param[exp_param_E.EP_OUTPUT_SIZE.ordinal()].value_def             = (Integer)1;
		_param[exp_param_E.EP_OUTPUT_ACTIVATION.ordinal()].value_def       = activation_E.ID;
		_param[exp_param_E.EP_OUTPUT_RECURRENCE.ordinal()].value_def       = false;
		_param[exp_param_E.EP_SIN_PARAM_SIZE.ordinal()].value_def          = (Integer)2;
		_param[exp_param_E.EP_SIN_PARAM.ordinal()].value_def               = new vector_C("{0.2, 0.0}");
		_param[exp_param_E.EP_PERFORM_RUNS.ordinal()].value_def            = (Integer)10;
		_param[exp_param_E.EP_PERFORM_RUN_REPEAT.ordinal()].value_def      = 0;//max number of repetitions till ESN OK
		_param[exp_param_E.EP_PERFORM_SEL_THRESH.ordinal()].value_def      = (Double)0.1;
		_param[exp_param_E.EP_PERFORM_LEL_THRESH.ordinal()].value_def      = (Double)0.001;
		_param[exp_param_E.EP_PERFORM_DATA.ordinal()].value_def            = new vector_C("{seq_mso2_700.dat}");
		_param[exp_param_E.EP_PERFORM_SEQ_TARGET_NUM.ordinal()].value_def  = new vector_C("{0, 0, 0}");
		_param[exp_param_E.EP_PERFORM_SEQ_START.ordinal()].value_def       = new vector_C("{ 0,100,400}");//MSO: 300 (roeschies09.pdf)
		_param[exp_param_E.EP_PERFORM_SEQ_STOP.ordinal()].value_def        = new vector_C("{99,399,699}");
		_param[exp_param_E.EP_PERFORM_SEQ_PURPOSE.ordinal()].value_def     = new vector_C("{'w', 't', 'e'}");
		_param[exp_param_E.EP_PERFORM_NOISE_TYPE.ordinal()].value_def      = noise_E.NOISE_NONE;
		_param[exp_param_E.EP_PERFORM_NOISE_BOUNDS.ordinal()].value_def    = new multi_val_C(-0.000001,+0.000001);
		_param[exp_param_E.EP_CONFIG_METHOD.ordinal()].value_def           = config_method_E.CM_None;
		_param[exp_param_E.EP_CONFIG_EA_MODE.ordinal()].value_def          = config_ea_mode_E.CFG_EA_MODE_UNKNOWN;
		_param[exp_param_E.EP_CONFIG_EA_POP_SIZE.ordinal()].value_def      = (Integer)100;
		_param[exp_param_E.EP_CONFIG_EA_DIFF_WEIGHT.ordinal()].value_def   = (Double)0.5;
		_param[exp_param_E.EP_CONFIG_EA_CROSS_PROB.ordinal()].value_def    = (Double)0.9;
		_param[exp_param_E.EP_CONFIG_EA_NUM_GEN.ordinal()].value_def       = (Integer)5;
		_param[exp_param_E.EP_CONFIG_EA_INIT.ordinal()].value_def          = config_ea_init_E.CFG_EA_INIT_UNKNOWN;
		_param[exp_param_E.EP_CONFIG_EA_FIT_LEN.ordinal()].value_def       = (Integer)1;
		_param[exp_param_E.EP_CONFIG_EA_MARGIN_MODULE.ordinal()].value_def = new vector_C("{0.0, 0.0, 0.0}");
		_param[exp_param_E.EP_CONFIG_EA_COEVOL_ORDER_METHOD.ordinal()].value_def    = config_ea_order_method_E.CFG_EA_ORDER_NORMAL;
		_param[exp_param_E.EP_CONFIG_EA_COEVOL_ORDER.ordinal()].value_def           = new vector_C("{0, 1, 2}");
		_param[exp_param_E.EP_CONFIG_EA_COEVOL_SORT_MIN_NUM.ordinal()].value_def    = (Integer)0;
		_param[exp_param_E.EP_CONFIG_EA_VALIDATION_LENGTH.ordinal()].value_def      = (Integer)0;
		_param[exp_param_E.EP_CONFIG_EA_CIOK_LEVEL.ordinal()].value_def    = (Double)0.001;
		_param[exp_param_E.EP_CONFIG_EA_ACTIV_THRESH.ordinal()].value_def  = (Double)1E-2;
		_param[exp_param_E.EP_DISTORT_METHOD.ordinal()].value_def          = distortion_E.DISTORT_NONE;
		_param[exp_param_E.EP_DISTORT_PERIOD.ordinal()].value_def          = new vector_C(1);
		_param[exp_param_E.EP_DISTORT_STRENGTH.ordinal()].value_def        = new vector_C(0.1);//10% of maximum config seq
		_param[exp_param_E.EP_DISTORT_DECAY.ordinal()].value_def           = new vector_C(0.001);//0.1%
		_param[exp_param_E.EP_DISTORT_SEQ.ordinal()].value_def             = new vector_C("{0, 0, 0}");
		_param[exp_param_E.EP_IO_SAVE_DATA.ordinal()].value_def            = false;
		_param[exp_param_E.EP_IO_LOAD_ESN.ordinal()].value_def             = new vector_C("{false,false,false}");
		_param[exp_param_E.EP_IO_LOAD_SEED.ordinal()].value_def            = new vector_C("{false,false,false}");
		_param[exp_param_E.EP_IO_LOAD_ESN_FILE.ordinal()].value_def        = new vector_C("{*,*,*}");
		_param[exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_0.ordinal()].value_def= new vector_C("{*,*,*}");
		_param[exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_1.ordinal()].value_def= new vector_C("{*,*,*}");
		
		//set default values of other parameters
		_exp_user_comment = "NONE";
		_dir_seq = "." + File.separator + "sequences";
		_dir_esn = "." + File.separator + "esn_files";
		
		//generate a new settings object with the proper data
		filedata = fromFile(new File(_name_param_file));
		
		//extract parameter values from file
		loadParamVal(filedata);
		_num_param_vectors = computeNumParamVectors();
		checkExpParam();
	}
	
	/**
	 * The function checks plausibility of experiment parameter settings.
	 * The function issues an error message and terminates execution if there are parameter settings that contradict
	 * each other.
	 */
	private void checkExpParam()
	{
		int i, j;
		int num_module;//number of required ESN modules
		int num_connect;//number of values in the vector of connectivity
		int num_sr;//number of values in the vector of spectral radius
		int num_seq_start;//number of start samples for sequence activation
		int subres_size;//reservoir size
		int num_disturb_param;//number of loaded values for a certain disturbance parameter
		int num_out;//number of the output neurons
		int num_runs_0, num_runs_cur;//number of runs for parameter vector 0 and for current parameter vector
		double connect_sub;//connectivity of a single sub-reservoir
		boolean   seq_several_files;//indicator that there are sequences with several files
		Character seq_purpose_cur;//character code of the current sequence purpose
		vector_C disturb_param;//values of a certain disturbance parameter
		vector_C res_size;//vector with sizes of all sub-reservoirs
		vector_C connect;//vector with connectivity of all sub-reservoirs
		vector_C seq_start;//vector of start samples for all sequence intervals
		vector_C seq_param;//vector of a parameter whose values should be specified for each sequence
		vector_C load_esn_param_0;//current vector of a certain parameter related to a loading of ESNs 
		vector_C load_esn_param_1;//current vector of another parameter related to a loading of ESNs
		vector_C load_ffann_param_0;//current vector of parameter related to loading of FF-ANN for output 0
		vector_C load_ffann_param_1;//current vector of parameter related to loading of FF-ANN for output 1
		vector_C vector_param;//loaded vector value
		Object[] seq_obj;//array of objects with sequence files
		ReservoirInitialization topology;//default reservoir topology
		config_method_E config_method;
		
		num_runs_0 = -1;
		seq_several_files = false;
		setFirstParamVector();
		for(i=0; i<_num_param_vectors; i++)
		{
			//check sequences only for the 1st parameter vector;
			//otherwise it can take much time to check sequences for all parameter vectors;
			//ASSUMPTION 1: sequences are the same for all parameter vectors
			//ASSUMPTION 2: the same number of runs must be performed for each parameter vector
			if(i==0)
			{
				num_runs_0 = (Integer)getParamVal(exp_param_E.EP_PERFORM_RUNS, req_val_E.RV_CUR);
				seq_obj = getParamValues(exp_param_E.EP_PERFORM_DATA);
				for(j=0; j<seq_obj.length; j++)
				{
					seq_several_files |= checkSeq(((vector_C)seq_obj[j]).getArrayStr(), num_runs_0);
				}
			}
			else//check the assumption 2 that the same number of runs must be done for all parameter vectors
			{
				num_runs_cur = (Integer)getParamVal(exp_param_E.EP_PERFORM_RUNS, req_val_E.RV_CUR);
				if(num_runs_cur!=num_runs_0)
				{
					System.err.println("checkExpParam: number of runs is not the same over parameter vectors");
					System.exit(1);
				}
			}
			
			//all modules must be loaded if a multi-file sequence is provided
			if(seq_several_files==true)
			{
				vector_param = (vector_C)getParamVal(exp_param_E.EP_IO_LOAD_ESN, req_val_E.RV_CUR);
				for(j=0; j<vector_param.getSize(); j++)
				{
					if(vector_param.getElementAsBoolean(j)==false)
					{
						System.err.println("checkExpParam: all modules must be loaded if multi-file seq is given");
						System.exit(1);
					}
				}
			}
			
			//check topology-independent parameters
			if((Integer)getParamVal(exp_param_E.EP_PERFORM_RUN_REPEAT, req_val_E.RV_CUR) < 0)
			{
				System.err.println("checkExpParam: number of run repetitions must be positive");
				System.exit(1);
			}
			//check required percentage of required activation function
			if((Double)getParamVal(exp_param_E.EP_RES_ACTIVATION_RATIO, req_val_E.RV_CUR) < 0 ||
			   (Double)getParamVal(exp_param_E.EP_RES_ACTIVATION_RATIO, req_val_E.RV_CUR) > 1.0)
			{
				System.err.println("checkExpParam: activation ratio must be defined between 0 and 1");
				System.exit(1);
			}
			//value for seeding random generators is loaded only together with an ESN			
			num_module = ((vector_C)getParamVal(exp_param_E.EP_IO_LOAD_ESN, req_val_E.RV_CUR)).getSize();
			if(num_module!=
			   ((vector_C)getParamVal(exp_param_E.EP_IO_LOAD_SEED, req_val_E.RV_CUR)).getSize())
			{
				System.err.println("checkExpParam: different numbers of loaded ESNs and loaded seeds");
				System.exit(1);
			}
			if(num_module!=
			   ((vector_C)getParamVal(exp_param_E.EP_IO_LOAD_ESN_FILE, req_val_E.RV_CUR)).getSize())
			{
				System.err.println("checkExpParam: different numbers of requests and names of loaded ESNs");
				System.exit(1);
			}
			if(num_module!=
			   ((vector_C)getParamVal(exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_0, req_val_E.RV_CUR)).getSize())
			{
				System.err.println("checkExpParam: different numbers of requests and names of FF-ANN for output 0");
				System.exit(1);
			}
			if(num_module!=
			   ((vector_C)getParamVal(exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_1, req_val_E.RV_CUR)).getSize())
			{
				System.err.println("checkExpParam: different numbers of requests and names of FF-ANN for output 1");
				System.exit(1);
			}
			
			load_esn_param_0 = (vector_C)getParamVal(exp_param_E.EP_IO_LOAD_ESN, req_val_E.RV_CUR);
			load_esn_param_1 = (vector_C)getParamVal(exp_param_E.EP_IO_LOAD_SEED, req_val_E.RV_CUR);
			for(j=0; j<num_module; j++)
			{
				if(load_esn_param_0.getElementAsBoolean(j)==false &&
				   load_esn_param_1.getElementAsBoolean(j)==true)
				{
					System.err.println("checkExpParam: seeding value can only be loaded together with an ESN");
					System.exit(1);
				}
			}
			load_esn_param_1   = (vector_C)getParamVal(exp_param_E.EP_IO_LOAD_ESN_FILE, req_val_E.RV_CUR);
			load_ffann_param_0 = (vector_C)getParamVal(exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_0, req_val_E.RV_CUR);
			load_ffann_param_1 = (vector_C)getParamVal(exp_param_E.EP_IO_LOAD_FFANN_FILE_OUT_1, req_val_E.RV_CUR);
			for(j=0; j<num_module; j++)
			{
				if(load_esn_param_0.getElementAsBoolean(j)==true)
			    {
			    	if(load_esn_param_1.getElement(j).toString().contains("*")==true)
			    	{
			    		System.err.println("checkExpParam: name of a loaded ESN file is not specified");
			    		System.exit(1);
			    	}
			    	if(load_ffann_param_0.getElement(j).toString().contains("*")==true)
			    	{
			    		System.err.println("checkExpParam: name of a loaded FF-ANN file for output 0 is not specified");
			    		System.exit(1);
			    	}
			    	if(load_ffann_param_1.getElement(j).toString().contains("*")==true)
			    	{
			    		System.err.println("checkExpParam: name of a loaded FF-ANN file for output 1 is not specified");
			    		System.exit(1);
			    	}
			    }
			}
			//number of parameters of OFB distortion should be equal to the number of sub-reservoirs
			if((distortion_E)getParamVal(exp_param_E.EP_DISTORT_METHOD, req_val_E.RV_CUR)!=distortion_E.DISTORT_NONE)
			{
				num_out       = (Integer)getParamVal(exp_param_E.EP_OUTPUT_SIZE, req_val_E.RV_CUR);
				disturb_param = (vector_C)getParamVal(exp_param_E.EP_DISTORT_PERIOD, req_val_E.RV_CUR);
				num_disturb_param = disturb_param.getSize();
				if(num_disturb_param!=num_out)
				{
					System.err.println("checkExpParam: number of loaded disturbance periods does not equal to number of output neurons");
					System.exit(1);
				}
				disturb_param = (vector_C)getParamVal(exp_param_E.EP_DISTORT_STRENGTH, req_val_E.RV_CUR);
				num_disturb_param = disturb_param.getSize();
				if(num_disturb_param!=num_out)
				{
					System.err.println("checkExpParam: number of loaded distortion strengths does not equal to number of output neurons");
					System.exit(1);
				}
				disturb_param = (vector_C)getParamVal(exp_param_E.EP_DISTORT_DECAY, req_val_E.RV_CUR);
				num_disturb_param = disturb_param.getSize();
				if(num_disturb_param!=num_out)
				{
					System.err.println("checkExpParam: number of loaded distortion decays does not equal to number of output neurons");
					System.exit(1);
				}
			}
			
			res_size   = (vector_C)getParamVal(exp_param_E.EP_SIZE, req_val_E.RV_CUR);
			connect    = (vector_C)getParamVal(exp_param_E.EP_RES_CONNECT, req_val_E.RV_CUR);
			num_sr     = ((vector_C)getParamVal(exp_param_E.EP_RES_SPECTR_RAD, req_val_E.RV_CUR)).getSize();
			seq_start  = (vector_C)getParamVal(exp_param_E.EP_PERFORM_SEQ_START, req_val_E.RV_CUR);
			num_module    = res_size.getSize();
			num_connect   = connect.getSize();
			num_seq_start = seq_start.getSize();
			if(num_module!=num_connect)
			{
				System.err.println("checkExpParam: connectivity is expected for each ESN module");
				System.exit(1);
			}
			if(num_module!=num_sr)
			{
				System.err.println("checkExpParam: spectral radius is expected for each ESN module");
				System.exit(1);
			}
			//check sequence related parameters
			seq_param = (vector_C)getParamVal(exp_param_E.EP_PERFORM_SEQ_TARGET_NUM, req_val_E.RV_CUR);
			if(seq_param.getSize()!=num_seq_start)
			{
				System.err.println("checkExpParam: number of active modules is not equal to number of start samples");
				System.exit(1);
			}
			seq_param = (vector_C)getParamVal(exp_param_E.EP_PERFORM_SEQ_STOP, req_val_E.RV_CUR);
			if(seq_param.getSize()!=num_seq_start)
			{
				System.err.println("checkExpParam: number of stop samples is not equal to number of start samples");
				System.exit(1);
			}
			seq_param = (vector_C)getParamVal(exp_param_E.EP_PERFORM_SEQ_PURPOSE, req_val_E.RV_CUR);
			if(seq_param.getSize()!=num_seq_start)
			{
				System.err.println("checkExpParam: number of purposes is not equal to number of start samples");
				System.exit(1);
			}
			config_method = (config_method_E)getParamVal(exp_param_E.EP_CONFIG_METHOD, req_val_E.RV_CUR);
			for(j=0; j<seq_param.getSize(); j++)
			{
				//Neither training nor washout are implemented when at least one sequence contains several files.
				seq_purpose_cur = seq_param.getElementAsChar(j);
				if((seq_purpose_cur=='w' || seq_purpose_cur=='t') &&
				   seq_several_files==true)
				{
					System.err.println("checkExpParam: train and washout are not implemented for directory sequences");
					System.exit(1);
				}
				//configuration method must be chosen if configuration is required
				if(seq_purpose_cur=='c' && config_method==config_method_E.CM_None)
				{
					System.err.println("checkExpParam: configuration method is not specified");
					System.exit(1);
				}
			}
			seq_param = (vector_C)getParamVal(exp_param_E.EP_DISTORT_SEQ, req_val_E.RV_CUR);
			if(seq_param.getSize()!=num_seq_start)
			{
				System.err.println("checkExpParam: num of distort requests is not equal to number of start samples");
				System.exit(1);
			}
			num_seq_start = seq_param.getSize();
			for(j=0; j<num_seq_start-1; j++)
			{
				if((Integer)seq_param._vector.get(j)!=0 && (Integer)seq_param._vector.get(j)!=1)
				{
					System.err.println("checkExpParam: invalid request for OFB distortion");
					System.exit(1);
				}
			}
			//check indices of start samples
			num_seq_start = seq_start.getSize();
			for(j=0; j<num_seq_start-1; j++)
			{
				if((Integer)seq_start._vector.get(j) < 0)
				{
					System.err.println("checkExpParam: start sample cannot be negative");
					System.exit(1);
				}
				if((Integer)seq_start._vector.get(j) >= (Integer)seq_start._vector.get(j+1))
				{
					System.err.println("checkExpParam: previous start sample must be larger than the next one");
					System.exit(1);
				}
			}
			
			if(config_method==config_method_E.CM_DiffEvolution)
			{
				vector_param = (vector_C)getParamVal(exp_param_E.EP_CONFIG_EA_COEVOL_ORDER, req_val_E.RV_CUR);
				if(num_module != vector_param.getSize())
				{
					System.err.println("checkExpParam: invalid co-evolution order");
					System.exit(1);
				}
				vector_param = (vector_C)getParamVal(exp_param_E.EP_CONFIG_EA_MARGIN_MODULE, req_val_E.RV_CUR);
				if(num_module != vector_param.getSize())
				{
					System.err.println("checkExpParam: invalid margin of module output");
					System.exit(1);
				}
			}
			
			topology = (ReservoirInitialization)getParamVal(exp_param_E.EP_RES_TOPOLOGY, req_val_E.RV_CUR);
			switch(topology)
			{
				case RANDOM:
					for(j=0; j<num_module; j++)
					{
						subres_size = (Integer)res_size.getElement(j);
						connect_sub = (Double)connect.getElement(j);
						if(subres_size < 1)
						{
							System.err.println("checkExpParam: sub-reservoir shall not be empty");
							System.exit(1);
						}
						if(connect_sub <= 0)
						{
							System.err.println("checkExpParam: connectivity of sub-reservoir must be positive");
							System.exit(1);
						}
					}
					if((Double)getParamVal(exp_param_E.EP_SUBRES_CONNECT_OUTSIDE, req_val_E.RV_CUR)!=0)
					{
						System.err.println("checkExpParam: connectivity outside sub-reservoirs is not expected for topology SUBRES_LINK_0");
						System.exit(1);
					}
					break;
				case SRR:
					for(j=0; j<num_module; j++)
					{
						connect_sub = (Double)connect.getElement(j);
						//connectivity shall be 100%, when the reservoir contains only self-recurrent connections
						if(connect_sub != 1.0)
						{
							System.err.println("checkExpParam: connectivity shall be 100%, when the reservoir contains only self-recurrent connections");
							System.exit(1);
						}
					}
					break;
				default:
					/*no checks as the required parameters are not known*/
					break;
			}//switch
			setNextParamVector();
		}//for i
	}
	
	/**
	 * The function checks existence of all specified sequences.
	 * A sequence can consist of several files or a single file.
	 * The function issues an error message and terminates execution if at least one of the sequence files
	 * does not exist or cannot be loaded.
	 * If a certain sequence consists of several files then the function checks that a number of these files equals
	 * to a number of required runs. If they are not equal, the function issues an error message
	 * and terminates execution.
	 * 
	 * @param seq_name: array of sequence names (directories or single files)
	 * @param num_runs: number of required runs
	 * @return: TRUE if at least one sequence contains several files
	 *          FALSE if all sequences are single files
	 */
	private boolean checkSeq(String[] seq_name, int num_runs)
	{
		int i,j;
		String   seq_path;//path to a sequence (it can be a directory of a single file)
		String[] seq_file;//files or single file in a specified sequence
		File file;//file object of specified directory or file
		FileReader     file_reader;
		BufferedReader reader;
		boolean seq_several_files;//output variable: indicator that there sequences with several files
		
		//check existence of directory
		file = new File(_dir_seq);
		if(file.exists()==false)
		{
			System.err.println("checkSeq: specified directory does not exist");
			System.exit(1);
		}
		
		seq_several_files = false;
		for(i=0; i<seq_name.length; i++)
		{
			//assign path to sequence file
			seq_path = _dir_seq + File.separator + seq_name[i];
			
			file = new File(seq_path);
			
			//check a number of sequence files if a series of sequences is specified for an experiment
			if(file.isDirectory()==true)
			{
				//it is not allowed to have multi-file sequences and single-file sequences in the same param vector
				//avoid a comparison to an initialization value at sequence 0
				if(i!=0)
				{
					if(seq_several_files==false)
					{
						System.err.println("checkSeq: multi-file and single-file sequences in the same param vector");
						System.exit(1);
					}
				}
				seq_several_files = true;
				
				seq_file = file.list();
				//number of sequence files must be equal to a number of runs
				if(seq_file.length!=num_runs)
				{
					System.err.println("checkSeq: mismatch between a number of runs and a number of sequence files");
					System.exit(1);
				}
				//convert simple file names to their paths
				for(j=0; j<seq_file.length; j++)
				{
					seq_file[j] = _dir_seq + File.separator + seq_name[i] + File.separator + seq_file[j]; 
				}
			}
			else//file, not a directory
			{
				//it is not allowed to have multi-file sequences and single-file sequences in the same param vector
				//avoid a comparison to an initialization value at sequence 0
				if(i!=0)
				{
					if(seq_several_files==true)
					{
						System.err.println("checkSeq: multi-file and single-file sequences in the same param vector");
						System.exit(1);
					}
				}
				seq_several_files = false;
				
				seq_file = new String[1];
				seq_file[0] = _dir_seq + File.separator + seq_name[i];
			}
			
			for(j=0; j<seq_file.length; j++)
			{
				try{
					file = new File(seq_file[j]);
					file_reader = new FileReader(file);
					reader      = new BufferedReader(file_reader);

					while (reader.ready())
					{
						//try to read a line from the file
						reader.readLine();
					}

				} catch (FileNotFoundException fnf) {
					System.err.println("checkSeq: sequence file does not exist");
					System.exit(1);
				} catch (IOException io) {
					System.err.println("checkSeq: cannot read from a sequence file");
					System.exit(1);
				}
			}
		}
		return seq_several_files;
	}
	
	/**
	 * compute total number of different combinations of the experiment parameter values
	 * @return, total number of combinations of parameter values
	 */
	private int computeNumParamVectors()
	{
		int i;
		int num_load;//number of loaded values of current parameter
		int num_vect;
		
		num_vect = 1;
		for(i=0; i<_param.length; i++)
		{
			//are there any provided parameter values?
			if(_param[i].value!=null)
			{
				num_load = _param[i].value.length;
				num_vect *= num_load;
			}
		}
		return num_vect;
	}
	
	/**
	 * The function converts a given string to a value of the specified class.
	 * @param str: given string
	 * @param c: specified class of the value
	 * @return: obtained value
	 */
	private Object convertStringToValue(String str, Class<?> c)
	{
		Object val;
		
		if(c==Integer.class)
		{
			val = Integer.valueOf(str);
		}
		else if(c==Double.class)
		{
			val = Double.valueOf(str);
		}
		else if(c==Boolean.class)
		{
			val = Boolean.valueOf(str);
		}
		else if(c==multi_val_C.class)
		{
			if(str.contains("(")==true)//'(' and ')' are attributes of an interval
			{
				interval_C tmp_interval;
				tmp_interval = new interval_C(str);
				val = new multi_val_C(tmp_interval);
			}
			else if(str.contains("{")==true)//'{' and '}' are attributes of a vector
			{
				val = new vector_C(str);
			}
			else
			{
				val = Double.NaN;
				System.err.println("convertStringToValue: unknown type of multi-value");
				System.exit(1);
			}
		}
		else if(c==vector_C.class)
		{
			val = new vector_C(str);
		}
		else if(c==activation_E.class)
		{
			val = activation_E.fromString(str);
		}
		else if(c==String.class)
		{
			val = str;
		}
		else if(c==noise_E.class)
		{
			val = MathNoise.noise_E.fromString(str);
		}
		else if(c==config_method_E.class)
		{
			val = config_method_E.fromString(str);
		}
		else if(c==config_ea_mode_E.class)
		{
			val = config_ea_mode_E.fromString(str);
		}
		else if(c==config_ea_init_E.class)
		{
			val = config_ea_init_E.fromString(str);
		}
		else if(c==config_ea_order_method_E.class)
		{
			val = config_ea_order_method_E.fromString(str);
		}
		else if(c==ReservoirInitialization.class)
		{
			val = ReservoirInitialization.fromString(str);
		}
		else if(c==distortion_E.class)
		{
			val = distortion_E.fromString(str);
		}
		else if(c==leakage_assign_E.class)
		{
			val = leakage_assign_E.fromString(str);
		}
		else
		{
			val = null;
			System.err.println("ExpParam.convertStringToValue: unknown class of default values");
			System.exit(1);
		}
		
		return val;
	}
	
	/**
	 * The function extracts parameter values which were specified as a range. The values are extracted
	 * in the specified format.
	 * @param str: given string
	 * @param c: specified class of the value
	 * @return: array of extracted values
	 */
	private Vector<Object> expandRangeVal(String str, Class<?> c)
	{
		Vector<Double>   val_all_double;//array of doubles which is used when a range of doubles is specified
		Vector<Integer>  val_all_int;//array of integers which is used when a range of integers is specified
		Vector<vector_C> val_all_vect;//array of vectors which is used when a range of vectors is specified
		Vector<Object> val_all;//output array
		
		val_all = new Vector<Object>(0, 1);
		if(c==Integer.class)
		{
			val_all_int = expandRangeInt(str);
			val_all.addAll(val_all_int);
		}
		else if(c==Double.class)
		{
			val_all_double = expandRangeDouble(str);
			val_all.addAll(val_all_double);
		}
		else if(c==multi_val_C.class)
		{
			if(str.contains("(")==true)//'(' and ')' are attributes of an interval
			{
				System.err.println("expandRangeVal: no range of values is expected for the interval class");
				System.exit(1);
			}
			else if(str.contains("{")==true)//'{' and '}' are attributes of a vector
			{
				val_all_vect = expandRangeVector(str);
				val_all.addAll(val_all_vect);
			}
			else
			{
				val_all = null;
				System.err.println("expandRangeVal: unknown type of multi-value");
				System.exit(1);
			}
		}
		else if(c==vector_C.class)
		{
			val_all_vect = expandRangeVector(str);
			val_all.addAll(val_all_vect);
		}
		else if(c==Boolean.class                 ||
				c==activation_E.class              ||
				c==String.class                  ||
				c==noise_E.class                 ||
				c==ReservoirInitialization.class ||
				c==config_method_E.class         ||
				c==config_ea_mode_E.class        ||
				c==config_ea_init_E.class        ||
				c==distortion_E.class)
		{
			System.err.println("ExpParam.expandRangeVal: no range of values is expected for the specified class");
			System.exit(1);
		}
		else
		{
			val_all = null;
			System.err.println("ExpParam.expandRangeVal: unknown class of default values");
			System.exit(1);
		}
		
		return val_all;
	}
	
	/**
	 * The function extracts all double values from the range which is defined in the provided string.
	 * @param str_double: provided string
	 * @return: array of extracted double values
	 */
	private Vector<Double> expandRangeDouble(String str_double)
	{
		int i;
		int num_steps;//number of required steps in the specified range
		double init_val;//initial value of the range
		double step;//step size
		double last_val;//last value of the range
		double range_width;//width of the range
		double val_cur;//currently computed value from the range
		double tolerance;//max deviation between the range and its coverage with the specified steps
		double deviation;//deviation between the range and its coverage with the specified steps
		String[] val_extracted;//strings with values of the initial and last values of the range and the step
		Vector<Double> val_all;//output array
		
		//check the input
		val_extracted = str_double.split(":");
		if(val_extracted.length!=3)
		{
			System.err.println("expandRangeDouble: incorrect syntax for range definition");
			System.exit(1);
		}
		//double values should not contain decimal points
		if(val_extracted[0].indexOf(".")==-1 || val_extracted[1].indexOf(".")==-1 || val_extracted[2].indexOf(".")==-1)
		{
			System.err.println("expandRangeDouble: floating point is expected in all values of the range");
			System.exit(1);
		}
		init_val = Double.valueOf(val_extracted[0]);
		step     = Double.valueOf(val_extracted[1]);
		last_val = Double.valueOf(val_extracted[2]);		
		
		//check the consistency of the specified range
		if(init_val >= last_val)
		{
			System.err.println("expandRangeDouble: last value is smaller than the initial value");
			System.exit(1);
		}
		range_width = last_val - init_val;
		tolerance = 0.0001*step;//set the tolerance to 0.01% of the required step length
		num_steps = (int)(Math.round(range_width / step));
		deviation = range_width - (step*num_steps);
		if(deviation > tolerance)
		{
			System.err.println("expandRangeDouble: step does not match the initial and last values");
			System.exit(1);
		}
		
		//assign the values for the output
		val_all = new Vector<Double>(0, 1);
		for(i=0; i<=num_steps; i++)
		{
			val_cur = init_val + (i*step);			
			val_all.add(val_cur);
		}
		return val_all;
	}
	
	/**
	 * The function extracts all integer values from the range which is defined in the provided string.
	 * @param str_int: provided string
	 * @return: array of extracted integer values
	 */
	private Vector<Integer> expandRangeInt(String str_int)
	{
		int i;
		int init_val;//initial value of the range
		int step;//step size
		int last_val;//last value of the range
		int range_width;//width of the provided range
		String[] val_extracted;//strings with values of the initial and last values of the range and the step
		Vector<Integer> val_all;//output array
		
		//integer values should not contain decimal points
		if(str_int.indexOf(".")!=-1)
		{
			System.err.println("expandRangeInt: floating point is not allowed in the range of integers");
			System.exit(1);
		}
		val_extracted = str_int.split(":");
		if(val_extracted.length!=3)
		{
			System.err.println("expandRangeInt: incorrect syntax for range definition");
			System.exit(1);
		}
		init_val = Integer.valueOf(val_extracted[0]);
		step     = Integer.valueOf(val_extracted[1]);
		last_val = Integer.valueOf(val_extracted[2]);
		
		//check the consistency of the specified range
		if(init_val >= last_val)
		{
			System.err.println("expandRangeInt: last value is smaller than the initial value");
			System.exit(1);
		}
		range_width = last_val - init_val;
		if(((range_width/step)*step)!=range_width)
		{
			System.err.println("expandRangeInt: step does not match the initial and last values");
			System.exit(1);
		}
		
		//assign the values for the output
		val_all = new Vector<Integer>(0, 1);
		for(i=init_val; i<=last_val; i+=step)
		{
			val_all.add(i);
		}
		return val_all;
	}
	
	/**
	 * The function extracts all vectors from the range which is defined in the provided string.
	 * @param str_vector: provided string
	 * @return: array of extracted integer values
	 */
	private Vector<vector_C> expandRangeVector(String str_vector)
	{
		int i;
		int cnt_field;//counter of the processed fields of the string
		int[] cnt_val_field;//counter of values on the same field
		boolean is_int;//indicator that the values of the vector's elements are integers, not doubles
		boolean start_last;//indicator that the fetching of the values of the last field should be started again
		String singleRangeStr;//string which encodes a range of a single element of the vector
		String str_one_vector;//string of one vector
		StringTokenizer st;//array of fields of the provided string		
		Object[][] val_all_extracted;//extracted values from all fields of the string
		vector_C vector_cur;//currently created vector
		Vector<Double>  val_range_double;//values from one field, if vector has double values
		Vector<Integer> val_range_int;//values from one field, if vector has integer values
		Vector<vector_C>  val_all;//output array
		
		//allocate the output array
		val_all = new Vector<vector_C>(0, 1);
		
		//extract the fields where separate vector elements are defined
		st = new StringTokenizer(str_vector, "{,}");
		
		//check whether the values are integers or doubles
		if(str_vector.contains(".")==true)//it should be double, if the string contains "."
		{
			is_int = false;
		}
		else
		{
			is_int = true;
		}
		val_all_extracted = new Object[st.countTokens()][];
		
		//extract values from each field
		cnt_field = 0;
		cnt_val_field = new int[st.countTokens()];
		while(st.hasMoreTokens())
		{
			singleRangeStr = st.nextToken();

			if(is_int==true)
			{
				if(singleRangeStr.contains(".")==true)//it should be double, if the string contains "."
				{
					System.err.println("expandRangeVector: mixing of doubles and integers is not accepted in vectors");
					System.exit(1);
				}
				val_range_int = expandRangeInt(singleRangeStr);
				val_all_extracted[cnt_field] = new Object[val_range_int.size()];				
				for(i=0; i<val_range_int.size(); i++)
				{
					val_all_extracted[cnt_field][i] = val_range_int.get(i);
				}
			}
			else
			{
				if(singleRangeStr.contains(".")==false)//it should be double, if the string contains "."
				{
					System.err.println("expandRangeVector: mixing of doubles and integers is not accepted in vectors");
					System.exit(1);
				}
				val_range_double = expandRangeDouble(singleRangeStr);
				val_all_extracted[cnt_field] = new Object[val_range_double.size()];
				for(i=0; i<val_range_double.size(); i++)
				{
					val_all_extracted[cnt_field][i] = val_range_double.get(i);
				}
			}
			cnt_val_field[cnt_field] = 0;//initialize the counter to store the values for the output below
			cnt_field++;
		}//store values
		
		//assign the values for the output
		val_all = new Vector<vector_C>(0, 1);
		cnt_field = val_all_extracted.length-1;//start with the last field of the string
		do
		{
			//create one vector
			str_one_vector = "{";
			for(i=0; i<val_all_extracted.length; i++)
			{
				str_one_vector+=val_all_extracted[i][cnt_val_field[i]];
				if(i!=val_all_extracted.length-1)
				{
					str_one_vector+=",";
				}
			}
			str_one_vector+= "}";
			vector_cur = new vector_C(str_one_vector);
			val_all.add(vector_cur);
			
			//update the indices to consider the next combination of values
			cnt_val_field[cnt_field]++;
			start_last = false;
			while(cnt_val_field[cnt_field]==val_all_extracted[cnt_field].length && cnt_field!=0)
			{
				//set counter of values the current field to 0 and go one index back
				cnt_val_field[cnt_field] = 0;
				//reduce the index of the field, if it is not the 1st one
				if(cnt_field!=0)
				{
					cnt_field--;
					//increment its counter
					cnt_val_field[cnt_field]++;
					//fetching of the values of the last field is started after updating an index of not-last element
					start_last = true;
				}
			}
			//fetching of the values of the last field is started after updating an index of not-last element
			if(start_last==true)
			{
				cnt_field = val_all_extracted.length-1;
			}
		}while(cnt_val_field[0] < val_all_extracted[0].length);//until all values of 1st vector element are considered
		return val_all;
	}
	
	/**
	 * The function extracts values of parameters for the experiment.
	 * @param filedata: data loaded from file with parameters for an experiment
	 * @return none
	 */
	private void loadParamVal(Vector<String> filedata)
	{
		int i, j; 
		String paramStrBegin;//string which begins a file's section corresponding to experiment parameter
		String paramStrEnd;//string which ends a file's section corresponding to experiment parameter
		int idxStrBegin;//index of the string beginning 
		int idxStrEnd;//index of the string end
		String paramStrCom;//variable which keeps a common part of the beginning and ending strings
		String paramValStr;//string of experiment parameter values
		String singleParamVal;//string of next parameter value
		int str_size;//size of file's content corresponding to parameter values
		boolean is_loaded;//are there some values loaded?
		Class<?> param_class;//class of values of the loaded parameter
		int param_idx;//index of loaded parameter
		Object single_param_val;//extracted parameter value
		Vector<Object> expand_param_val;//extracted array of parameters obtained with expanding the range
		Vector<Object> loaded_values;//all loaded values of one parameter
		
		is_loaded = false;
		//go over experiment parameter
		for(i=0; i<_param.length; i++)
		{
			paramStrCom = _param[i].name;
			paramStrBegin = "<begin " + paramStrCom;
			paramStrBegin+= ">";
			idxStrBegin = filedata.indexOf(paramStrBegin);
			paramStrEnd   = "<end " + paramStrCom;			
			paramStrEnd  += ">";
			idxStrEnd   = filedata.indexOf(paramStrEnd);
			
			if(idxStrBegin!=-1)//check that the parameter is provided
			{
				if(idxStrEnd!=-1)//check that the file sysntax is OK
				{
					str_size = idxStrEnd - idxStrBegin;
					if(str_size>1)//check that there are values provided
					{
						//set class of expected values
						param_idx   = getParamIdxByName(paramStrCom);
						param_class = _param[param_idx].value_def.getClass();
						//parameter values are stored in the next line after the beginning of the section
						paramValStr = filedata.get(filedata.indexOf(paramStrBegin) + 1);
						if(paramValStr.isEmpty()==false)
						{
							//construct tokenizer with a set of default delimiters (space is a default delimiter)
							StringTokenizer st = new StringTokenizer(paramValStr);
							_param[i].idx_value_cur = 0;
							loaded_values = new Vector<Object>(0, 1);
							while(st.hasMoreTokens())
							{
								is_loaded = true;
								singleParamVal = st.nextToken();
								//extract a single parameter value or several values, if their range is provided
								if(singleParamVal.indexOf(":")==-1)//single value
								{
									single_param_val = convertStringToValue(singleParamVal, param_class);
									//store the extracted value in array of all values of the experimental parameter
									loaded_values.add(single_param_val);
									_param[i].idx_value_cur++;
								}
								else//range of values must be expanded
								{
									expand_param_val = expandRangeVal(singleParamVal, param_class);
									//store the expansion values in array of all values of the experimental parameter
									loaded_values.addAll(expand_param_val);
									_param[i].idx_value_cur += expand_param_val.size();
								}
							}//loaded values
							
							//store the loaded values in a global array
							_param[i].value = new Object[_param[i].idx_value_cur];
							for(j=0; j<_param[i].idx_value_cur; j++)
							{
								_param[i].value[j] = loaded_values.get(j);
							}
						}
					}//if "str_size"
				}
				else
				{
					System.err.println("loadParamVal: field start without field end in the parameter file");
					System.exit(1);
				}
			}
		}//for "i"
		//check the loaded data
		if(is_loaded==false)
		{
			System.err.println("loadParamVal: no values are provided for experimental parameters");
			System.exit(1);
		}
	}
	
	/**
	 * load parameters for an experiment from a file
	 * @param paramfile, path to the file with the parameters for an experiment
	 * @return Vector<String>, data from the file as an array of strings
	 */
	public Vector<String> fromFile(File paramfile)
	{
		Vector<String> data = new Vector<String>();
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(paramfile));
			
			while (reader.ready()) {
				String line = reader.readLine();
				
				//filter out lines with comments
				if (!line.startsWith("#"))
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
		
		return data;
	}
	
	/**
	 * return a list of all parameter names
	 * @return list of all parameter names
	 */
	public Vector<String> getAllParamNames()
	{
		Vector<String> out_list;//output list with parameter names
		
		out_list = new Vector<String>(0, 1);
		//go over all experiment parameters
		for(exp_param_E e : exp_param_E.values())
		{
			out_list.add(_param[e.ordinal()].name);
		}
		return out_list;
	}
	
	/**
	 * The function returns connectivity of a specified module. The module is specified by its index.
	 * 
	 * @return: connectivity of specified module
	 */
    public double getConnectivity(int idx_module)
    {
    	double   connect_module;//output variable
    	vector_C connect;//vector of connectivity of all modules

    	connect = (vector_C)getParamVal(exp_param_E.EP_RES_CONNECT, req_val_E.RV_CUR);
    	connect_module = (Double)connect._vector.get(idx_module);
    	
    	return connect_module;
    }
    
    /**
	 * The function returns a leakage rate of a specified module. The module is specified by its index.
	 * 
	 * @return: leakage rate of the specified module
	 */
    public double getLeakageRate(int idx_module)
    {
    	double   leakage_rate_module;//output variable
    	vector_C leakage_rate;//vector of leakage rates of all modules

    	leakage_rate = (vector_C)getParamVal(exp_param_E.EP_LEAKAGE_RATE, req_val_E.RV_CUR);
    	leakage_rate_module = (Double)leakage_rate._vector.get(idx_module);
    	
    	return leakage_rate_module;
    }
    
    /**
	 * The function returns a reservoir size of a specified module. The module is specified by its index.
	 * 
	 * @return: reservoir size of specified module
	 */
    public int getSize(int idx_module)
    {
    	int      size_module;//output variable
    	vector_C size;//vector of reservoir sizes of all modules

    	size = (vector_C)getParamVal(exp_param_E.EP_SIZE, req_val_E.RV_CUR);
    	size_module = (Integer)size._vector.get(idx_module);
    	
    	return size_module;
    }
	
	/**
	 * return a total number of parameter values for the experiments
	 * @return number of parameter vectors
	 */
	public int getNumParamVectors()
	{
		return _num_param_vectors;
	}
	
	/**
	 * return index of experimental parameter by its name
	 * @return index of experiment parameter
	 */
	public int getParamIdxByName(String name)
	{
		int i;
		int param_idx;
		
		//go over all experiment parameters
		param_idx = _param.length;
		for(i=0; i<_param.length; i++)
		{
			if(_param[i].name.compareTo(name)==0)
			{
				if(param_idx == _param.length)
				{
					param_idx = i;
				}
				else
				{
					System.err.println("getParamIdxByName: muliple experiment parameter with the same name");
					System.exit(1);
				}
			}
		}
		if(param_idx==_param.length)
		{
			System.err.println("getParamIdxByName: no experiment parameter with the specified name");
			System.exit(1);
		}
		return param_idx;
	}
	
	/**
	 * return a value of a specified parameter  
	 * @param exp_param, specified parameter
	 * @param req_val, type of required value (default or current)
	 * @return value of experiment parameter
	 */
	public Object getParamVal(exp_param_E exp_param, req_val_E req_val)
	{
		Object param_val;//parameter value for output
		
		switch(req_val)
		{
			case RV_DEF:
				param_val = _param[exp_param.ordinal()].value_def;
			break;
			case RV_CUR:
				param_val = _param[exp_param.ordinal()].value_cur;
			break;
			default:
				System.err.println("getParamVal: unknown type of required value");
				System.exit(1);
				param_val = null;
			break;
		}
		
		return param_val;
	}
	
	/**
	 * return a value of a specified parameter  
	 * @param param_name, name of specified parameter
	 * @param req_val, type of required value (default or current)
	 * @return value of experiment parameter
	 */
	public Object getParamVal(String param_name, req_val_E req_val)
	{
		Object param_val;//parameter value for output
		
		switch(req_val)
		{
			case RV_DEF:
				param_val = _param[getParamIdxByName(param_name)].value_def;
			break;
			case RV_CUR:
				param_val = _param[getParamIdxByName(param_name)].value_cur;
			break;
			default:
				System.err.println("getParamVal: unknown type of required value");
				System.exit(1);
				param_val = null;
			break;
		}
		
		return param_val;
	}
	
	/**
	 * return a value of experimental parameter by its name; the required parameter value is specified by its index 
	 * @param name, name of the parameter
	 * @param val_idx, index of value
	 * @return value of experiment parameter
	 */
	public Object getParamValByName(String name, int val_idx)
	{
		int param_idx;//parameter index
		
		param_idx = getParamIdxByName(name);
		
		return _param[param_idx].value[val_idx];
	}
	
	/**
	 * return all values of the specified parameter available for the experiment. If no parameter values were loaded
	 * then the function returns only the default value.
	 * @param exp_param, specified parameter
	 * @return, array of available parameter values
	 */
	public Object[] getParamValues(exp_param_E exp_param)
	{
		Object[] values;
		
		if(_param[exp_param.ordinal()].value==null)
		{
			//allocate arrays only for default value and assign the default value to the first element of the array 
			values    = new Object[1];
			values[0] = _param[exp_param.ordinal()].value_def; 
		}
		else
		{
			values = _param[exp_param.ordinal()].value;
		}
		return values;
	}
	
	/**
	 * return all values of the specified parameter available for the experiment. If no parameter values were loaded
	 * then the function returns only the default value.
	 * @param param_name, specified parameter
	 * @return, array of available parameter values
	 */
	public Object[] getParamValues(String param_name)
	{
		Object[] values;
		int param_idx;//index of parameter specified by its name
		
		param_idx = getParamIdxByName(param_name);
		
		if(_param[param_idx].value==null)
		{
			//allocate arrays only for default value and assign the default value to the first element of the array 
			values    = new Object[1];
			values[0] = _param[param_idx].value_def; 
		}
		else
		{
			values = _param[param_idx].value;
		}
		return values;
	}
	
	/**
	 * The function searches for a sequence interval where a sequence is used for the specified purpose.
	 * The function returns an index of the last interval, if there are several intervals for the specified purpose.
	 *
	 * @param seq_purpose: specified sequence purpose
	 * @return: index of an interval with a testing sequence
	 */
	public int getIdxInterv(Character seq_purpose)
	{
		int i;
		int idx_interv;//output variable
		Character seq_purpose_cur;//currently checked sequence purpose
		vector_C seq_purpose_all;//array with sequence purposes on all intervals
		
		//initialize the output
		idx_interv = -1;
		
		seq_purpose_all = (vector_C)getParamVal(exp_param_E.EP_PERFORM_SEQ_PURPOSE, req_val_E.RV_CUR);
		for(i=0; i<seq_purpose_all.getSize(); i++)
		{
			seq_purpose_cur = seq_purpose_all.getElementAsChar(i);
			if(seq_purpose_cur == seq_purpose)
			{
				idx_interv = i;
			}
		}
		
		return idx_interv;
	}
	
	/**
	 * The function returns a path to a directory with sequence files.
	 * @return: path to a directory with sequence files
	 */
	public String getPathSeq()
	{
		return _dir_seq;
	}
	
	/**
	 * The function returns a path to a directory with ESN files
	 * @return: path to a directory with ESN files
	 */
	public String getPathEsn()
	{
		return _dir_esn;
	}
	
	/**
	 * The function indicates a path to a loaded file with parameters of experiment.
	 * 
	 * @return: path to a loaded file with parameters of experiment
	 */
	public String getNameParamFile()
	{
		return _name_param_file;
	}
	
	/**
	 * return names of parameters where values were provided for experiment
	 * @return list of names of parameters with provided values
	 */
	public Vector<String> getProvidedParamNames()
	{
		Vector<String> param_names = new Vector<String>(0, 1);
		
		for (exp_param_E e : exp_param_E.values())
		{
			if(isParamProvided(e))
			{
				param_names.add(_param[e.ordinal()].name);
			}
		}
		return param_names;
	}
	
	/**
	 * The function returns a spectral radius of the specified module. The module is specified by its index.
	 *
	 * @param module_idx: module index
	 * @return: spectral radius of a specifed module
	 */
	public double getSpectralRadius(int module_idx)
    {
		double sr;//output variable
		vector_C sr_vect;//spectral radii in form of a vector
		
		sr_vect = (vector_C)getParamVal(exp_param_E.EP_RES_SPECTR_RAD, req_val_E.RV_CUR);
		sr = (Double)sr_vect._vector.get(module_idx);
    	
		return sr;
    }
	
	/**
	 * return the content of a possible user comment for the experiment
	 * @return value of experiment parameter
	 */
	public String getUsrComment()
	{
		return _exp_user_comment;
	}
	
	/**
	 * check whether the specified parameter name is valid
	 * @param name, specified parameter name
	 * @return
	 */
	public boolean isParam(String name)
	{
		int i;
		boolean is_valid; 
		
		is_valid = false;
		for(i=0; i<_param.length; i++)
		{
			if(_param[i].name.compareTo(name)==0)
			{
				is_valid = true;
			}
		}
		return is_valid;
	}
	
	/**
	 * indicate whether values were loaded for the specified experiment parameter 
	 * @param name, specified experiment parameter
	 * @return, "true", if values were provided; "false", no values were provided 
	 */
	public boolean isParamProvided(exp_param_E name)
	{
		boolean is_provided;
		
		if(_param[name.ordinal()].value!=null)
		{
			is_provided = true;
		}
		else
		{
			is_provided = false;
		}
		return is_provided;
	}

	/**
	 * set first combination of the experiment parameters
	 */
	public void setFirstParamVector()
	{
		int i;
		
		//go over all possible experiment parameters
		for(i=0; i<_param.length; i++)
		{
			//are there loaded parameter values?
			if(_param[i].value!=null)
			{
				_param[i].idx_value_cur = 0;
				_param[i].value_cur = _param[i].value[ _param[i].idx_value_cur ];
			}
			else
			{
				_param[i].value_cur = _param[i].value_def;
			}
		}
	}
	
	/**
	 * set next combination of the experiment parameters
	 * @return, "true", if a new combination of parameter values was set;
	 *          "false", if no other parameter values are available anymore
	 */
	public boolean setNextParamVector()
	{
		int i;
		boolean is_new;
		
		//go over all possible experiment parameters
		is_new = false;
		for(i=0; i<_param.length && is_new==false; i++)
		{
			//are there loaded parameter values?
			if(_param[i].value!=null)
			{
				if(_param[i].idx_value_cur < (_param[i].value.length-1))
				{
					_param[i].idx_value_cur++;
					is_new = true;
				}
				else
				{
					_param[i].idx_value_cur = 0;
				}
				_param[i].value_cur = _param[i].value[ _param[i].idx_value_cur ];
			}
			else
			{
				_param[i].value_cur = _param[i].value_def;
			}
		}
		return is_new; 
	}
}
