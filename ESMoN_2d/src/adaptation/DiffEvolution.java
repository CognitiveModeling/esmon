package adaptation;

import java.util.Random;

import adaptation.DiffEvolutionParam.config_ea_init_E;
import adaptation.DiffEvolutionTypes.error_trend_E;
import adaptation.DiffEvolutionTypes.param_rand_fetch_E;

import esn.Module.layer_type_E;
import esn.mESN;

import types.interval_C;

/**
 * This class implements an differential evolution.
 * @author Danil Koryakin
 *
 */
public class DiffEvolution
{
	public class output_history_C
	{
		private int _idx;//index in the history array where the next value can be stored
		private double[][] _output;//array of outputs for all time steps in a fitness window
                                   //1D: time step, 2D: outputs of each output neuron
		
		/**
		 * Constructor of the class to store a history of outputs.
		 * @param fit_len: width of a fitness window
		 * @param num_out: number of output neurons
		 */
		public output_history_C(int fit_len, int num_out)
		{
			_output = new double[fit_len][num_out];
			_idx = 0;
		}
		
		/**
		 * The functions assigns 0 to all elements of the history array.
		 * It also sets a counter of elements to 0;
		 */
		public void resetHistory()
		{
			int i, j;
			
			//count the 2nd index in the outer loop for more effectiveness in order to avoid unnecessary checks
			//   of the stop condition; the 2nd dimension is usually "1"
			for(j=0; j<_output[0].length; j++)
			{
				for(i=0; i<_output.length; i++)
				{
					_output[i][j] = 0;
				}
			}
			_idx = 0;
		}
		
		/**
		 * The function sets a counter of elements in the history array to "0".
		 */
		public void resetHistoryIdx()
		{
			_idx = 0;
		}

		/**
		 * The function stores a provided vector of module output values at the next free element.
		 * After the storing, the function increments an index of the next free element.
		 * 
		 * @param output: vector to be stored
		 */
		public void storeNextElement(double[] output)
		{
			int i;
			
			for(i=0; i<output.length; i++)
			{
				_output[_idx][i] = output[i];
			}
			_idx++;
			if(_idx == _output.length)
			{
				_idx = 0;
			}
		}
		
		/**
		 * The function retrieves an output of a specified individual at a required time step of a fitness window.
		 * The time step is an offset with respect to the 1st time step of a fitness window.
		 * The function stores a retrieved output in a provided array.
		 * 
		 * @param time_offset: offset of a time step within a fitness window
		 * @param storage: array where an output must be stored
		 */
		public void getOutputByIndex(int time_offset, double[] storage)
		{
			int i;
			int idx_time_step;//index of an element for a required time step
			int time_idx_sum;//index of element without considering a possible index overflow
			int num_out;//number of output neurons
			int fitlen;//width of a fitness window
			
			fitlen = _output.length;
			time_idx_sum = _idx + time_offset;
			if(time_idx_sum >= fitlen)
			{
				idx_time_step = time_idx_sum - fitlen;
			}
			else
			{
				idx_time_step = time_idx_sum;
			}
			//store values for the output
			num_out = _output[idx_time_step].length;
			for(i=0; i<num_out; i++)
			{
				storage[i] = _output[idx_time_step][i];
			}
		}
		
		/**
		 * The function indicates a history array.
		 * 
		 * @return: identifier of an history array
		 */
		public double[][] getArray()
		{
			return _output;
		}
		
		/**
		 * The function returns the output history as an ordered array where the first element is the oldest one.
		 * 
		 * @return ordered array of the module's outputs
		 */
		public double[][] getOrderedArray()
		{
			int i;
			int idx_source;
			double[][] ordered_output;//output variable
			
			ordered_output = new double[_output.length][];
			
			idx_source = _idx;
			for(i=0; i<ordered_output.length; i++)
			{
				ordered_output[i] = _output[idx_source].clone();
				
				idx_source++;
				if(idx_source==_output.length)
				{
					idx_source = 0;
				}
			}
			
			return ordered_output;
		}
	};
	
	private boolean _is_initialized;//indicator whether the population has been initialized
	private boolean _is_inactive;//indicator that a module of this population is inactive
	private int _sub_idx;//index of considered sub-reservoir
	private int _best_idx;//index of the best individual in the population
	private int _fitness_length;//length of a fitness window
	private int _num_out;//number of output neurons
	private int _num_in;//number of input neurons
	private int _len_individual;//length of an individual
	private int _sort_min_num;//minimum number of individuals to be sorted 
	private double[][] _sub_output;//output of considered module for all output neurons
	private double[][] _violation_sub_out_max;//largest violation of a valid range by a module output of each individual
	private double[][] _violation_sub_out_sum;//sum of all violations of valid range by module output over whole life
	public output_history_C[] _sub_output_history;//history of outputs of individuals for the whole fitness window
	private DiffEvolutionError[] _error;//array with error status of all individuals
	private boolean[] _do_sort;//array with indicators whether an individual is allowed for sorting
	private error_trend_E[] _error_trend;//trend of a prediction error for all individuals
	private double[][] _population;//population: 1D: different individuals, 2D: values of the corresponding individual
	private double[][] _end_individual;//set of module states corresponding to individuals at the end of fitness window
	private double _F;//differential weights
	private double _CR;//probability of recombination
	private double _suppress_probab;
	private double[] _lim_opt_max;//max absolute value of limits of the optimized vector
	private double[] _max_sub_contribution;//max absolute values of a contribution of the module on each output element
	private interval_C[][] _lim_opt;//valid ranges of the optimized vector
	private interval_C[][] _lim_sub_output;//valid ranges of output modes of a considered module
	private interval_C[] _lim_esn_output;//lower and upper limits of valid ranges of output neurons of the whole mESN
	private interval_C[] _extreme_esn_output;//extreme values for each dimension of the target dynamics
	private Random _rand;//generator of random numbers
	private mESN _esn;//optimized ESN. It is needed to get an access to a function for computing an error.
	
	//temporary arrays that are defined globally not to allocate them in called functions separately
	private double[] _tmp_responsibility;
	
	private final int _min_population_size = 4;//smallest size of the population
	private final double _min_CR = 0;//smallest acceptable probability of recombination
	private final double _max_CR = 1;//largest acceptable probability of recombination
	private final double _min_F = 0;//smallest value of the differential weight
	private final double _max_F = 2;//largest value of the differential weight
	private final double _suppress_probab_init = 0.2;//0.01;//initial value of probability of suppression of responsibilities
	
	/**
	 * A constructor which sets the same lower and upper limits for all elements of the optimized vector.
	 * 
	 * @param esn: modular ESN
	 * @param F: required value of the differential weight
	 * @param CR: required value of the probability of recombination
	 * @param population_size: required population size
	 * @param sort_min_num: minimum number of individuals to be sorted 
	 * @param lim: valid range for each element of the optimized vector
	 * @param lim_sub_output: valid range for each output neuron of the considered module
	 * @param range_output_bias: valid range for each element of the module's output bias
	 * @param range_responsibility: valid ranges of responsibilities of module for all output nodes
	 * @param range_input: valid ranges of all elements of the input vector
	 * @param sub_idx: index of considered module
	 * @param fitlen: size of a fitness window
	 * @param seed: value for seeding random generators
	 */
	public DiffEvolution(mESN esn, double F, double CR, int population_size, int sort_min_num,
			             interval_C[][] lim, interval_C[][] lim_sub_output, interval_C[] range_output_bias,
			             interval_C[] range_responsibility, interval_C[][] range_input, int sub_idx, int fitlen,
			             int seed)
	{
		int i, k;
		int idx_gen;//index of the current genotype's element
		int num_res;//number of reservoir neurons
		int num_out;//number of output neurons
		int num_in;//number of input neurons
		int idx_last_interval;//index of the last valid interval of a node
		double lower_lim, upper_lim;//left and right borders of an interval parameter
		double max_val;//found largest value

		_esn = esn;
		_rand = new Random(seed);
		_sub_idx = sub_idx;
		_is_initialized = false;
		_fitness_length = fitlen;
		_sort_min_num = sort_min_num;
		
		num_res = _esn.getNumNodesModule(_sub_idx, layer_type_E.LT_RES);
		num_out = _esn.getNumNodesModule(_sub_idx, layer_type_E.LT_OUTPUT);
		num_in = _esn.getNumNodesModule(_sub_idx, layer_type_E.LT_INPUT);
		//individual = output bias + responsibilities + reservoir states + input vector
		_len_individual = num_out + num_out + num_res + num_in;
		_num_out        = num_out;
		_num_in         = num_in;
		_is_inactive    = false;
		
		//check a required population size
		if(population_size < _min_population_size)
		{
			System.err.println("DiffEvolution: invalid size of the population");
			System.exit(1);
		}
		
		_extreme_esn_output = new interval_C[_num_out];
		_max_sub_contribution = new double[_num_out];
		for(i=0; i<_num_out; i++)
		{
			_extreme_esn_output[i] = new interval_C(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
		}
		
		//allocate temporary variables
		_tmp_responsibility = new double[num_out];
		
		//allocate the population and its fitness
		_population = new double[population_size][];
		_end_individual = new double[population_size][];
		_error = new DiffEvolutionError[population_size];
		_error_trend = new error_trend_E[population_size];
		_sub_output = new double[population_size][];
		_violation_sub_out_max = new double[population_size][];
		_violation_sub_out_sum = new double[population_size][];
		_sub_output_history = new output_history_C[population_size];
		_do_sort = new boolean[population_size];
		for(i=0; i<population_size; i++)
		{
			_population[i]     = new double[_len_individual];
			_end_individual[i] = new double[_len_individual];
			_sub_output_history[i] = new output_history_C(fitlen, num_out);
			_error[i] = new DiffEvolutionError(num_out);
			_error_trend[i] = error_trend_E.ET_UNKNOWN;
			_sub_output[i] = new double[num_out];
			_violation_sub_out_max[i] = new double[num_out];
			_violation_sub_out_sum[i] = new double[num_out];
			_do_sort[i] = false;
		}
		
		//set the differential weight
		if(F>=_min_F && F<=_max_F)
		{
			_F = F;
		}		
		else
		{
			System.err.println("DiffEvolution: invalid value of the differential weight");
			System.exit(1);
		}
		
		//set the probability of recombination
		if(CR>=_min_CR && CR<=_max_CR)
		{
			_CR = CR;
		}
		else
		{
			System.err.println("DiffEvolution: invalid value of the probability of recombination");
			System.exit(1);
		}

		_lim_opt = new interval_C[_len_individual][];
		_lim_opt_max = new double[_len_individual];
		
		//assign valid ranges of module's output bias
		for(i=0; i<num_out; i++)
		{
			//element of output bias  has always a single valid interval; therefore "[1]"
			_lim_opt[i] = new interval_C[1];
			
			_lim_opt[i][0] = new interval_C( range_output_bias[i] );
			lower_lim = range_output_bias[i].getLowerLimitAsDouble();
			upper_lim = range_output_bias[i].getUpperLimitAsDouble();
			if(Math.abs(lower_lim) > Math.abs(upper_lim))
			{
				_lim_opt_max[i] = Math.abs(lower_lim);
			}
			else
			{
				_lim_opt_max[i] = Math.abs(upper_lim);
			}
		}
		//assign valid ranges of responsibilities
		idx_gen = num_out;
		for(i=0; i<num_out; i++)
		{
			//element of responsibility  has always a single valid interval; therefore "[1]"
			_lim_opt[idx_gen] = new interval_C[1];
			
			_lim_opt[idx_gen][0] = new interval_C( range_responsibility[i] );
			lower_lim = range_responsibility[i].getLowerLimitAsDouble();
			upper_lim = range_responsibility[i].getUpperLimitAsDouble();
			if(Math.abs(lower_lim) > Math.abs(upper_lim))
			{
				_lim_opt_max[idx_gen] = Math.abs(lower_lim);
			}
			else
			{
				_lim_opt_max[idx_gen] = Math.abs(upper_lim);
			}
			idx_gen++;
		}
		//assign valid ranges of tuned states
		for(i=0; i<lim.length; i++)
		{
			_lim_opt[idx_gen] = new interval_C[lim[i].length];
			for(k=0; k<_lim_opt[idx_gen].length; k++)
			{
				_lim_opt[idx_gen][k] = new interval_C( lim[i][k] );
			}
			
			//lower limit of the 1st interval is always the smallest possible value
			lower_lim = lim[i][0].getLowerLimitAsDouble();
			//upper limit of the last interval is always the largest possible value
			idx_last_interval = lim[i].length - 1;
			upper_lim = lim[i][idx_last_interval].getUpperLimitAsDouble();
			if(Math.abs(lower_lim) > Math.abs(upper_lim))
			{
				_lim_opt_max[idx_gen] = Math.abs(lower_lim);
			}
			else
			{
				_lim_opt_max[idx_gen] = Math.abs(upper_lim);
			}
			idx_gen++;
		}
		//assign valid ranges of the input vector
		for(i=0; i<range_input.length; i++)
		{
			//element of responsibility  has always a single valid interval; therefore "[1]"
			_lim_opt[idx_gen] = new interval_C[1];
			_lim_opt[idx_gen][0] = new interval_C( range_input[i][0] );
			idx_gen++;
		}
		
		//assign valid ranges of output nodes
		_lim_sub_output = new interval_C[lim_sub_output.length][];
		for(i=0; i<_lim_sub_output.length; i++)
		{
			_lim_sub_output[i] = new interval_C[lim_sub_output[i].length];
			for(k=0; k<lim_sub_output[i].length; k++)
			{
				_lim_sub_output[i][k] = new interval_C( lim_sub_output[i][k] );
			}
			
			//compute the largest possible contribution of the module
			//"0" because there is always a single interval of output range
			max_val = Math.abs(_lim_sub_output[i][0].getLowerLimitAsDouble());
			if(max_val < Math.abs(_lim_sub_output[i][0].getUpperLimitAsDouble()))
			{
				max_val = Math.abs(_lim_sub_output[i][0].getUpperLimitAsDouble());
			}
			_max_sub_contribution[i] = _lim_opt_max[i]*max_val;
		}
		
		//allocate an array with valid intervals;
		//it will be assigned later with calling the function "setRangeEsnOutput()" 
		_lim_esn_output = new interval_C[lim_sub_output.length];
		for(i=0; i<_lim_sub_output.length; i++)
		{
			_lim_esn_output[i] = new interval_C();
		}
	}
	
	/**
	 * The function initializes elements of a provided individual with values that are uniformly distributed on
	 * the corresponding intervals [MIN_value, MAX_value]
	 * 
	 * @param individual: provided individual
	 */
	private void initializeRandom(double[] individual)
	{
		int j;
		int idx_interval;//chosen interval
		double lower_lim, upper_lim;//smallest and largest value of the current element
		double interval_length;//length of an interval for initialization of the current element
		double m;//mean value for the Gaussian distribution
		double sigma;//variance for the Gaussian distribution
		double responsibility;
		boolean do_gauss;//request to initialize a value according to the Gaussian distribution
		
		//generate a value for responsibility
		lower_lim = _lim_opt[_num_out][0].getLowerLimitAsDouble();
		upper_lim = _lim_opt[_num_out][0].getUpperLimitAsDouble();
		interval_length = upper_lim - lower_lim;
		do{
			responsibility  = (interval_length*_rand.nextDouble()) + lower_lim;
		}while(responsibility > upper_lim || responsibility < lower_lim);
		
		for(j=0; j<_len_individual; j++)
		{
			if(j < _num_out)//output bias
			{
				//"0" because there is always only one interval for an output bias
				idx_interval = 0;
				do_gauss = true;
			}
			else if(j>=_num_out && j < 2*_num_out)//responsibilities
			{
				//"0" because there is always only one interval for a responsibility
				idx_interval = 0;
				do_gauss = false;
			}
			else if(j>=_len_individual - _num_in)//input vector
			{
				//"0" because there is always only one interval for a responsibility
				idx_interval = 0;
				do_gauss = false;
			}
			else
			{
				idx_interval = _rand.nextInt(_lim_opt[j].length);
				do_gauss = false;
			}
			lower_lim = _lim_opt[j][idx_interval].getLowerLimitAsDouble();
			upper_lim = _lim_opt[j][idx_interval].getUpperLimitAsDouble();
			interval_length = upper_lim - lower_lim;

			if(do_gauss==true)
			{
				do{
					m = (upper_lim + lower_lim)/2;
					//sigma is the 3rd of a half of the whole range => sigma = (length/2)/3 = length/6 
					sigma = interval_length/6;
					individual[j] = sigma*_rand.nextGaussian() + m;
				}while(individual[j]<lower_lim || individual[j]>upper_lim);
			}
			else
			{
				//use a previously generated value for responsibility
				if(j>=_num_out && j < 2*_num_out)
				{
					individual[j] = responsibility;
				}
				else
				{
					individual[j] = (interval_length*_rand.nextDouble()) + lower_lim;
				}
			}
		}
	}
	
	/**
	 * The function initializes elements of reservoir states and the input element of the provided individual with
	 * values that are uniformly distributed on the corresponding intervals [MIN_value, MAX_value].
	 * 
	 * @param individual: provided individual
	 */
	private void initializeRandom_State_Input(double[] individual)
	{
		int j;
		int idx_interval;//chosen interval
		double lower_lim, upper_lim;//smallest and largest value of the current element
		double interval_length;//length of an interval for initialization of the current element
		boolean do_init;//request to do initialization of the current element
		
		for(j=0; j<_len_individual; j++)
		{
			if(j < _num_out)//output bias
			{
				idx_interval = -1;//dummy assignment
				do_init = false;//no initialization request
			}
			else if(j>=_num_out && j < 2*_num_out)//responsibilities
			{
				idx_interval = -1;//dummy assignment
				do_init = false;//no initialization request
			}
			else if(j>=_len_individual - _num_in)//input vector
			{
				//"0" because there is always only one interval for a responsibility
				idx_interval = 0;
				do_init = true;
			}
			else//reservoir states
			{
				idx_interval = _rand.nextInt(_lim_opt[j].length);
				do_init = true;
			}

			if(do_init==true)
			{
				lower_lim = _lim_opt[j][idx_interval].getLowerLimitAsDouble();
				upper_lim = _lim_opt[j][idx_interval].getUpperLimitAsDouble();
				interval_length = upper_lim - lower_lim;
				
				individual[j] = (interval_length*_rand.nextDouble()) + lower_lim;
			}
		}
	}
	
	/**
	 * The function initializes elements of a provided individual with values that are obtained in hyper-mutation of
	 * values of the best individual.
	 * A new value of each element is uniformly distributed on the interval whose zero is set to the corresponding
	 * value of the best individual.
	 * The upper limit is computed when a relative upper limit is added to the corresponding value of
	 * the best individual.
	 * The lower limit is computed when a relative lower limit is added to the corresponding value of
	 * the best individual.
	 * 
	 * @param individual: provided individual
	 */
	private void initializeHyperUniform(double[] individual)
	{
		int j;
		
		//TODO: implement this initialization method
		System.err.println("initializeHyperUniform: method is not realized yet");
		System.exit(1);

		for(j=0; j<_len_individual; j++)
		{

		}
	}
	
	/**
	 * The function initializes elements of a provided individual with values that are obtained in hyper-mutation of
	 * values of the best individual.
	 * A new value of each element is distributed according to a Gaussian distribution whose math expectation is set to
	 * the corresponding value of the best individual.
	 * 
	 * @param individual: provided individual
	 */
	private void initializeHyperGauss(double[] individual)
	{
		int j;
		
		//TODO: implement this initialization method
		System.err.println("initializeHyperGaussian: method is not realized yet");
		System.exit(1);

		for(j=0; j<_len_individual; j++)
		{

		}
	}
	
	/**
	 * The function checks whether a provided value is valid, that is, whether it is in one of valid intervals.
	 * 
	 * @param value: provided value to be checked
	 * @param valid_region: set of valid intervals
	 * @return "true": value is valid; "false" - otherwise
	 */
	private boolean isValueValid(double value, interval_C[] valid_region)
	{
		int i;
		double lower_lim;//smallest value of an interval
		double upper_lim;//largest value of an interval
		boolean is_valid;//output value
		
		is_valid = false;
		for(i=0; i<valid_region.length && is_valid==false; i++)
		{
			lower_lim = valid_region[i].getLowerLimitAsDouble();
			upper_lim = valid_region[i].getUpperLimitAsDouble();
			
			if(value>=lower_lim && value<=upper_lim)
			{
				is_valid = true;
			}
		}
		
		return is_valid;
	}
	
	/**
	 * The function indicates an index of valid interval containing the provided value.
	 * The function returns "-1" if the provided value does not belong to any valid interval.
	 * 
	 * @param value: provided value to be checked
	 * @param valid_region: set of valid intervals
	 * @return: index of relevant valid interval; "-1" if value does not belong to any interval
	 */
	private int getIdxValidInterval(double value, interval_C[] valid_region)
	{
		int i;
		int idx_interval;//output value
		double lower_lim;//smallest value of an interval
		double upper_lim;//largest value of an interval
		
		idx_interval = -1;
		for(i=0; i<valid_region.length && idx_interval==-1; i++)
		{
			lower_lim = valid_region[i].getLowerLimitAsDouble();
			upper_lim = valid_region[i].getUpperLimitAsDouble();
			
			if(value>=lower_lim && value<=upper_lim)
			{
				idx_interval = i;
			}
		}
		
		return idx_interval;
	}
	
	/**
	 * The function indicates an index of a valid interval whose border is the closest to the provided value.
	 * The function returns "-1" if the provided value belongs to one of valid intervals.
	 * 
	 * @param value: provided value to be checked
	 * @param valid_region: set of valid intervals
	 * @return: index of the closest valid interval; "-1" if value belongs to one of intervals
	 */
	private int getIdxClosestValidInterval(double value, interval_C[] valid_region)
	{
		int i;
		int idx_interval;//output value
		double lower_lim;//smallest value of an interval
		double upper_lim;//largest value of an interval
		double dist_to_lower_lim;//distance between a checked value and the lower limit
		double dist_to_upper_lim;//distance between a checked value and the upper limit
		double min_dist;//smallest found distance
		
		//initialize after comparison with the 1st interval
		lower_lim = valid_region[0].getLowerLimitAsDouble();
		upper_lim = valid_region[0].getUpperLimitAsDouble();
		if(value>=lower_lim && value<=upper_lim)
		{
			idx_interval = -1;//error condition
			min_dist = Double.POSITIVE_INFINITY;
		}
		else
		{
			dist_to_lower_lim = Math.abs(value - lower_lim);
			dist_to_upper_lim = Math.abs(value - upper_lim);
			if(dist_to_lower_lim < dist_to_upper_lim)
			{
				min_dist = dist_to_lower_lim;
			}
			else
			{
				min_dist = dist_to_upper_lim;
			}
			idx_interval = 0;
		}
		
		for(i=1; i<valid_region.length && idx_interval!=-1; i++)
		{
			lower_lim = valid_region[i].getLowerLimitAsDouble();
			upper_lim = valid_region[i].getUpperLimitAsDouble();
			if(value>=lower_lim && value<=upper_lim)
			{
				idx_interval = -1;//error condition
				min_dist = Double.POSITIVE_INFINITY;
			}
			else
			{
				dist_to_lower_lim = Math.abs(value - lower_lim);
				dist_to_upper_lim = Math.abs(value - upper_lim);
				if(dist_to_lower_lim < dist_to_upper_lim)
				{
					if(dist_to_lower_lim < min_dist)
					{
						min_dist = dist_to_lower_lim;
						idx_interval = i;
					}
				}
				else
				{
					if(dist_to_upper_lim < min_dist)
					{
						min_dist = dist_to_upper_lim;
						idx_interval = i;
					}
				}
			}
		}
		
		return idx_interval;
	}
	
	/**
	 * The function fetches an individual from the population according to the provided parameter value.
	 * @param param: provided parameter to fetch an individual at random (see a description of its type for details)
	 * @return: "-1" if it is not possible to fetch an individual according to the provided parameter value;
	 *          index of a fetched individual otherwise
	 */
	private int fetchIndividualAtRandom(param_rand_fetch_E param)
	{
		int i, j;
		int population_size;
		int num_out;//number of module outputs
		int idx;//(output variable) index of a chosen individual
		boolean all_in_range;//indicator that all output values are in range
		
		population_size = _population.length;
		
		switch(param)
		{
			case EVOL_PRF_ANY:
				idx = _rand.nextInt(population_size);
				break;
			case EVOL_PRF_VALID_OUT:
				//check whether there are individuals providing the module outputs in a valid range
				num_out = _violation_sub_out_max.length;
				all_in_range = false;
				for(i=0; i<population_size && all_in_range==false; i++)
				{
					all_in_range = true;
					for(j=0; j<num_out && all_in_range==true; j++)
					{
						if(_violation_sub_out_max[i][j] > 0)
						{
							all_in_range = false;
						}
					}
				}
				//fetch an appropriate individual at random if there is at least one in the population
				if(all_in_range==true)
				{
					do{
						idx = _rand.nextInt(population_size);
						all_in_range = true;
						for(j=0; j<num_out; j++)
						{
							if(_violation_sub_out_max[i][j] > 0)
							{
								all_in_range = false;
							}
						}
					}while(all_in_range==false);
				}
				else
				{
					idx = -1;
				}
				break;
			default:
				idx = -1;
				System.err.println("fetchIndividualAtRandom: unknown parameter value");
				System.exit(1);
				break;
		}
		
		return idx;
	}
	
	/**
	 * The function compares a performance of an offspring to that of its parent.
	 * 
	 * @param error_new: error of offspring
	 * @param idx_individ: index of parental individual
	 * @param max_violation_sub_out_par: largest violation of valid range among output neurons of parent
	 * @param max_violation_sub_out_off: largest violation of valid range among output neurons of offspring
	 * @return "true": offspring has a better performance; "false": otherwise
	 */
	private boolean comparePerformances(DiffEvolutionError error_new, int idx_individ, double max_violation_sub_out_par,
			                            double max_violation_sub_out_off)
	{
		boolean do_replace;//indicator that an offspring is better than its parent and should replace the latter
		boolean do_compare;//indicator that comparison of an offspring and its parent must be continued
		
		//select according to a ranked attributes
		do_replace = false;
		do_compare = true;
		//stage 1: comparison according to "error increasing"
		if(error_new.isIncrease()==false && _error[idx_individ].isIncrease()==true)
		{
			do_replace = true;
		}
		else if(error_new.isIncrease()==false && _error[idx_individ].isIncrease()==false)
		{
			//they are equally good => keep comparing an offspring and its parent
		}
		else
		{
			//either an offspring is worse or both are bad => stop comparison to keep the parent
			do_compare = false;
		}
		do_compare = true;
		
		//stage 2: comparison according to the largest violation of valid ranges by module outputs
		if(do_replace==false && do_compare==true)
		{
			if(max_violation_sub_out_off < max_violation_sub_out_par)
			{
				do_replace = true;
			}
			else if(max_violation_sub_out_off == max_violation_sub_out_par)
			{
				//they are equally good => keep comparing an offspring and its parent
			}
			else
			{
				//either an offspring is worse => stop comparison to keep the parent
				do_compare = false;
			}
		}
		do_compare = true;
		
		//stage 3: comparison according to average error
		if(do_replace==false && do_compare==true)
		{
			if(error_new.isBetter(_error[idx_individ].getAverageError())==true)
			{
				do_replace = true;
			}
			else if(error_new.isSame(_error[idx_individ].getAverageError())==true)
			{
				//they are equally good => keep comparing an offspring and its parent
			}
			else
			{
				//either an offspring is worse => stop comparison to keep the parent
				do_compare = false;
			}
		}

		return do_replace;
	}

	/**
	 * The function computes an output and an error of a provided offspring.
	 * The function needs an index of the corresponding parental individual to get outputs of the other modules.
	 * 
	 * @param offspring: provided offspring
	 * @param idx_individ: index of parental individual
	 * @param error_off: computed error of an offspring
	 * @param sub_output_off: module output of an offspring
	 * @param sub_output_curr: module output of an offspring at a current advancing time step
	 * @param violation_sub_out_max_off: largest violations over a fitness window by module outputs of offspring
	 * @param violation_sub_out_sum_off: sum of violations over a fitness window by module outputs of offspring
	 * @param sample_in: input values of samples to compute an error of the synchronized mESN
	 * @param sample_out: output values of samples to compute an error of the synchronized mESN
	 * @return largest violation of valid range among output neurons of offspring
	 */
	private double obtainOffspringBehavior(double[] offspring, int idx_individ, DiffEvolutionError error_off,
			                               double[] sub_output_off, double[] sub_output_curr,
			                               double[] violation_sub_out_max_off,
			                               double[] violation_sub_out_sum_off,
			                               double[][] sample_in, double[][] sample_out)
	{
		int j, k;
		double max_violation_sub_out_off;//output variables
		
		//assign new values to reservoir states and compute a module output for the current individual
		_esn.configDiffEvolutionDecodeIndividual(_sub_idx, offspring);
		//setting an module input is needed to compute its output
		//in this SW version, the input vector does not come from the config sequence but is obtained
		//as a product of synchronization
		//not needed _esn.setModuleNodes(_sub_idx, layer_type_E.LT_INPUT, sample_in[0]);
		_esn.computeSubOutput(_sub_idx, sub_output_off);
		//setting of module output is needed for further advancing
		_esn.setModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, sub_output_off);
		//start a nodes history of the new individual
		_esn.startModuleNodesHistory(_sub_idx);
		_esn.storeModuleNodesHistory(_sub_idx);

		//collect statistics of violations of valid ranges by module outputs of current offspring
		for(k=0; k<_num_out; k++)
		{
			violation_sub_out_max_off[k] = 0;
			violation_sub_out_sum_off[k] = 0;
		}
		updateModuleOutViolations(violation_sub_out_max_off,
				                  violation_sub_out_sum_off, sub_output_off);

		//advance a new individual on a fitness window
		for(j=1; j<sample_in.length; j++)
		{
			_esn.advanceModule(_sub_idx, null, true);
			_esn.getModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, sub_output_curr);
			updateModuleOutViolations(violation_sub_out_max_off,
		                              violation_sub_out_sum_off, sub_output_curr);
		}
		//collect statistics over all output neurons
		max_violation_sub_out_off = 0;
		for(j=0; j<_num_out; j++)
		{
			if(max_violation_sub_out_off < violation_sub_out_max_off[j])
			{
				max_violation_sub_out_off = violation_sub_out_max_off[j];
			}
		}

		//update fitness of the current individual
		_esn.configDiffEvolutionEvaluate(_sub_idx, error_off, idx_individ, sample_out, _lim_esn_output);
		
		return max_violation_sub_out_off;
	}
	
	/**
	 * The function replaces a parental individual with its offspring.
	 * 
	 * @param idx_individ: index of parental individual
	 * @param offspring: offspring to replace the parental individual
	 * @param error_off: error of an offspring
	 * @param sub_output_new: module output of an offspring
	 * @param violation_sub_out_max_off: largest violations over a fitness window by module outputs of offspring
	 * @param violation_sub_out_sum_off: sum of violations over a fitness window by module outputs of offspring
	 */
	private void doReplacement(int idx_individ, double[] individual_new, DiffEvolutionError error_new, double[] sub_output_new,
			                   double[] violation_sub_out_max_off, double[] violation_sub_out_sum_off)
	{
		int j;
		
		_error[idx_individ].copy(error_new);
		_error_trend[idx_individ] = error_trend_E.ET_UNKNOWN;//reset an error trend
		//_esn.configDiffEvolutionDisableSort(i);//because the error of offspring is not increasing
		for(j=0; j<_len_individual; j++)
		{
			_population[idx_individ][j] = individual_new[j];
		}
		for(j=0; j<_num_out; j++)
		{
			_sub_output[idx_individ][j] = sub_output_new[j];
		}
		_esn.getModuleNodesHistory(_sub_idx, layer_type_E.LT_OUTPUT, _sub_output_history[idx_individ].getArray());
		//TODO: remove it when getModuleNodesHistory is replaced with "_esn.getModuleNodesHistory().copy()"
		_sub_output_history[idx_individ].resetHistoryIdx();//new history starts with index "0"

		_esn.configDiffEvolutionEncodeIndividual(_sub_idx, _end_individual[idx_individ]);
		
		storeModuleOutViolations(idx_individ, violation_sub_out_max_off, violation_sub_out_sum_off);
	}
	
	/**
	 * The function computes a request to suppress activity in the i-th module according to a predefined probability.
	 * @return "true": module activity shall be suppressed; "false": otherwise
	 */
	private boolean suppressModule()
	{
		int    rand_number;
		int    int_probability;//probability as an integer
		double total_probab;//total probability to suppress activity in a module
		boolean output;
		
		//total_probab = module_probab*esn_probab;
		total_probab = _suppress_probab;
		
        int_probability = (int)(100*total_probab);
		
		rand_number = _rand.nextInt(100);
		
		if(rand_number < int_probability)
		{
			output = true;
		}
		else
		{
			output = false;
		}
		
		return output;
	}
	
	/**
	 * The function produces an offspring by DE on all elements of a parental individual.
	 * 
	 * @param idx_par: index of parental individual
	 * @param idx_out: index of the current module outputs
	 * @param offspring: offspring where reservoir states are elements
	 * @param max_violation_sub_out_par: largest violation of valid range among output neurons of parent
	 * @param sample_in: input values of samples to compute an error of the optimized ESN
	 * @param sample_out: output values of samples to compute an error of the optimized ESN
	 */
	private void produceResState(int idx_par, int idx_out, double[] offspring, double max_violation_sub_out_par,
			                     double[][] sample_in, double[][] sample_out)
	{
		int j;
		int a_idx, b_idx;//indices of individuals that are chosen for recombination with the current individual
		int R;//random index
		int idx_interval;//index of a relevant valid interval
		double a_val, b_val;//values of individuals chosen for recombination with the current individual
		double event_number;//value from [0, 1] that defines whether a recombination will be executed
		double y_j;//result of recombination
		double tmp_diff, tmp_diff2;//1st and 2nd difference term to create a donor vector
		double sigma;//standard deviation of a randomly distributed value
		double dist_to_lower;//distance of initial value to the lower border of its valid region
		double dist_to_upper;//distance of initial value to the upper border of its valid region
		double lower_lim, upper_lim;//left and right borders of an interval value
		boolean is_in_range;//indicator that an assigned random value is in a valid range

		//choose random individuals for recombination
		do{
			a_idx = fetchIndividualAtRandom(param_rand_fetch_E.EVOL_PRF_ANY);
		}while(a_idx==idx_par || a_idx==_best_idx);
		do{
			b_idx = fetchIndividualAtRandom(param_rand_fetch_E.EVOL_PRF_ANY);
		}while(b_idx==idx_par || b_idx==_best_idx || b_idx==a_idx);

		R = _rand.nextInt(_len_individual);

		for(j=0; j<_len_individual; j++)
		{
			//do not change responsibility
			if(j>=_num_out && j < 2*_num_out)
			{
				y_j = _population[idx_par][j];
			}
			else
			{
				a_val = _population[a_idx][j];
				b_val = _population[b_idx][j];

				event_number = _rand.nextDouble();

				//recombination
				if(event_number < _CR || j==R)
				{
					tmp_diff  = _F*(_population[_best_idx][j] - _population[idx_par][j]);
					tmp_diff2 = _F*(a_val - b_val);

					y_j  = _population[idx_par][j];
					y_j += tmp_diff;
					y_j += tmp_diff2;
				}
				else
				{
					y_j = _population[idx_par][j];
				}
				
				//needed call for possible mapping of sine phase to [0, 2pi]
				//"j - _num_out" to exclude responsibilities and the output bias
				y_j = _esn.getModule(_sub_idx).mapParamToEquivalentValue(j - 2*_num_out, y_j);

				//*** restrict the feasibility region ***

				is_in_range = isValueValid(y_j, _lim_opt[j]);
				if(is_in_range==false)
				{
					//check whether the parent value itself is in a valid range
					//   (it can be out-of-range as a result of previous "advancePopulation()")
					//if not then try to gradually move a new value into the valid range
					is_in_range = isValueValid(_population[idx_par][j], _lim_opt[j]);
					if(is_in_range==true)
					{
						idx_interval = getIdxValidInterval(_population[idx_par][j], _lim_opt[j]);
						lower_lim = _lim_opt[j][idx_interval].getLowerLimitAsDouble();
						upper_lim = _lim_opt[j][idx_interval].getUpperLimitAsDouble();

						//assign normally distributed value randomly distributed with "sigma = dist to close border / 3"
						dist_to_lower = _population[idx_par][j] - lower_lim;
						dist_to_upper = upper_lim - _population[idx_par][j];
						if(dist_to_lower < dist_to_upper)
						{
							sigma = dist_to_lower / 3;
						}
						else
						{
							sigma = dist_to_upper / 3;
						}
						//generate a new value until it is in the valid range
						do{
							y_j  = _rand.nextGaussian();
							y_j *= sigma;
							y_j += _population[idx_par][j];

							//check a newly generated value against borders of a valid range
							//(if the random value is behind "3*sigma" then it can lie again outside the valid range)
							is_in_range = true;
							if(y_j < lower_lim ||
							   y_j > upper_lim)
							{
								is_in_range = false;
							}
						}while(is_in_range==false);
					}
					else
					{
						idx_interval = getIdxClosestValidInterval(_population[idx_par][j], _lim_opt[j]);
						lower_lim = _lim_opt[j][idx_interval].getLowerLimitAsDouble();
						upper_lim = _lim_opt[j][idx_interval].getUpperLimitAsDouble();

						//parent value is smaller or coincides with the lower border
						if(_population[idx_par][j] <= lower_lim)
						{
							//"3*sigma" is computed from a distance from the far border (upper border)
							dist_to_upper = upper_lim - _population[idx_par][j];
							sigma = dist_to_upper / 3;
							//generate a new value until it is below the far border (below upper border)
							do{
								//in order to move a new value into the valid range, a random value must be generated
								//strictly on the right side
								do{
									y_j  = _rand.nextGaussian();
								}while(y_j <= 0);
								y_j *= sigma;
								y_j += _population[idx_par][j];
							}while(y_j >= upper_lim);
						}
						//parent value is smaller or coincides with the upper border
						else if(_population[idx_par][j] >= upper_lim)
						{
							//"3*sigma" is computed from a distance from the far border (lower border)
							dist_to_upper = _population[idx_par][j] - lower_lim;
							sigma = dist_to_upper / 3;
							//generate a new value until it is above the far border (above lower border)
							do{
								//in order to move a new value into the valid range, a random value must be generated
								//strictly on the left side
								do{
									y_j  = _rand.nextGaussian();
								}while(y_j >= 0);
								y_j *= sigma;
								y_j += _population[idx_par][j];
							}while(y_j <= lower_lim);
						}
						else
						{
							System.err.println("createNextGeneration: parental value must be out of valid range");
							System.exit(1);
						}
					}
				}
				else
				{
					//keep a previously computed value
				}
			}

			//assign of a previously computed and restricted value at the current element of a new individual
			offspring[j] = y_j;
		}//for "j" over elements of one individual
	}
	
	/**
	 * The function produces a new responsibility for a specified individual.
	 * 
	 * @param idx_par: index of a parental individual
	 * @param offspring: offspring where responsibility is an element
	 * @param resp_thresh: threshold of responsibility to detect when generated value is close to a valid limit
	 */
	private void produceResponsibility(int idx_par, double[] offspring, double resp_thresh)
	{
		int j;
		double sigma;//used variance of the Gaussian distribution
		double resp_offspr;//produced responsibility
		double resp_parent;//responsibility of the parent
		double lower_lim, upper_lim;//smallest and largest value of responsibility
		double dist_to_lower, dist_to_upper;//distance of parent responsibility to the lower and upper limit of the valid range
		
		resp_parent = _population[idx_par][_num_out];
		lower_lim = _lim_opt[_num_out][0].getLowerLimitAsDouble();
		upper_lim = _lim_opt[_num_out][0].getUpperLimitAsDouble();
		dist_to_lower = resp_parent - lower_lim;
		dist_to_upper = upper_lim - resp_parent;
		
		//find variance of responsibility depending on the parent responsibility
		if(dist_to_lower < dist_to_upper)
		{
			sigma = dist_to_lower / 3;
		}
		else
		{
			sigma = dist_to_upper / 3;
		}
		
		//produce new responsibility for both outputs
		do{
			resp_offspr  = sigma*_rand.nextGaussian();
			//move the distribution curve to "m = parental_value" from "m = 0.0";
			//"parental_value" is the same for both outputs
			resp_offspr += resp_parent;
		}while(resp_offspr > upper_lim || resp_offspr < lower_lim);
		
		//try to round new responsibility value down
		if(resp_offspr < lower_lim+resp_thresh)
		{
			resp_offspr = lower_lim;
		}
		//try to round new responsibility value up
		if(resp_offspr > upper_lim-resp_thresh)
		{
			resp_offspr = upper_lim;
		}

		for(j=0; j<_len_individual; j++)
		{
			//output bias is reset because it is not needed for an inactive module
			if(j < _num_out)
			{
				offspring[j] = 0;
			}
			else if((j >= _num_out) && (j < 2*_num_out))//responsibility can only be suppressed
			{
				offspring[j] = resp_offspr;
			}
			else//module states are simply taken over
			{
				//simply take over reservoir states from the parent; 
				//reservoir states are produced in the function "produceResState()"
				offspring[j] = _population[idx_par][j];
			}
		}//for "j" over elements of one individual
	}
	
	/**
	 * The function creates individuals of the next generation.
	 * 
	 * @param sample_in: input values of samples to compute an error of the optimized ESN
	 * @param sample_out: output values of samples to compute an error of the optimized ESN
	 * @param resp_thresh: threshold of responsibility to detect when generated value is close to a valid limit
	 */
	private void createNextGeneration(double[][] sample_in, double[][] sample_out, double resp_thresh)
	{
		int i, j;
		int population_size;
		double max_violation_sub_out_off;//largest violation of valid range among output neurons of offspring
		double max_violation_sub_out_par;//largest violation of valid range among output neurons of parent
		double[] offspring;//offspring
		double[] sub_output_off;//module output of an offspring
		double[] sub_output_cur;//module output of an offspring at a current advancing time step
		double[] violation_sub_out_max_off;//largest violations over a fitness window by module outputs of offspring
		double[] violation_sub_out_sum_off;//sum of violations over a fitness window by module outputs of offspring
		boolean do_replace;//indicator that offspring shall replace its parent
		boolean do_normal;//request to generate an offspring value according to normal distribution
		DiffEvolutionError error_off;//computed error of an offspring
		
		error_off      = new DiffEvolutionError(_num_out);
		sub_output_off = new double[_num_out];
		sub_output_cur = new double[_num_out];
		offspring      = new double[_len_individual];
		violation_sub_out_max_off = new double[_num_out];
		violation_sub_out_sum_off = new double[_num_out];
		
		population_size = _population.length;
		for(i=0; i<population_size; i++)
		{
			//check whether the current individual has a valid output of sub-reservoirs
			max_violation_sub_out_par = 0;
			for(j=0; j<_num_out; j++)
			{
				if(max_violation_sub_out_par < _violation_sub_out_max[i][j])
				{
					max_violation_sub_out_par = _violation_sub_out_max[i][j];
				}
			}
			if(max_violation_sub_out_par > 0)
			{
				_esn.configDiffEvolutionEnableSort(i);
			}
			
			//assign responsibility for calculations below
			getResponsibilityByIndex(i, _tmp_responsibility);
			
			do_normal = suppressModule();

			//Choice between mutation of reservoir states and module responsibility is done independently of each other
			if(do_normal==true)
			{
				//generation of suitable responsibility before more difficult generation of new reservoir states
				produceResponsibility(i, offspring, resp_thresh);
				max_violation_sub_out_off = obtainOffspringBehavior(offspring, i, error_off, sub_output_off, sub_output_cur, violation_sub_out_max_off, violation_sub_out_sum_off, sample_in, sample_out);
				do_replace = comparePerformances(error_off, i, max_violation_sub_out_par, max_violation_sub_out_off);
				if(do_replace==true)
				{
					doReplacement(i, offspring, error_off, sub_output_off, violation_sub_out_max_off, violation_sub_out_sum_off);
				}
			}
			else
			{
				//go over output nodes;

				//Thus, it can be that an individual is mutated several times as compared to a module with a single output.
				//Independence of the choice for all module outputs provides independent scaling of separate module outputs.
				for(j=0; j<_num_out; j++)
				{
					//reservoir states are generated only if it was not possible to improve a module output with
					//a new responsibility of module output
					produceResState(i, j, offspring, max_violation_sub_out_par, sample_in, sample_out);
					max_violation_sub_out_off = obtainOffspringBehavior(offspring, i, error_off, sub_output_off, sub_output_cur, violation_sub_out_max_off, violation_sub_out_sum_off, sample_in, sample_out);
					do_replace = comparePerformances(error_off, i, max_violation_sub_out_par, max_violation_sub_out_off);
					if(do_replace==true)
					{
						doReplacement(i, offspring, error_off, sub_output_off, violation_sub_out_max_off, violation_sub_out_sum_off);
					}
				}
			}
		}//for "i" over population
		//System.out.println("createNextGeneration: number of reset fitness is " + dummy_cnt_reset_fitness);
		//System.out.println("createNextGeneration: number of replacements is " + dummy_cnt_replacement);
	}
	
	/**
	 * The function stores a specified individual and output of its sub-reservoir in the provided storage.
	 * The individual is specified by its index.
	 * If storing of either individual or its module output is not needed, the corresponding parameter must be
	 * set to "null".
	 * 
	 * @param storage_individ: provided storage for the individual
	 * @param storage_subout: provided storage for sub-reservoir output
	 * @param idx: index of individual
	 */
	private void storeIndividual(double[] storage_individ, double[] storage_subout, int idx)
	{
		int i;
		int size;//size of copied array

		if(storage_individ!=null)
		{
			size = storage_individ.length;
			for(i=0; i<size; i++)
			{
				storage_individ[i] = _population[idx][i];
			}
		}
		
		if(storage_subout!=null)
		{
			size = storage_subout.length;
			for(i=0; i<size; i++)
			{
				storage_subout[i] = _sub_output[idx][i];
			}
		}
	}
	
	/**
	 * The function stores a specified end individual and an output of its module in provided storages.
	 * The end individual is specified by its index.
	 * 
	 * @param storage_end_individ: provided storage for the end individual
	 * @param storage_end_subout: provided storage for module output
	 * @param idx: index of end individual
	 */
	private void storeEndIndividual(double[] storage_end_individ, double[] storage_end_subout, int idx)
	{
		int i;
		int size;//size of copied array
		
		size = storage_end_individ.length;
		for(i=0; i<size; i++)
		{
			storage_end_individ[i] = _end_individual[idx][i];
		}
		
		_sub_output_history[idx].getOutputByIndex(_fitness_length-1, storage_end_subout);
	}
	
	/**
	 * The function updates statistics about violations of valid ranges by module outputs of an individual.
	 * The function must be called for one individual at every time only once.
	 * Module outputs are provided as an input array.
	 *
	 * @param idx_individ: index of individual
	 * @param output: array with module outputs of a single individual
	 */
	private void updateModuleOutViolations(int idx_individ, double[] output)
	{
		int j;
		int idx_last_interval;//index of the last valid interval of a node
		double exaggeration;
		double lower_lim, upper_lim;//left and right borders of an interval
		
		for(j=0; j<_num_out; j++)
		{
			//lower limit of the 1st interval is always the smallest possible output of a module
			lower_lim = _lim_sub_output[j][0].getLowerLimitAsDouble();
			//upper limit of the last interval is always the largest possible output of a module
			idx_last_interval = _lim_sub_output[j].length - 1;
			upper_lim = _lim_sub_output[j][idx_last_interval].getUpperLimitAsDouble();
			
			if(output[j] < lower_lim)
			{
				exaggeration = Math.abs(lower_lim - output[j]);
			}
			else if(output[j] > upper_lim)
			{
				exaggeration = Math.abs(output[j] - upper_lim);
			}
			else
			{
				exaggeration = 0;
			}
			
			//update statistics
			if(exaggeration > _violation_sub_out_max[idx_individ][j])
			{
				_violation_sub_out_max[idx_individ][j] = exaggeration;
			}
			_violation_sub_out_sum[idx_individ][j] += exaggeration;
		}
	}
	
	/**
	 * The function updates statistics about violations of valid ranges by module outputs of an individual.
	 * Arrays with current statistics and module outputs are provided as input arrays.
	 * 
	 * @param violation_sub_out_max: largest violation of valid ranges for each output neuron
	 * @param violation_sub_out_sum: sum of violation of valid ranges for each output neuron
	 * @param output: current module outputs
	 */
	private void updateModuleOutViolations(double[] violation_sub_out_max,
			                               double[] violation_sub_out_sum,
			                               double[] output)
	{
		int j;
		int idx_last_interval;//index of the last valid interval of a node
		double exaggeration;
		double lower_lim, upper_lim;//left and right borders of an interval
		
		for(j=0; j<_num_out; j++)
		{
			//lower limit of the 1st interval is always the smallest possible output of a module
			lower_lim = _lim_sub_output[j][0].getLowerLimitAsDouble();
			//upper limit of the last interval is always the largest possible output of a module
			idx_last_interval = _lim_sub_output[j].length - 1;
			upper_lim = _lim_sub_output[j][idx_last_interval].getUpperLimitAsDouble();
			if(output[j] < lower_lim)
			{
				exaggeration = Math.abs(lower_lim - output[j]);
			}
			else if(output[j] > upper_lim)
			{
				exaggeration = Math.abs(output[j] - upper_lim);
			}
			else
			{
				exaggeration = 0;
			}
			
			//update statistics
			if(exaggeration > violation_sub_out_max[j])
			{
				violation_sub_out_max[j] = exaggeration;
			}
			violation_sub_out_sum[j] += exaggeration;
		}
	}
	
	/**
	 * The function stores provided statistics of violations by module outputs as the statistics of a specified
	 * individual.
	 * 
	 * @param idx_individ: index of specified individual
	 * @param violation_sub_out_max: largest violation for each output neuron
	 * @param violation_sub_out_sum: sum of violations for each output neuron
	 */
	private void storeModuleOutViolations(int idx_individ,double[] violation_sub_out_max,double[] violation_sub_out_sum)
	{
		int i;
		
		for(i=0; i<_num_out; i++)
		{
			_violation_sub_out_max[idx_individ][i] = violation_sub_out_max[i];
			_violation_sub_out_sum[idx_individ][i] = violation_sub_out_sum[i];
		}
	}
	
	/**
	 * The function resets all individuals of the population because no evolution for this module is needed.
	 */
	public void resetPopulation()
	{
		int i, j;
		
		_is_inactive = true;//assignment necessary for deactivation of SIN modules by the counter-phase
		
		//reset all individual of a population
		for(i=0; i<_population.length; i++)
		{
			for(j=0; j<_len_individual; j++)
			{
				_population    [i][j] = 0;
				_end_individual[i][j] = 0;
			}
			for(j=0; j<_num_out; j++)
			{
				_sub_output[i][j] = 0;
			}
			_sub_output_history[i].resetHistory();
		}
	}
	
	/**
	 * The function initializes elements of every individual in the population with random numbers uniformly
	 * distributed on the interval [low_lim_i, up_lim_i] where "low_lim_i" is the lower limit of the i-th
	 * element and "low_lim_i" is the upper limit of the i-th element.
	 * The function computes a development of generated individuals as a history of their outputs.
	 * The function computes the development of the individuals using an array of provided input samples.
	 *
	 * @param init: initialization method
	 * @param sample_in: input values of samples to compute the development of individuals
	 */
	public void initializePopulation(config_ea_init_E init, double[][] sample_in)
	{
		int i, j;
		int num_out;//number of output neurons
		int population_size;
		boolean all_in_range;//indicator that all output values are in range
		double[] sub_output_loc;//local array to store an output of considered sub-reservoir for all output neurons
		
		//store the output of the considered sub-reservoir
		num_out = _sub_output[0].length;
		sub_output_loc = new double[num_out];
		
		population_size = _population.length;
		for(i=0; i<population_size; i++)
		{
			do{
				if(_is_initialized==false)
				{
					initializeRandom(_population[i]);
				}
				else
				{
					switch(init)
					{
						case CFG_EA_INIT_NONE:
							System.err.println("initializePopulation: function is called but no init is needed");
							System.exit(1);
							break;
						case CFG_EA_INIT_RANDOM:
							initializeRandom(_population[i]);
							break;
						case CFG_EA_INIT_RANDOM_RES_STATES_INPUT:
							initializeRandom_State_Input(_population[i]);
							break;
						case CFG_EA_INIT_HYPER_UNIFORM:
							initializeHyperUniform(_population[i]);
							break;
						case CFG_EA_INIT_HYPER_GAUSS:
							initializeHyperGauss(_population[i]);
							break;
						default:
							System.err.println("initializePopulation: unknown initialization method");
							System.exit(1);
							break;
					}
				}
				//assign new values to reservoir states and compute the sub-reservoir output for the current individual
				_esn.configDiffEvolutionDecodeIndividual(_sub_idx, _population[i]);
				//setting an module input is needed to compute its output
			    //in this SW version, the input vector does not come from the config sequence but is obtained
				//as a product of synchronization
				//not needed _esn.setModuleNodes(_sub_idx, layer_type_E.LT_INPUT, sample_in[0]);
				_esn.computeSubOutput(_sub_idx, _sub_output[i]);
				_sub_output_history[i].storeNextElement(_sub_output[i]);
				//setting of module output is needed for further advancing
				_esn.setModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, _sub_output[i]);
				
				//check whether the generated individual provides a valid output of sub-reservoirs
				all_in_range = true;
				for(j=0; j<num_out && all_in_range==true; j++)
				{
					all_in_range &= isValueValid(_sub_output[i][j], _lim_sub_output[j]);
				}
				
				//it makes sense to advance an individual only if its 1st output was in range
				if(all_in_range==true)
				{
					//initialize an array with violations of the current individual before following advancing
					for(j=0; j<num_out; j++)
					{
						_violation_sub_out_max[i][j] = 0;
						_violation_sub_out_sum[i][j] = 0;
					}
					//advance a new individual on a fitness window;
					for(j=1; j<sample_in.length; j++)
					{
						_esn.advanceModule(_sub_idx, null, false);
						_esn.getModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, sub_output_loc);
						_sub_output_history[i].storeNextElement(sub_output_loc);
						updateModuleOutViolations(i, sub_output_loc);
					}
					//store the current status as a status at the last step of a window
					_esn.configDiffEvolutionEncodeIndividual(_sub_idx, _end_individual[i]);
				}
				else
				{
					_sub_output_history[i].resetHistoryIdx();
				}
			}while(all_in_range==false);
		}//for i
		
		//since fitness has not been computed yet, assign the 1st individual to be the best one
		//(in general an assignment of an arbitrary individual is equally good)
		_best_idx = 0;
		
		_is_initialized = true;
	}
	
	/**
	 * The function advances all individuals of the populations by one time step
	 * The provided array is an external ESN's input in the new time step.
	 * The function advances a pool of the best individuals as well.
	 * 
	 * @param sample_in: external ESN's input in the new time step
	 */
	public void advancePopulation(double[] sample_in)
	{
		int      i;
		int      population_size;
		double[] output_end;//module output at the end step of a fitness window
		
		output_end      = new double[_num_out];
		population_size = _population.length;
		for(i=0; i<population_size; i++)
		{
			//advance an individual
			_esn.configDiffEvolutionDecodeIndividual(_sub_idx, _population[i]);
			_esn.setModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, _sub_output[i]);
			//update an ESN module of the current population
			_esn.advanceModule(_sub_idx, null, false);
			//store an updated individual and a new value of a module output
			_esn.configDiffEvolutionEncodeIndividual(_sub_idx, _population[i]);
			_esn.getModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, _sub_output[i]);
			
			//advance a set of end individuals
			_sub_output_history[i].getOutputByIndex(_fitness_length-1, output_end);
			_esn.configDiffEvolutionDecodeIndividual(_sub_idx, _end_individual[i]);
			_esn.setModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, output_end);
			//update an ESN module of the current population
			_esn.advanceModule(_sub_idx, null, false);
			//store an updated end individual and a value of a module output
			_esn.configDiffEvolutionEncodeIndividual(_sub_idx, _end_individual[i]);
			_esn.getModuleNodes(_sub_idx, layer_type_E.LT_OUTPUT, output_end);
			//store a new state of the end individual
			_sub_output_history[i].storeNextElement(output_end);
			
			//update violations for the currently computed output
			updateModuleOutViolations(i, output_end);
		}//for i
	}
	
	/**
	 * The function searches for the best individual in the population.
	 *
	 * @return: index of the best individual in the population
	 */
	public void findBestVectorIdx()
	{
		int i;
		int population_size;
		
		population_size = _population.length;
		
		_best_idx  = 0;//it should stay "-1" if no individual satisfied a required error trend 
		for(i=_best_idx+1; i<population_size; i++)
		{
			if(_error[i].isBetter(_error[_best_idx].getAverageError())==true)
			{
				_best_idx = i;
			}
		}
	}
	
	/**
	 * The function sorts individuals of the population according to their errors in the ascending order.
	 * 
	 * return: number of sorted individuals
	 */
	public int sortPopulation()
	{
		int i, j;
		int population_size;
		int num_sort;//number of individuals to be sorted in the beginning on the population
		int num_sort_true;//number of candidate individuals for sorting
		int num_sort_miss;//number of missing individuals in a list for sorting
		int idx_worst;//index of the next worst individual to be added for sorting
		double[] error_worst;//error of the currently worst individual
		double[] sub_output_tmp;//temporary array to keep module outputs
		double[] violation_tmp;//temporary array to keep a statistics of violations
		double[] individual;//temporary array to keep an individual
		DiffEvolutionError error_tmp;//temporary variable to keep an error status
		error_trend_E error_trend_tmp;//temporary variable to keep a trend of a prediction error
		output_history_C history_tmp;//temporary object to keep an output history
		
		population_size = _population.length;
		
		//add individuals for sorting only if required min number of individuals is larger than 0
		if(_sort_min_num > 0)
		{
			//compute a number of individuals that are current candidates for sorting
			num_sort_true = 0;
			for(i=0; i<population_size; i++)
			{
				if(_do_sort[i]==true)
				{
					num_sort_true++;
				}
			}
			num_sort_miss = _sort_min_num - num_sort_true;

			//choose the worst individuals as additional individuals to be sorted
			for(i=0; i<num_sort_miss; i++)
			{
				idx_worst = -1;
				error_worst = null;//dummy assignment which is never used
				for(j=0; j<population_size; j++)
				{
					if(_do_sort[j]==false)
					{
						if(idx_worst==-1)//none individual has been added yet in this loop
						{
							idx_worst = j;
							error_worst = _error[j].getAverageError();
						}
						else
						{
							if(_error[j].isWorse(error_worst)==true)
							{
								idx_worst = j;
								error_worst = _error[j].getAverageError();
							}
						}
					}
				}//for j
				//assign the worst individual for sorting
				_esn.configDiffEvolutionEnableSort(idx_worst);
			}//for i
		}//if more individuals for sorting must be added (_sort_min_num > 0)
		
		//move unsortable individuals to the upper part of the population
		
		//find the 1st individual that can be moved
		j = population_size-1;
		while(j>=0 && _do_sort[j]==false)
		{
			j--;
		}
		//if none of individuals is enabled for the sorting then "j==-1"
		for(i=0; j>0 && i<population_size && j>i; i++)
		{
			if(_do_sort[i]==false)
			{
				_esn.configDiffEvolutionExchangeIndividuals(i, j);
				
				j--;
				while(j>=0 && _do_sort[j]==false)
				{
					j--;
				}
			}
		}
		
		//obtain a number of individuals to be sorted
		num_sort = -1;//value "-1" is used as a marker in the end of the function
		for(i=0; i<population_size && _do_sort[i]==true; i++)
		{
			num_sort = i + 1;
		}
		
		//sort individuals that are enabled for the sorting
		for(i=0; i<num_sort-1; i++)
		{
			for(j=i+1; j<num_sort; j++)
			{
				if(_error[j].isBetter(_error[i].getAverageError())==true)
				{
					error_tmp = _error[i];
					_error[i] = _error[j];
					_error[j] = error_tmp;
					
					error_trend_tmp = _error_trend[i];
					_error_trend[i] = _error_trend[j];
					_error_trend[j] = error_trend_tmp;
					
					sub_output_tmp = _sub_output[i];
					_sub_output[i] = _sub_output[j];
					_sub_output[j] = sub_output_tmp;
					
					violation_tmp             = _violation_sub_out_max[i];
					_violation_sub_out_max[i] = _violation_sub_out_max[j];
					_violation_sub_out_max[j] = violation_tmp;
					
					violation_tmp             = _violation_sub_out_sum[i];
					_violation_sub_out_sum[i] = _violation_sub_out_sum[j];
					_violation_sub_out_sum[j] = violation_tmp;
					
					individual     = _population[i];
					_population[i] = _population[j];
					_population[j] = individual;
					
					individual         = _end_individual[i];
					_end_individual[i] = _end_individual[j];
					_end_individual[j] = individual;
					
					history_tmp            = _sub_output_history[i];
					_sub_output_history[i] = _sub_output_history[j];
					_sub_output_history[j] = history_tmp;
				}
			}
		}
		
		//assign "0" instead of "-1" if there were no sorted individuals
		if(num_sort==-1)
		{
			num_sort = 0;
		}
		
		return num_sort;
	}
	
	/**
	 * The function runs differential evolution to produce a single generation.
	 * If output of the best individual is very small, the function sets all individuals to "0".
	 *
	 * @param sample_in: input values of samples to compute fitness
	 * @param sample_out: output values of samples to compute fitness
	 * @param resp_thresh: threshold of responsibility to detect when generated value is close to a valid limit
	 */
	public void run(double[][] sample_in, double[][] sample_out, double resp_thresh)
	{
		//it is not allowed to call evolution for an inactive population
		if(_is_inactive==true)
		{
			System.err.println("DiffEvolution.run: no evolution is allowed for inactive population");
			System.exit(1);
		}
		
		//find an index of the best individual for the next generation
		findBestVectorIdx();
		
		//create the next generation
		createNextGeneration(sample_in, sample_out, resp_thresh);
	}
	
	/**
	 * The function computes errors for all individuals on a provided set of samples.
	 * The function uses the computed errors to update an error status of the individuals
	 * 
	 * @param sample_in: input values of a provided set of samples
	 * @param sample_out: output values of a provided set of samples
	 */
	public void updateErrorAll(double[][] sample_in, double[][] sample_out)
	{
		int i;
		int population_size;
		DiffEvolutionError error;//object to store an error of the current individual

		error = new DiffEvolutionError(_num_out);
		
		population_size = _population.length;
		for(i=0; i<population_size; i++)
		{
			_esn.configDiffEvolutionEvaluate(-1, error, i, sample_out, _lim_esn_output);
			_error[i].copy( error );

			//check whether the error increases over the last errors
			if(_error[i].isIncrease()==true)
			{
				_esn.configDiffEvolutionEnableSort(i);
			}
			else
			{
				_esn.configDiffEvolutionDisableSort(i);
			}
		}
	}
	
	/**
	 * The function returns an index of the best individual in the population.
	 * 
	 * @return: index of the best individual
	 */
	public int getBestIndividual()
	{
		return _best_idx;
	}
	
	/**
	 * The function calls to fetch a random individual from the population.
	 * First, it tries to fetch an individual whose module outputs are in a valid range.
	 * If the population does not contain such individuals then the function fetches an arbitrary individual.
	 * 
	 * @return: index of a fetched individual
	 */
	public int getIndividualAtRandom()
	{
		int idx;//index of a fetched individual
		
		//fetch a random individual whose module outputs are in a valid range
		idx = fetchIndividualAtRandom(param_rand_fetch_E.EVOL_PRF_VALID_OUT);
	    //fetch a random individual if no individual has all module outputs in a valid range
		if(idx==-1)
		{
			idx = fetchIndividualAtRandom(param_rand_fetch_E.EVOL_PRF_ANY);
		}
		
		return idx;
	}
	
	/**
	 * The function stores a specified individual and its output in provided storages.
	 * The individual is specified by its index.
	 * 
	 * @param idx: index of the individual
	 * @param individual: genotype of the individual
	 * @param sub_output: module outputs of the individual
	 */
	public void getIndividualByIndex(int idx, double[] individual, double[] sub_output)
	{
		storeIndividual(individual, sub_output, idx);
	}
	
	/**
	 * The function returns a specified individual.
	 * The individual is specified by its index.
	 * 
	 * @param idx: index of the individual
	 * @return: requested individual
	 */	
	public double[] getIndividualByIndex(int idx)
	{
		return _population[idx];
	}
	
	/**
	 * The function stores a specified end individual and its output in a provided storage.
	 * The end individual is specified by its index.
	 * 
	 * @param idx: index of the end individual
	 * @param end_sub_output: module outputs of the end individual
	 * @return: array with contents of the end individual
	 */
	public double[] getEndIndividualByIndex(int idx, double[] end_sub_output)
	{
		double[] end_individual;//storage to save the end individual
		
		end_individual = new double[_len_individual];
		
		storeEndIndividual(end_individual, end_sub_output, idx);
		
		return end_individual;
	}
	
	/**
	 * The function retrieves an output of a specified individual at a required time step of a fitness window.
	 * The individual is specified by its index in a population.
	 * The time step is an offset with respect to the 1st time step of a fitness window.
	 * The function stores a retrieved output in a provided array.
	 * 
	 * @param idx_individual: index of an individual in a fitness window
	 * @param time_offset: offset of a time step within a fitness window
	 * @param storage: array where an output must be stored
	 */
	public void getOutputByIndex(int idx_individual, int time_offset, double[] storage)
	{
		_sub_output_history[idx_individual].getOutputByIndex(time_offset, storage);
	}
	
	/**
	 * The function retrieves the output bias of the module from the specified individual.
	 * The individual is specified by its index.
	 * 
	 * @param idx_individual: index of individual
	 * @param out_bias: output storage for the retrieved output bias
	 */
	public void getOutputBiasByIndex(int idx_individual, double[] out_bias)
	{
		int i;
		
		for(i=0; i<_num_out; i++)
		{
			out_bias[i] = _population[idx_individual][i];
		}
	}
	
	/**
	 * The function retrieves responsibility of the module from the specified individual.
	 * The individual is specified by its index.
	 * 
	 * @param idx_individual: index of individual
	 * @param responsibility: output storage for retrieved responsibilities
	 */
	public void getResponsibilityByIndex(int idx_individual, double[] responsibility)
	{
		int i;
		int idx_element;//currently retrieved element
		
		idx_element = 0;
		for(i=_num_out; i<2*_num_out; i++)
		{
			responsibility[idx_element] = _population[idx_individual][i];
			idx_element++;
		}
	}
	
	/**
	 * The function indicates an average error of the best individual.
	 * 
	 * @return: average error of the best individual
	 */
	public double[] getErrorBestAverage()
	{
		return _error[_best_idx].getAverageError();
	}
	
	/**
	 * The function indicates the largest element of the average error vector of the best individual.
	 * 
	 * @return: largest element of the average error vector of the best individual
	 */
	public double getErrorBestMaxAverage()
	{
		return _error[_best_idx].getMaxAverageMse();
	}
	
	/**
	 * The function returns the whole error status of the specified individual.
	 * The individual is specified by its index.
	 * 
	 * @param idx: index of an individual
	 * @return: full error status
	 */
	public DiffEvolutionError getErrorFull(int idx)
	{
		return _error[idx];
	}
	
	/**
	 * The function indicates whether a prediction error of a specified individual has an acceptable error trend.
	 * The individual is specified by its index.
	 * 
	 * @return: "true" if error is increasing; "false" - otherwise
	 */
	public boolean isAcceptableErrorTrend(int idx)
	{
		boolean is_acceptable;
		
		is_acceptable = false;
		if(_error_trend[idx]==error_trend_E.ET_DECREASE                      ||
		   _error_trend[idx]==error_trend_E.ET_INCREASE_LESS_THAN_1PERCENT   ||
		   _error_trend[idx]==error_trend_E.ET_INCREASE_LESS_THAN_10PERCENT  ||
		   _error_trend[idx]==error_trend_E.ET_INCREASE_LESS_THAN_100PERCENT)
		{
			is_acceptable = true;
		}
		
		return is_acceptable;
	}
	
	/**
	 * The function indicates whether an error trend was computed for at least one individual.
	 * 
	 * @return: "true" - error trend was computed; "false" - otherwise
	 */
	public boolean isErrorTrendComputed()
	{
		int i;
		int pop_size;
		boolean is_computed;
		
		is_computed = false;
		pop_size = _population.length;
		for(i=0; i<pop_size; i++)
		{
			if(_error_trend[i]!=error_trend_E.ET_UNKNOWN)
			{
				is_computed = true;
			}
		}
		return is_computed;
	}
	
	/**
	 * The function indicated whether the population is inactive.
	 * For an inactive population, no evolution is done.
	 * 
	 * @return "true" if population is inactive; "false" - otherwise
	 */
	public boolean isInactive()
	{
		return _is_inactive;
	}
	
	/**
	 * The function computes a variance of current errors in the population.
	 * Considered errors are errors that were computed in the last time step.
	 * 
	 * On multidimensional sequences, the returned variance is the largest variance among all dimensions.
	 * 
	 * @return: computed variance of errors
	 */
	public double computeErrVariance()
	{
		int i,j;
		double[] error_cur;//current error
		double[] variance;//variance for each dimension of the error vector
		double   variance_max;//output variable
		Number[][] errors;//array of errors to compute the variance
		                  //1D: number of outputs
		                  //2D: errors in the corresponding output
		
		errors = new Number[_num_out][_population.length];
		for(i=0; i<_population.length; i++)
		{
			error_cur = _error[i].getDeviationLast();
			for(j=0; j<_num_out; j++)
			{
				errors[j][i] = error_cur[j];
			}
		}
		variance = new double[_num_out];
		for(j=0; j<_num_out; j++)
		{
			variance[j] = MathStat.StatCommon.computeVariance(errors[j]);
		}
		
		//find the largest element of the variance vector
		variance_max = variance[0];
		for(j=1; j<_num_out; j++)
		{
			if(variance_max < variance[j])
			{
				variance_max = variance[j];
			}
		}
		
		return variance_max;
	}
	
	/**
	 * The function computes a variance of individuals in the population.
	 * The computed variance is a general variance over all elements of the individuals.
	 * In order to combine variances of different elements, each element is first normalized using a valid range of
	 * the corresponding element. 
	 * 
	 * @return: computed variance of individuals
	 */
	public double computeIndividVariance()
	{
		int i, j;
		double   variance;//output variable
		double[] variance_i;//variance of all elements
		Number[] norm_element;//array of normalized values of the current element
		
		variance   = 0;
		variance_i = new double[_len_individual];
		norm_element = new Number[_population.length];
		//go over elements
		for(i=0; i<_len_individual; i++)
		{
			for(j=0; j<_population.length; j++)
			{
				norm_element[j] = _population[j][i] / _lim_opt_max[i];
			}
			variance_i[i] = MathStat.StatCommon.computeVariance(norm_element);
			variance += variance_i[i];
		}
		variance /= _len_individual;
		
		return variance;
	}
	
	/**
	 * The function enables the sorting of an individual at the provided index.
	 * 
	 * @param idx: index of an individual
	 */
	public void enableSort(int idx)
	{
		_do_sort[idx] = true;
	}
	
	/**
	 * The function disables the sorting of an individual at the provided index.
	 * 
	 * @param idx: index of an individual
	 */
	public void disableSort(int idx)
	{
		_do_sort[idx] = false;
	}
	
	/**
	 * The function exchanges individuals at two specified indices.
	 * The function also exchanges data, that belong to these individuals, in all arrays.
	 * These data are, for example, module output of these individuals, their end individuals and so on.
	 * 
	 * @param idx_0: index of the 1st individual
	 * @param idx_1: index of the 2nd individual
	 */
	public void exchangeIndividuals(int idx_0, int idx_1)
	{
		boolean  do_sort_tmp;
		double[] sub_output_tmp;//temporary array to keep module outputs
		double[] violation_tmp;//temporary array to keep a statistics of violations
		double[] individual;//temporary array to keep an individual
		DiffEvolutionError error_tmp;//temporary variable to keep an error status
		error_trend_E error_trend_tmp;//temporary variable to keep a trend of a prediction error
		output_history_C history_tmp;//temporary object to keep an output history
		
		do_sort_tmp     = _do_sort[idx_0];
		_do_sort[idx_0] = _do_sort[idx_1];
		_do_sort[idx_1] = do_sort_tmp;
		
		error_tmp     = _error[idx_0];
		_error[idx_0] = _error[idx_1];
		_error[idx_1] = error_tmp;
		
		error_trend_tmp     = _error_trend[idx_0];
		_error_trend[idx_0] = _error_trend[idx_1];
		_error_trend[idx_1] = error_trend_tmp;
		
		sub_output_tmp     = _sub_output[idx_0];
		_sub_output[idx_0] = _sub_output[idx_1];
		_sub_output[idx_1] = sub_output_tmp;
		
		violation_tmp                 = _violation_sub_out_max[idx_0];
		_violation_sub_out_max[idx_0] = _violation_sub_out_max[idx_1];
		_violation_sub_out_max[idx_1] = violation_tmp;
		
		violation_tmp                 = _violation_sub_out_sum[idx_0];
		_violation_sub_out_sum[idx_0] = _violation_sub_out_sum[idx_1];
		_violation_sub_out_sum[idx_1] = violation_tmp;
		
		individual         = _population[idx_0];
		_population[idx_0] = _population[idx_1];
		_population[idx_1] = individual;
		
		individual             = _end_individual[idx_0];
		_end_individual[idx_0] = _end_individual[idx_1];
		_end_individual[idx_1] = individual;
		
		history_tmp                = _sub_output_history[idx_0];
		_sub_output_history[idx_0] = _sub_output_history[idx_1];
		_sub_output_history[idx_1] = history_tmp;
	}
	
	/**
	 * The function updates the largest magnitude of ESN outputs.
	 * 
	 * @param lim_esn_output: new limits for all output neurons of an mESN
	 */
	public void setMaxTargetMagnitude(interval_C[] lim_esn_output)
	{
		int i;
		double lower_lim_cur, upper_lim_cur;//current left and right extreme values
		double lower_lim_chk, upper_lim_chk;//left and right borders of a checked interval
		
		for(i=0; i<_lim_sub_output.length; i++)
		{
			lower_lim_cur = _extreme_esn_output[i].getLowerLimitAsDouble();
			lower_lim_chk = lim_esn_output[i].getLowerLimitAsDouble();
			//update the lowest extreme value
			if(lower_lim_cur > lower_lim_chk)
			{
				_extreme_esn_output[i].setLeftBorder(lower_lim_chk);
			}
			
			upper_lim_cur = _extreme_esn_output[i].getUpperLimitAsDouble();
			upper_lim_chk = lim_esn_output[i].getUpperLimitAsDouble();
			//update the highest extreme value
			if(upper_lim_cur < upper_lim_chk)
			{
				_extreme_esn_output[i].setRightBorder(upper_lim_chk);
			}
			
			//TODO: delete after check of an impact of max values on success of synchronization
			_extreme_esn_output[i].setLeftBorder(0.0);
			_extreme_esn_output[i].setRightBorder(1.0);
		}
	}

	/**
	 * The function updates a valid range of mESN outputs.
	 * 
	 * @param lim_esn_output: new limits for all output neurons of an mESN
	 */
	public void setRangeEsnOutput(interval_C[] lim_esn_output)
	{
		int i;
		
		for(i=0; i<_lim_esn_output.length; i++)
		{
			_lim_esn_output[i].copy(lim_esn_output[i]);
		}
	}

	/**
	 * The function checks whether responsibility of the module is smaller than a provided threshold at all output neurons.
	 * If responsibility is small then an indicator of low activity is set.
	 * 
	 * The function checks activity of the module only for the best individual in a population.
	 * In order to avoid a situation that the population is disabled directly after initialization because
	 * responsibilities were initialized with very small values, the error of the best individual must be also below
	 * the provided threshold.
	 * 
	 * @param threshold: activity threshold
	 * @return: "true" - contribution of the module is very small; "false" - otherwise
	 */
	public boolean checkForInactiveModule(double threshold)
	{
		int i;
		boolean is_small_responsibility;//indicator of small responsibility of this module
		
		//check the error of the best individual, first
		if(_error[_best_idx].isBetterThresh(threshold)==true)
		{
			is_small_responsibility = true;
			getResponsibilityByIndex(_best_idx, _tmp_responsibility);
			//stop evaluation if responsibility is large at least at one output
			for(i=0; i<_num_out && is_small_responsibility==true; i++)
			{ 
				if(_tmp_responsibility[i] > threshold)
				{
					is_small_responsibility = false;
				}
			}
		}
		else
		{
			is_small_responsibility = false;
		}
		
		return is_small_responsibility;
	}
	
	/**
	 * The function sets a probability of module suppression.
	 * If it is not requested to reduce the initial probability, it is set to the initial value.
	 * Otherwise, the current probability value is decremented by a value depending on the provided number of
	 * time steps. The decrement is chosen in away to get probability 0% at the last time step.
	 * 
	 * @param do_reduce: request to reduce the probability of suppression
	 * @param num_steps: number of time steps
	 */
	public void SetSuppressProbability(boolean do_reduce, int num_steps)
	{
		double decrement;
		
		if(do_reduce==false)
		{
			_suppress_probab = _suppress_probab_init;
		}
		else
		{
			decrement = _suppress_probab_init / num_steps;
			_suppress_probab -= decrement;
			
			//invalid negative value of probability can be obtained at the end because of rounding errors
			if(_suppress_probab < 0)
			{
				_suppress_probab = 0;
			}
		}
	}
}