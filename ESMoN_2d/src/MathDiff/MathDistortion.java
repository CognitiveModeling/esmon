package MathDiff;

import java.util.Random;

/**
 * This class provides methods for simulation of different kinds of disturbances.
 * 
 * @author Danil Koryakin
 */
public class MathDistortion
{
	/**
	 * This enumeration provides all available types of disturbances to be simulated.
	 */
	public enum distortion_E
	{
		DISTORT_CONST,//disturbance with a constant value
		DISTORT_GAUSS,//disturbance with a normally distributed value
		DISTORT_GAUSS_EXP,//disturbance with a normally distributed value whose sigma decreases exponentially
		DISTORT_NONE;//value to specify the case when no disturbance must be applied
		
		/**
		 * The function converts the provided string into an instance of the returned enumeration type. The function
		 * tries to match the name of each enumeration instance with the provided string.
		 * 
		 * @param str: provided string
		 * @return instance of the enumeration type, if the match was found; "null", otherwise
		 */
		public static distortion_E fromString(String str)
		{
			String   enum_prefix;
			String   tmp_str;
			String[] tmp_array;
			distortion_E match = null;
			
			//extract a prefix of the enumeration
			tmp_str = distortion_E.values()[0].toString();//convert a value to string
			tmp_array = tmp_str.split("_");//split at "_"
			enum_prefix = tmp_array[0];
			
			//add a prefix to the input string
			str = enum_prefix + "_" + str;
			
			//go over all values of the enumeration 
			for(distortion_E value_i : distortion_E.values())
			{
				if(str.equals(value_i.name()))
				{
					match = value_i;
				}
			}
			
			return match;
		}
	}
	
	distortion_E _type;//type of the simulated disturbance
	double[] _strength;//value for simulation of regular disturbance with a constant value
	double[] _exp_coef;//degree coefficient of the exponential decay of the variance in the Gaussian distribution
	double[] _gauss_sigma;//variance of the Gaussian distribution
	int[]  _period;//period to apply a regular disturbance
	Random _rand;//generator of random numbers
	
	/**
	 * This is a class constructor to create a dummy object for simulating zero disturbances.
	 * The constructor is provided a disturbance type to make an additional check that the dummy object is created when
	 * no disturbances should be simulated. If the selected disturbance type is NOT NONE then an error message will be
	 * issued.
	 * 
	 * @param type: provided disturbance type
	 * @param num_out: number of the output neurons
	 */
	public MathDistortion(distortion_E type, int num_out)
	{
		int i;
		
		if(type!=distortion_E.DISTORT_NONE)
		{
			System.err.println("MathDisturbance: specified disturbance type requires a call to another constructor");
			System.exit(1);
		}
		_strength    = new double[num_out];
		_period      = new int[num_out];
		_exp_coef    = new double[num_out];
		_gauss_sigma = new double[num_out];
		_rand        = null;
		
		_type = type;
		for(i=0; i<num_out; i++)
		{
			_strength   [i] = Double.NaN;
			_period     [i] = Integer.MAX_VALUE;
			_exp_coef   [i] = Double.NaN;
			_gauss_sigma[i] = Double.NaN;
		}
	}
	
	/**
	 * This is a class constructor to create an object for simulation of disturbances
	 * of the type "DISTURB_CONST". This must also be the provided type. Otherwise, the function will issue
	 * an error.
	 * 
	 * @param type: value corresponding to the with a constant value
	 * @param strength: array of values to be used for the disturbance
	 * @param period: array of periods to apply the disturbance
	 */
	public MathDistortion(distortion_E type, double[] strength, int[] period)
	{
		int i;
		int num_elements;//number of element in the array of disturbance
		
		num_elements = strength.length;
		if(type!=distortion_E.DISTORT_CONST)
		{
			System.err.println("MathDisturbance: parameters do not match the chosen disturbance type");
			System.exit(1);
		}
		for(i=0; i<num_elements; i++)
		{
			if(period[i] <= 0)
			{
				System.err.println("MathDisturbance: provided period must be a positive number");
				System.exit(1);
			}
		}
		
		_strength    = new double[num_elements];
		_period      = new int   [num_elements];
		_exp_coef    = new double[num_elements];
		_gauss_sigma = new double[num_elements];
		_rand        = null;
		
		_type        = type;
		for(i=0; i<num_elements; i++)
		{
			_strength   [i] = strength[i];
			_period     [i] = period[i];
			_exp_coef   [i] = Double.NaN;
			_gauss_sigma[i] = Double.NaN;
		}
	}
	
	/**
	 * This is a class constructor to create an object for simulation of disturbances
	 * of the type "DISTURB_GAUSS_EXP". This must also be the provided type. Otherwise, the function will issue
	 * an error.
	 * For the Gaussian distribution, the initial variance is provided. The math expectation of the Gaussian
	 * distribution is always 0. The exponential decay is defined by the provided degree coefficient.
	 * 
	 * @param type: value corresponding to a disturbance with a normally distributed value whose sigma decreases
	 *              exponentially
	 * @param num_out: number of the output neurons
	 * @param sigma: initial variance of the Gaussian distribution
	 * @param coef: degree coefficient for the exponential decay
	 * @param period: period to compute the disturbance
	 * @param seed: value to be used for seeding the generator of random numbers
	 */
	public MathDistortion(distortion_E type, int num_out, double sigma, double coef, int period, int seed)
	{
		int i;
		
		if(type!=distortion_E.DISTORT_GAUSS_EXP)
		{
			System.err.println("MathDisturbance: parameters do not match the chosen disturbance type");
			System.exit(1);
		}
		if(period <= 0)
		{
			System.err.println("MathDisturbance: provided period must be a positive number");
			System.exit(1);
		}
		_strength    = new double[num_out];
		_period      = new int[num_out];
		_exp_coef    = new double[num_out];
		_gauss_sigma = new double[num_out];
		_rand        = new Random(seed);
		
		_type        = type;
		for(i=0; i<num_out; i++)
		{
			_strength   [i] = Double.NaN;
			_period     [i] = period;
			_exp_coef   [i] = coef;
			_gauss_sigma[i] = sigma;
		}
	}
	
	/**
	 * This is a class constructor to create an object for simulation of disturbances
	 * of the type "DISTURB_GAUSS_EXP". This must also be the provided type. Otherwise, the function will issue
	 * an error.
	 * For the Gaussian distribution, the initial variance is provided. The math expectation of the Gaussian
	 * distribution is always 0. The exponential decay is defined by the required variance at the end of
	 * the configuration sequence.
	 * 
	 * @param type: value corresponding to a disturbance with a normally distributed value whose sigma decreases
	 *              exponentially
	 * @param sigma_0: array with initial variances of the Gaussian distributions
	 * @param sigma_i: array with variances at the provided time step
	 * @param seq_len: length of the configuration sequence to compute a coefficient of the exponential decay
	 * @param period: array with periods to apply the disturbance
	 * @param seed: value to be used for seeding the generator of random numbers
	 */
	public MathDistortion(distortion_E type, double[] sigma_0, double[] sigma_i, int seq_len, int[] period, int seed)
	{
		int i;
		int num_elements;
		
		num_elements = sigma_0.length;
		if(type!=distortion_E.DISTORT_GAUSS_EXP)
		{
			System.err.println("MathDisturbance: parameters do not match the chosen disturbance type");
			System.exit(1);
		}
		for(i=0; i<num_elements; i++)
		{
			if(period[i] <= 0)
			{
				System.err.println("MathDisturbance: provided period must be a positive number");
				System.exit(1);
			}
		}
		_strength    = new double[num_elements];
		_period      = new int   [num_elements];
		_exp_coef    = new double[num_elements];
		_gauss_sigma = new double[num_elements];
	    _rand        = new Random(seed);
		
		_type        = type;
		for(i=0; i<num_elements; i++)
		{
			_strength   [i] = Double.NaN;
			_period     [i] = period[i];
			_gauss_sigma[i] = sigma_0[i];
			//compute the degree coefficient from Sigma(time_step) = exp(-a * seq_len)
			//                        it leads to a = ln( Sigma(time_step) ) / (-seq_len)
			_exp_coef[i] = Math.log(sigma_i[i]) / (- seq_len);
		}
	}
	
	/**
	 * The function computes a disturbance value for the chosen disturbance type at the provided time step. The type of
	 * the disturbance has been chosen at the construction of the disturbance object. The provided time step is a time
	 * step which is related to the beginning of the configuration sequence.
	 * 
	 * @param time_step: provided time step
	 * @return: value of each element of the disturbance vector
	 */
	public double[] computeDisturbance(int time_step)
	{
		int i;
		int div_rest;//rest of the division
		int num_period;//integer number of periods
		int num_disturb;//number of elements of the disturbance vector
		double sigma;//variance at the provided time step
		double[] disturb_val;//simulated values of the disturbance
		
		num_disturb = _period.length;
		
		//allocate the output array
		disturb_val = new double[num_disturb];
		
		//go over all elements of the disturbance vector
		for(i=0; i<num_disturb; i++)
		{
			//non-zero disturbance value is computed only, if the provided time step starts a new period
			num_period = time_step/_period[i];
			div_rest = time_step - (num_period * _period[i]);
			if(div_rest!=0)
			{
				disturb_val[i] = 0;
			}
			else
			{
				switch(_type)
				{
					case DISTORT_CONST:
						disturb_val[i] = _strength[i];
						break;
					case DISTORT_GAUSS:
						//math expectation is always "0"
						disturb_val[i] = _rand.nextGaussian() * _gauss_sigma[i] + 0.0;
						break;
					case DISTORT_GAUSS_EXP:
						//compute a variance as "sigma=sigma0 * EXP(-a * t)"
						sigma = _gauss_sigma[i] * Math.exp(-_exp_coef[i] * time_step);
						//math expectation is always "0"
						disturb_val[i] = _rand.nextGaussian() * sigma + 0.0;
						break;
					case DISTORT_NONE:
						disturb_val[i] = 0;
						break;
					default:
						disturb_val[i] = 0;
						System.err.println("computeDisturbance: unknown disturbance type");
						System.exit(1);
						break;
				}//switch
			}
		}
		
		return disturb_val;
	}
}