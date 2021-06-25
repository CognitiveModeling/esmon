package esn;

import java.util.Arrays;
import java.util.Random;
import java.util.Vector;

import types.interval_C;

import esn.Activation.activation_E;
import experiment.ExpParam.exp_param_E;

import Jama.Matrix;
import MathDiff.MathNoise;
import MathDiff.MathNoise.noise_E;

/**
 * Instances of this class represent the dynamic reservoir of an esn. Hence it possess
 * special attributes, for instance values for the connectivity or the spectral radius.
 * @author Johannes Lohmann, Danil Koryakin
 *
 */
public class InternalLayer extends Layer
{
	private double _connect;//square array of exclusive connectivity and connectivity in the overlaps
	private double _sr;//spectral radius
	public double _eigen_max;//largest eigenvalue of reservoir

	private ReservoirInitialization initializationPolicy = null;
	private noise_E _noise_type = null;
	private interval_C _noise_bounds = null;
	
	/**
	 * Constructs a new internal layer. It is not parameterized with a double array
	 * for the value-bounds of the weights of this layer, but they can be set if necessary.
	 * 
	 * @param res_size: number of reservoir neurons
	 * @param connect: connectivity of reservoir
	 * @param sr: spectral radius of reservoir
	 * @param seed: seeding value for random number generators
	 */
	public InternalLayer(int    res_size,
			             double connect,
			             double sr,
			             int    seed)
	{
		super(res_size, res_size, res_size, null);

		_bounds  = new double[] {-1.0, 1.0};//set default boundaries for initialization of internal weights
		_connect = connect;
		_sr      = sr;
		_weights = new Matrix(res_size, res_size);//all matrix's elements are initialized with zeros
		_weights_init = new Matrix(res_size, res_size);//all matrix's elements are initialized with zeros
		_noise_type  = MathNoise.noise_E.NOISE_NONE;
		_noise_bounds = new interval_C();
		_seed         = seed;
	}
	
	/**
	 * The function calculates a number of connections which shall have non-zero weights for a given connectivity and for
	 * a given total number of connections.
	 * 
	 * @return: number of connections with non-zero-weights
	 */
	private int computeNumConnectNonZero()
	{
		int    num_total;//total number of connections
		int    num_nonzero;//output variable (number of non-zero weights)
		double portion_nonzero;//portion of non-zero weights
		double decimal;//decimal part of the value to be rounded
		
		num_total = _weights_init.getRowDimension() * _weights_init.getColumnDimension(); 
		if(_connect > 0 && num_total > 0)
		{
			num_nonzero = (int)(_connect*num_total);
			if(num_nonzero==0)//since connectivity is larger than 1, there must be at least one non-zero connection
			{
				num_nonzero = 1;
			}
			else//make correct rounding
			{
				portion_nonzero = _connect*num_total;
				decimal = portion_nonzero - num_nonzero;
				if(decimal >= 0.5)
				{
					num_nonzero = num_nonzero + 1;
				}
				else
				{
					//keep the value computed above
				}
			}
		}
		else
		{
			num_nonzero = 0;
		}
		
		//computed number of non-zero connections shall be smaller or equal to a total number of connections
		if(num_nonzero > num_total)
		{
			System.err.println("getNumConnectNonZero: number of non-zeros is larger than total number of weights");
			System.exit(1);
		}
		
		return num_nonzero;
	}
	
	/**
	 * The function assigns random values to unscaled (initial) weights of reservoir connections according
	 * to a required connectivity. This supposes that some weights can stay "0".
	 * Random values are uniformly distributed on an specified interval.
	 */
	private void generateSubReservoirs()
	{
		int     i;
		int     idx_start;//index of neuron where the chosen connection starts
		int     idx_end;//index of neuron where the chosen connection ends
		int     num_nonzero;//number of non-zero weights to be assigned
		double  weight_connect;//current weight of chosen connection
		double[][] weights;//matrix of weights to be assigned
		Random rand;//random generator

		//prepare a generator of random numbers
		rand = new Random(_seed);//temporary a seeding value of 1st module is used for generation of the whole reservoir
		//it must be changed when each module is stored as a separate ESN
		//get an array of reservoir weights to be assigned
		weights = _weights_init.getArray();

		//get number of connections to be set
		num_nonzero = computeNumConnectNonZero();

		//go over weights and assign them random numbers
		for(i=0; i<num_nonzero; i++)
		{
			do{
				idx_start = rand.nextInt(_weights_init.getColumnDimension());
				idx_end   = rand.nextInt(_weights_init.getColumnDimension());
				weight_connect = weights[idx_start][idx_end];
			}while(weight_connect!=0);//chosen connection should have not been assigned yet
			//assign a random value to the weight
			weights[idx_start][idx_end] = _bounds[0] + Math.abs(_bounds[1] - _bounds[0]) * rand.nextDouble();
		}
	}
	
	/**
	 * This function generates a set of random initial states for reservoir neurons.
	 * The random values are uniformly distributed on the interval [-1, +1].
	 */
	public void generateInitState()
	{
		int    i,j;
		double rnd_value;//randomly generated value to be assigned as a value for neuron initialization
		Random rand;//random number generator
		
		rand = new Random(_seed);
		for(i=0; i<_init_state.getRowDimension(); i++)
		{
			for(j=0; j<_init_state.getColumnDimension(); j++)
			{
				rnd_value = 2.0 * rand.nextDouble() - 1.0;
				_init_state.set(i, j, rnd_value);
			}
		}
	}
	
	/**
	 * The function sets the states of all output neurons to their predefined initial values. The set of the initial
	 * values was defined in the constructor of the class or was provided from the outside.
	 */
	public void assignInitState()
	{
		_nodes.setMatrix(0, _nodes.getRowDimension()-1, 0, _nodes.getColumnDimension()-1,
				         _init_state);
	}
	
	/**
	 * The function indicates a number of neurons in the ESN module.
	 * 
	 * @return: number of neurons in the specified sub-reservoir
	 */
	public int getSize()
	{
		return _weights_init.getRowDimension();
	}
	
	/**
	 * The function randomly generates a constant bias for every reservoir neuron. The generated value is uniformly
	 * distributed over a specified interval. If the interval has equal left and right bounds then the bias of all
	 * neurons is set to the value given by the bound.
	 * @param res_bias, array with min and max values to generate a bias for every reservoir neuron 
	 */
	public void initializeBias(interval_C res_bias)
	{
		int    i;
		double bias_to_assign;//randomly generated bias value to be assigned
		double lower_lim, upper_lim;//left and right borders of an interval
		int    num_nodes;//number of reservoir neurons
		Random rand;//generator of random numbers
		
		rand = new Random(_seed);
		
		lower_lim = res_bias.getLowerLimitAsDouble();
		upper_lim = res_bias.getUpperLimitAsDouble();
		if(lower_lim == upper_lim)
		{
			//set the bias to the same value
			setBias(lower_lim, -1);
		}
		else
		{
			num_nodes = _nodes.getRowDimension();
			for(i=0; i<num_nodes; i++)
			{
				bias_to_assign = lower_lim + Math.abs(upper_lim - lower_lim) * rand.nextDouble();
				setBias(bias_to_assign, i);
			}
		}
	}
	
	/**
	 * There are different possibilities to define the dynamic reservoir. Mainly the random initialization or
	 * the enforcement of a specific topology. The way of initialization is chosen here, according to the enumeration
	 * constant initializationPolicy from the type {@link ReservoirInitialization}.
	 * Reservoir weights are generated only, if it is required.
	 */
	@Override
	public void initializeWeights()
	{
		if(initializationPolicy != null)
		{
			switch(initializationPolicy)
			{
				case RANDOM:
					generateSubReservoirs();
					break;
				case SCR:
					setWeightsInit(ReservoirInitialization.initializeSCR(_bounds, _seed, _rows, _cols));
					break;
				case DLR:
					setWeightsInit(ReservoirInitialization.initializeDLR(_bounds, _seed, _rows, _cols));
					break;
				case DLRB:
					setWeightsInit(ReservoirInitialization.initializeDLRB(_bounds, _seed, _rows, _cols));
					break;
				case SRR:
					setWeightsInit(ReservoirInitialization.initializeSRR(_bounds, _seed, _rows, _cols));
					break;
				default:
					System.err.println("InternalLayer.initializeWeights: unknown topology");
					System.exit(1);
					break;
			}
		}
		else
		{
			System.err.println("InternalLayer.initializeWeights: topology is not specified");
			System.exit(1);
		}
	}
	
	/**
	 * The function returns a spectral radius of the ESN module.
	 *  
	 * @return: array with spectral radius
	 */
	public double getSpectralRadius()
	{
		return _sr;
	}
	
	/**
	 * The function sets a provided value as a spectral radius for the ESN module.
	 * 
	 * @param sr: provided value of a spectral radius
	 */
	public void setSpectralRadius(double sr)
	{
		_sr = sr;
	}

	/**
	 * Returns the value of the connectivity.
	 * @return the connectivity.
	 */
	public double getConnectivity()
	{
		return _connect;
	}
	
	/**
	 * The function produces a connectivity matrix for current reservoir weights.
	 * 
	 * @return: connectivity matrix as an array of strings
	 */
	public Vector<String> getConnectivityMatrixAsStr()
	{
		int i, j;
		String str;
		Vector<String> connectivity;
		
		connectivity = new Vector<String>(0,1);
		for(i=0; i<_rows; i++)
		{
			str = "";
			for(j=0; j<_cols; j++)
			{
				if(_weights.get(i, j)!=0)
				{
					str += "1 ";
				}
				else
				{
					str += "0 ";
				}
			}
			connectivity.add(str);
		}
		return connectivity;
	}
	
	/**
	 * The function assigns a provided value to connectivity of the ESN module.
	 * 
	 * @param connect: provided connectivity
	 */
	public void setConnectivity(double connect)
	{
		_connect = connect;
	}

	/**
	 * @return the initializationPolicy
	 */
	public ReservoirInitialization getInitializationPolicy() {
		return initializationPolicy;
	}

	/**
	 * @param initializationPolicy the initializationPolicy to set
	 */
	public void setInitializationPolicy(ReservoirInitialization initializationPolicy)
	{
		this.initializationPolicy = initializationPolicy;
	}
	
	/**
	 * The function converts values of the specified parameter to an array of strings.
	 * The function terminates a program with an error, if no conversion is implemented for the specified parameter.
	 * 
	 * @param param: specified parameter whose values should be converted to an array of strings
	 * @return: values of the specified parameter as an array of strings
	 */
	public Vector<String> getParamValAsStr(exp_param_E param)
	{
		String str;
		Vector<String> param_val;
		
		param_val = new Vector<String>(0,1);
		
		//choose a parameter to be converted to an array of strings
		switch(param)
		{
			case EP_RES_SPECTR_RAD:
				str = "";
				str+= _sr;
				param_val.add(str);
				break;
			case EP_RES_CONNECT:
				str = "";
				str+= _connect;
				param_val.add(str);
				break;
			case EP_RES_TOPOLOGY:
				str = "";
				str+= initializationPolicy.toString();
				param_val.add(str);
				break;
			default:
				System.err.println("getParamValAsStr: conversion is not implemented for the specified parameter");
				System.exit(1);
				break;
		}

		return param_val;
	}
	
	/**
	 * The function returns the actual leakage rate of the reservoir neurons.
	 * @return: actual leakage rate of the reservoir neurons
	 */
	public double[] getLeakageRate()
	{
		return _leakage_rate;
	}
	
	/**
	 * The function assigns a leakage rate from the provided array to each reservoir neuron.
	 * @param leakage_rate: provided leakage rate
	 */
	public void setLeakageRate(double[] leakage_rate)
	{
		int i;
		
		if(_leakage_rate.length != leakage_rate.length)
		{
			System.err.println("setLeakageRate: mismatch of array size");
			System.exit(1);
		}
		
		for(i=0; i<leakage_rate.length; i++)
		{
			_leakage_rate[i] = leakage_rate[i];
		}
	}
	
	/**
	 * The function assigns a leakage rate according to a provided assignment rule using the provided parameter value.
	 * 
	 * @param leakage_assign: identifier of leakage assignment rule
	 * @param leakage_rate:   parameter value for leakage rate
	 */
	public void setLeakageRate(leakage_assign_E leakage_assign, double leakage_rate)
	{
		int    i;
		Random rand;
		
		switch(leakage_assign)
		{
			case LA_SAME:
				for(i=0; i<_leakage_rate.length; i++)
				{
					_leakage_rate[i] = leakage_rate;
				}
				break;
			case LA_RANDOM:
				rand = new Random(_seed);
				for(i=0; i<_leakage_rate.length; i++)
				{
					_leakage_rate[i] = leakage_rate*rand.nextDouble();
				}
				break;
			default:
				System.err.println("setLeakageRate: unknown assignment rule");
				System.exit(1);
				break;
		}
	}
	
	/**
	 * The function returns the noise type applied under computing the states of the reservoir neurons.
	 * @return: noise type
	 */
	public noise_E getNoiseType()
	{
		return _noise_type;
	}
	
	/**
	 * The function sets the provided noise type to be applied under computing the states of the reservoir neurons.
	 * @param noise_type: provided noise type
	 */
	public void setNoiseType(noise_E noise_type)
	{
		_noise_type = noise_type;
	}
	
	/**
	 * The function returns the bounds of the interval where the noise values are generated from.
	 * @return: interval of noise values
	 */
	public interval_C getNoiseBounds()
	{
		return _noise_bounds;
	}
	
	/**
	 * The function sets the provided bounds of the interval where the noise values are generated from.
	 * @param noise_bounds: provided interval of noise values
	 */
	public void setNoiseBounds(interval_C noise_bounds)
	{
		_noise_bounds.copy(noise_bounds);
	}
	
	/**
	 * The function assigns a randomly generated activation function to every reservoir neuron. The activation function
	 * can be one of two types provided as input parameters. The provided ratio determines an amount of
	 * the reservoir neurons with the activation of the 1st type. If the ratio is a value between 0.0 and 1.0 then
	 * the reservoir contains neurons with both types of activation functions.
	 * The function sets a provided value as a parameter of the logistic function if it is chosen
	 * as an activation function.
	 * 
	 * @param activation_type1: activation of the 1st type
	 * @param activation_type2: activation of the 2nd type
	 * @param activ_ratio: activation ratio
	 * @param activ_logistic_param: parameter of logistic function
	 */
	public void generateActivations(activation_E activation_type1, activation_E activation_type2, double activ_ratio, double activ_logistic_param)
	{
		int  i;
		int  idx_neuron;//index of neuron
		long num_activ;//number of neurons which should have activation of the 1st type
		Random rand;//random generator
		
		rand = new Random(_seed);
		 
		//initialize activation function of all neurons with activation different from the required one
		setActivation(activation_type2, -1);

		num_activ = Math.round(_rows*activ_ratio);
		
		//set required activation function of neurons at random
		for(i=0; i<num_activ; i++)
		{
			do{
				idx_neuron = rand.nextInt(_rows);
			//assign only activation functions which were not assigned before
			}while(_func[idx_neuron].equals(activation_type1));
			setActivation(activation_type1, idx_neuron);
			
			//set a parameter of the logistic function
			if(_func[idx_neuron].equals(activation_E.LOGISTIC))
			{
				_func[idx_neuron].setLogisticParam(activ_logistic_param);
			}
		}
	}
	
	/**
	 * The function assigns and scales reservoir weights.
	 * Scaling of weights is done only for ESN modules.
	 * 
	 * The procedure has different realizations in all classes the "InputLayer", "OutputLayer", "InternalLayer" and
	 * "BackLayer".
	 * 
	 * @param do_scaling: request to scale reservoir weights for an ESN module
	 */
	public void setActiveWeights(boolean do_scale)
	{ 
		Matrix module_weights;//weights of one module
		
		//extract weights of current module
		module_weights = _weights_init.copy();

		//scale weights of current module
		//(it is requested only for ESN modules)
		if(do_scale==true)
		{
			module_weights = module_weights.times(_sr);
			module_weights = module_weights.times(1 / _eigen_max);
		}

		//assign scaled weights of current module
		_weights.setMatrix(0, module_weights.getRowDimension()-1, 0, module_weights.getColumnDimension()-1, module_weights);
	}
	
	/**
	 * The function indicates whether the largest eigenvalue of a matrix of initial weights does not equal "0".
	 * 
	 * @return TRUE if the largest eigenvalue is not "0"; FALSE - otherwise.
	 */
	public boolean isMaxEigenvalueNotZero()
	{
		return (_eigen_max!=0);
	}
	
	/**
	 * The function computes the largest eigenvalue of a matrix of initial weights.
	 * The computed eigenvalue is assigned as a global variable.
	 *
	 * @return: obtained maximum eigenvalue
	 */
	public void computeMaxEigenvalue()
	{
		double[] eigenWerte;//array of eigenvalues
		
		//get the eigenvalues
		eigenWerte = _weights_init.eig().getRealEigenvalues();
		//sort the eigenvalues in ascending order to get the maximum of absolute as 1st or last element
		Arrays.sort(eigenWerte);
	
		//store an eigenvalue with the largest absolute value
		if(Math.abs(eigenWerte[0]) > Math.abs(eigenWerte[eigenWerte.length - 1]))
		{
			_eigen_max = Math.abs(eigenWerte[0]);
		}
		else
		{
			_eigen_max = Math.abs(eigenWerte[eigenWerte.length - 1]);
		}
	}
}
