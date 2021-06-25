package esn;

import java.util.Random;
import java.util.Vector;

public class Activation {
	/**
	 * This enumeration implements activation functions of neurons.
	 * @author Danil Koryakin
	 */
	public enum activation_E {

			TANH,//hyperbolic tangent
			TANH_IN,//inverse hyperbolic tangent
			ID,//identity activation y(x)=x
			ID125,//linear activation y(x)=0.125*x
			ID05,//linear activation y(x)=0.5*x
			ID2,//linear activation y(x)=2*x
			LIM_ID,//limited identity activation with y(-1<x<1)=x, y(x<-1)=-1 and y(x>1)=1
			LIM_ID125,//limited linear activation with y(-8<x<8)=0.125*x, y(x<-8)=-1 and y(x>8)=1
			LIM_ID05,//limited linear activation with y(-2<x<2)=0.5*x, y(x<-2)=-1 and y(x>2)=1
			LOGISTIC,//logistic function
			NONE;//no activation is defined
			
			/**
			 * This method is used to obtain an instance of this enumeration from string
			 * values. This is necessary to read settings from files.
			 * @param value, a string extracted from a file.
			 * @return match, an instance of this enumeration according to the input string.
			 */
			public static activation_E fromString(String value)
			{
				activation_E match = null;
				
				for(activation_E func : activation_E.values())
				{
					if (value.equals(func.name())) {
						match = func;
					}
				}
				if(match==null)
				{
					System.err.println("fromString: invalid string to convert into enum functions");
					System.exit(1);
				}
				
				return match;
			}
			
			/**
			 * The function converts the provided array of strings into an array of activation functions.
			 * Each string of the provided array should contain a single activation function.
			 *
			 * @param str_functions: array of strings with activation functions
			 * @return: extracted array of activation functions
			 */
			public static activation_E[] fromStringArray(Vector<String> str_functions)
			{
				int i;
				activation_E[] func_val;
				
				func_val = new activation_E[str_functions.size()];
				
				for(i=0; i<str_functions.size(); i++)
				{
					func_val[i] = fromString(str_functions.get(i));
				}
				
				return func_val;
			}
			
			/**
			 * The function indicates an inverse activation function for a given activation function.
			 * 
			 * @param func: given activation
			 * @return: inverse activation function
			 */
			public static activation_E getInverseFunction(activation_E func)
			{
				activation_E inverse_funcion;//output variable
				
				switch (func)
				{
					case TANH:
						inverse_funcion = TANH_IN;
						break;
					case TANH_IN:
						inverse_funcion = TANH;
						break;
					case ID:
						inverse_funcion = ID;
						break;
					case ID125:
						inverse_funcion = NONE;
						break;
					case ID05:
						inverse_funcion = NONE;
						break;
					case ID2:
						inverse_funcion = NONE;
						break;
					case LIM_ID:
						inverse_funcion = NONE;
						break;
					case LIM_ID125:
						inverse_funcion = NONE;
						break;
					case LIM_ID05:
						inverse_funcion = NONE;
						break;
					case LOGISTIC:
						inverse_funcion = NONE;
						break;
					default:
						System.err.println("TransferFunctions.getInverseFunction: no inverse function is available");
						System.exit(1);
						inverse_funcion = NONE;
						break;
				}
				
				return inverse_funcion;
			}
			
			/**
			 * get a neuron activation function at random
			 * @return, name of the generated activation function
			 */
			public static activation_E getRandom(int seedTransfer)
			{
				Random rand = new Random(seedTransfer);
				int idx_func = rand.nextInt(2);//it can be chosen between two activation functions: "0"<=>TANH und "1"<=>ID
				activation_E output;//selected activation function
				
				switch(idx_func)
				{
				case 0:
					output = TANH;
					break;
				case 1:
					output = ID;
					break;
				default:
					System.err.println("getRandom: cannot generate an activation");
					System.exit(1);
					output = NONE;
				}
			
				return output;
			}
	}
	
	private activation_E _activation;
	private double       _param_logistic;//coefficient of the logistic function
	
	/**
	 * This is a constructor of the class.
	 */
	public Activation()
	{
		_activation     = activation_E.NONE;
		_param_logistic = Double.NaN;
	}
	
	/**
	 * Returns the hyperbolic tangent of value.
	 * @param value, the input value.
	 * @return double, the hyperbolic tangent of value.
	 */
	private double tanh(double value) {
		return Math.tanh(value);
	}
	
	/**
	 * Returns the inverse hyperbolic tangent of value.
	 * @param value, the input value.
	 * @return double, the inverse hyperbolic tangent of value.
	 */
	private double atanh (double value) {
		return 0.5D * Math.log((1.0D + (value)) / (1.0D - (value)));
	}
	
	/**
	 * The function calculates a value of activation function using a provided argument.
	 * 
	 * @param v: argument of activation function
	 * @return: computed activation function
	 */
	public double calculateValue(double v)
	{
		double returnValue = 0.0;
		
		switch (_activation)
		{
			case TANH:
				returnValue = this.tanh(v);
				break;
			case TANH_IN:
				returnValue = this.atanh(v);
				break;
			case ID:
				returnValue = v;
				break;
			case ID125:
				returnValue = 0.125*v;
				break;
			case ID05:
				returnValue = 0.5*v;
				break;
			case ID2:
				returnValue = 2*v;
				break;
			case LIM_ID:
				if(v < -1)
				{
					returnValue = -1;
				}
				else if(v > 1)
				{
					returnValue = 1;
				}
				else
				{
					returnValue = v;
				}
				break;
			case LIM_ID125:
				if(v < -8)
				{
					returnValue = -1;
				}
				else if(v > 8)
				{
					returnValue = 1;
				}
				else
				{
					returnValue = 0.125*v;
				}
				break;
			case LIM_ID05:
				if(v < -2)
				{
					returnValue = -1;
				}
				else if(v > 2)
				{
					returnValue = 1;
				}
				else
				{
					returnValue = 0.5*v;
				}
				break;
			case LOGISTIC:
				returnValue = (-_param_logistic)*v;
				returnValue = Math.exp(returnValue);
				returnValue = returnValue + 1;
				returnValue = 1/returnValue;
				returnValue = returnValue - 0.5;//center to y=0
				returnValue = 2*returnValue;//scale from [-0.5,+0.5] to [-1,+1]
				break;
			default:
				System.err.println("calculateValue: unknown activation");
				System.exit(1);
				returnValue = Double.NaN;
		}
		
		return returnValue;
	}
	
	/**
	 * The function sets a provided value as the parameter of the logistic activation function.
	 * The function issues an error message if the current activation is not the logistic function.
	 * 
	 * @param param: provided parameter value
	 */
	public void setLogisticParam(double param)
	{
		if(_activation==activation_E.LOGISTIC)
		{
			_param_logistic = param;
		}
		else
		{
			System.err.println("Actvation.setLogisticParam: current activation is not a logistic function");
			System.exit(1);
		}
	}
	
	/**
	 * The function indicates a value of the parameter of the logistic activation function.
	 * The function issues an error message if the current activation is not the logistic function.
	 * 
	 * @param return: parameter of the logistic function
	 */
	public double getLogisticParam()
	{
		double param;//output value
		
		if(_activation==activation_E.LOGISTIC)
		{
			param = _param_logistic;
		}
		else
		{
			System.err.println("Actvation.getLogisticParam: current activation is not a logistic function");
			System.exit(1);
			param = Double.NaN;
		}
		
		return param;
	}
	
	/**
	 * The function compared a provided activation function to an activation of the host object.
	 * 
	 * @param activation_given: given activation fnction
	 * @return "true" if the activation functions are equal, "false" - otherwise
	 */
	public boolean equals(activation_E activation_given)
	{
		boolean is_equal;//output variable
		
		if(_activation==activation_given)
		{
			is_equal = true;
		}
		else
		{
			is_equal = false;
		}
		return is_equal;
	}
	
	/**
	 * The function assigns a provided value as a name of the activation function.
	 * 
	 * @param activation_given: provided value
	 */
	public void setActivation(activation_E activation_given)
	{
		_activation = activation_given;
	}
	
	/**
	 * The function indicates a name of the activation function.
	 * 
	 * @return: name of the activation function
	 */
	public activation_E getActivation()
	{
		return _activation;
	}
	
	/**
	 * The function indicates a name of the inverse activation function of the host object.
	 * 
	 * @return: name of the activation function
	 */
	public activation_E getInverseActivation()
	{
		return activation_E.getInverseFunction(_activation);
	}
}
