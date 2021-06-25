package types;

import java.util.Vector;

public class seq_parameter_C
{
	private Vector<oscillator_param_C> _oscillator_param;//set of parameters of several oscillators
	
	/**
	 * This is a sub-class of the class "parameter_set_C".
	 * 
	 * @author danil koryakin
	 */
	private class parameter_C
	{
		private String _name;
		private String _value;
		
		/**
		 * This is a class constructor which extracts a name and a value of a single parameter from a provided string.
		 * 
		 * @param param_str: provided string
		 */
		public parameter_C(String param_str)
		{
			String[] param_extracted;//array of substrings of a name and a value of a single parameter
			
			param_extracted = param_str.split("=");
			//remove possible spaces from extracted information
			param_extracted[0] = param_extracted[0].trim();
			param_extracted[1] = param_extracted[1].trim();
			//assign information about a parameter
			_name  = param_extracted[0];
			_value = param_extracted[1];
		}
		
		/**
		 * This is a class constructor which creates a copy of the provided parameter.
		 * 
		 * @param param: provided parameter
		 */
		public parameter_C(parameter_C param)
		{
			_name  = param._name;
			_value = param._value;
		}
		
		/**
		 * The function compares names of a provided and the host parameters.
		 * 
		 * @param param: provided parameter information
		 * @return: "true" if names are equal; "false" - otherwise
		 */
		private boolean compareName(parameter_C param)
		{
			boolean equal;//indicator that compared names are equal

			equal = _name.equals(param._name);
			
			return equal;
		}
		
		/**
		 * The function compares values of a provided and the host parameters.
		 * 
		 * @param param: provided parameter information
		 * @return: "true" if values are equal; "false" - otherwise
		 */
		private boolean compareValue(parameter_C param)
		{
			boolean equal;//indicator that compared values are equal

			equal = _value.equals(param._value);

			return equal;
		}
	};
	
	/**
	 * This class defines a set of parameters of a single oscillator.
	 *  
	 * @author danil koryakin
	 */
	private class oscillator_param_C
	{
		private Vector<parameter_C> _parameter_set;
		
		/**
		 * This is a class constructor to extract all parameters of a single oscillator from a provided string.
		 * 
		 * @param param_str: provided string
		 */
		public oscillator_param_C(String param_str)
		{
			int i;
			int idx_char;//index of a searched character
			parameter_C parameter;//extracted information of a single parameter
			String[] param_extracted;//array of substrings each of them containing a name and value of single parameter
			
			_parameter_set = new Vector<parameter_C>(0, 1);
			
			//find a number of parameters in a provided string
			param_extracted = param_str.split(",");
			for(i=0; i<param_extracted.length; i++)
			{
				//remove possible leading "(" or tailing ")"
				idx_char = param_extracted[i].indexOf('(');
				if(idx_char!=-1)
				{
					param_extracted[i] = param_extracted[i].substring(idx_char+1, param_extracted[i].length());
				}
				idx_char = param_extracted[i].indexOf(')');
				if(idx_char!=-1)
				{
					param_extracted[i] = param_extracted[i].substring(0, idx_char);
				}
				parameter = new parameter_C(param_extracted[i]);
				
				_parameter_set.add(parameter);
			}
		}
		
		/**
		 * This is a class constructor to create a copy of a provided object.
		 * 
		 * @param osci_param
		 */
		public oscillator_param_C(oscillator_param_C osci_param)
		{
			int i;
			parameter_C parameter;//extracted information of a single parameter
			
			_parameter_set = new Vector<parameter_C>(0, 1);
			
			for(i=0; i<osci_param._parameter_set.size(); i++)
			{
				parameter = new parameter_C(osci_param._parameter_set.get(i));
				_parameter_set.add(parameter);
			}
		}
		
		/**
		 * The function compares parameters of a provided oscillator to parameters of the host oscillator.
		 * 
		 * @param param: parameters of a provided oscillator
		 * @return: "true" if oscillators are equal, "false" - otherwise
		 */
		public boolean compareOscillators(oscillator_param_C param)
		{
			int i,j;
			boolean is_equal;//output variable
			boolean[] found_param;//indicator array to mark already found parameters
			parameter_C host_param;//currently compared host parameter
			parameter_C provided_param;//currently compared provided parameter
			
			if(_parameter_set.size()!=param._parameter_set.size())
			{
				is_equal = false;
			}
			else
			{
				//initialize the indicator array
				found_param = new boolean[_parameter_set.size()];
				for(i=0; i<_parameter_set.size(); i++)
				{
					found_param[i] = false;
				}
				
				for(i=0; i<_parameter_set.size(); i++)
				{
					host_param = _parameter_set.get(i);
					for(j=0; j<param._parameter_set.size() && found_param[i]==false; j++)
					{
						provided_param = param._parameter_set.get(j);
						if(host_param.compareName (provided_param)==true &&
						   host_param.compareValue(provided_param)==true)
						{
							found_param[i] = true;
						}
					}
				}
				
				//all parameters of compared oscillators must be equal
				is_equal = true;
				for(i=0; i<_parameter_set.size() && is_equal==true; i++)
				{
					if(found_param[i]==false)
					{
						is_equal = false;
					}
				}
			}
			
			return is_equal;
		}
	}
	
	/**
	 * This is a simple class constructor.
	 */
	public seq_parameter_C()
	{
		_oscillator_param = new Vector<oscillator_param_C>(0, 1);
	}
	
	/**
	 * This is a class constructor to create a description of a sequence as a set of parameters of all oscillators
	 * that constitute this sequence.
	 * Parameters of those oscillators shall be provided by a input array of strings where each string contains
	 * parameters of a single oscillator.
	 * 
	 * @param param_str: provided array of strings with parameters of oscillators
	 */
	public seq_parameter_C(Vector<String> param_str)
	{
		int i;
		oscillator_param_C param;
		
		_oscillator_param = new Vector<oscillator_param_C>(0, 1);
		
		for(i=0; i<param_str.size(); i++)
		{
			param = new oscillator_param_C(param_str.get(i));
			_oscillator_param.add(param);
		}
	}
	
	/**
	 * The function searches through a provided set of oscillators for an oscillator which is equal
	 * to a host oscillator with a provided index.
	 * A provided set of oscillators is restricted by an indicator array. This array shows which oscillators are
	 * relevant for the search. They are marked by "true". The other elements are "false" and are not considered during
	 * the search.
	 * If an equal oscillator is found, the function marks it with "false" in the provided indicator array.
	 * 
	 * @param idx_osci: index of a host oscillator
	 * @param param: provided set of oscillators
	 * @param is_relevant: indicator array
	 * @return "true" if an oscillator was found; "false" - otherwise
	 */
	private boolean doesEqualOscillatorExist(int idx_osci, seq_parameter_C param, boolean[] is_relevant)
	{
		int i;
		boolean exist;//indicator that an equal oscillator exists
		oscillator_param_C host_osci;//parameters of a host oscillator
		oscillator_param_C provided_osci;//parameters of a provided oscillator
		
		exist = false;
		host_osci = _oscillator_param.get(idx_osci);
		for(i=0; i<param._oscillator_param.size() && exist==false; i++)
		{
			provided_osci = param._oscillator_param.get(i);
			exist = host_osci.compareOscillators(provided_osci);
			if(exist==true)
			{
				is_relevant[i] = false;
			}
		}
		
		return exist;
	}
	
	/**
	 * This function completes a set of parameters with parameters of additional oscillator.
	 * Its parameters are contained in a provided string
	 * 
	 * @param param_str: string of parameters of an additional oscillator
	 */
	public void addOscillatorParam(String param_str)
	{
		oscillator_param_C param;
		
		param = new oscillator_param_C(param_str);
		_oscillator_param.add(param);
	}
	
	/**
	 * The function completes a set of parameters with parameters of oscillators from a provided set.
	 */
	public void addOscillatorParam(seq_parameter_C seq_param)
	{
		int i;
		oscillator_param_C osci_param;
		
		for(i=0; i<seq_param._oscillator_param.size(); i++)
		{
			osci_param = new oscillator_param_C( seq_param._oscillator_param.get(i) );
			_oscillator_param.add(osci_param);
		}
	}
	
	/**
	 * The function compares a provided set of oscillators to the host set of oscillators.
	 * 
	 * @param param: provided set of parameters
	 * @return: "true" if compared parameter sets are equal, "false" - otherwise
	 */
	public boolean compareSeqParameter(seq_parameter_C param)
	{
		int i;
		boolean is_equal;//output variable
		boolean[] is_relevant;//indicator array to show which provided oscillators are relevant for the search
		
		if(_oscillator_param.size()!=param._oscillator_param.size())
		{
			is_equal = false;
		}
		else
		{
			//initialize the indicator array
			is_relevant = new boolean[param._oscillator_param.size()];
			for(i=0; i<param._oscillator_param.size(); i++)
			{
				is_relevant[i] = true;
			}
			
			is_equal = true;
			for(i=0; i<_oscillator_param.size() && is_equal==true; i++)
			{
				//sequence parameters are not equal if equal oscillator was not found for at least one host oscillator
				is_equal = doesEqualOscillatorExist(i, param, is_relevant);
			}
		}
		
		return is_equal;
	}
}
