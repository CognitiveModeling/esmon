package types;

import java.util.StringTokenizer;
import java.util.Vector;

public class vector_C extends multi_val_C
{
	public vector_C(Number a)
	{
		super();
		
		storeValue(a);
	}
	
	public vector_C(Vector<Object> vector)
	{
		super();
		
		int i;
		Object obj;
		
		for(i=0; i<vector.size(); i++)
		{
			obj = new Object();
			obj = vector.get(i);
			
			_vector.add(obj);
		}
	}
	
	public vector_C(String str)
	{
		super();
		
		String  singleValStr;//string containing a single value
		Double  singleValDoub;//extracted double value
		Integer singleValInt;//extracted integer value
		StringTokenizer st = new StringTokenizer(str, "{,}");
		
		while(st.hasMoreTokens())
		{
			singleValStr  = st.nextToken();
			try
			{
				if(singleValStr.contains("."))//it should be double, if the string contains "."
				{
					singleValDoub = Double.valueOf(singleValStr);
					_vector.add(singleValDoub);
				}
				else
				{
					singleValInt = Integer.valueOf(singleValStr);
					_vector.add(singleValInt);
				}
			}
			catch(NumberFormatException e)
			{
				_vector.add(singleValStr);
			}								
		}//store values
		
		if(st.hasMoreTokens())
		{
			System.err.println("interval_C: invalid string to convert into interval");
			System.exit(1);
		}
	}
	
	/**
	 * obtain the vector as an array of doubles
	 * @return, array of doubles
	 */
	public double[] getArrayDouble()
	{
		int i;
		double[] out;//output variable
		
		//check, if there are data in the vector
		if(_vector.isEmpty()==true)
		{
			out = null;
		}
		else
		{
			//check the class of array elements
			if(_vector.get(0).getClass()!=Double.class)
			{
				System.err.println("getArray: vector does not contain floating values");
				System.exit(1);
			}
			out = new double[_vector.size()];
			for(i=0; i<_vector.size(); i++)
			{
				out[i] = (Double)_vector.get(i);
			}
		}
		return out;
	}
	
	/**
	 * obtain the vector as an array of integers
	 * @return, array of integers
	 */
	public int[] getArrayInt()
	{
		int i;
		int[] out;//output variable
		
		//check, if there are data in the vector
		if(_vector.isEmpty()==true)
		{
			out = null;
		}
		else
		{
			//check the class of array elements
			if(_vector.get(0).getClass()!=Integer.class)
			{
				System.err.println("getArray: vector does not contain integer values");
				System.exit(1);
			}
			out = new int[_vector.size()];
			for(i=0; i<_vector.size(); i++)
			{
				out[i] = (Integer)_vector.get(i);
			}
		}
		return out;
	}
	
	/**
	 * The function returns the vector as an array of Booleans.
	 * 
	 * @return: array of booleans
	 */
	public boolean[] getArrayBoolean()
	{
		int i;
		boolean[] out;//output variable
		
		//check, if there are data in the vector
		if(_vector.isEmpty()==true)
		{
			out = null;
		}
		else
		{
			out = new boolean[_vector.size()];
			for(i=0; i<_vector.size(); i++)
			{
				out[i] = Boolean.valueOf( _vector.get(i).toString() );
			}
		}
		return out;
	}
	
	/**
	 * obtain the vector as an array of strings
	 * @return, array of strings
	 */
	public String[] getArrayStr()
	{
		int i;
		String[] out;//output variable
		
		//check, if there are data in the vector
		if(_vector.isEmpty()==true)
		{
			out = null;
		}
		else
		{
			out = new String[_vector.size()];
			for(i=0; i<_vector.size(); i++)
			{
				out[i] = (String)_vector.get(i);
			}
		}
		return out;
	}
	
	/**
	 * The function returns an element by the provided index.
	 * @param idx: index of the required element
	 * @return retrieved element of the vector
	 */
	public Object getElement(int idx)
	{
		Object element;//output variable
		
		if(idx < _vector.size())
		{
			element = _vector.get(idx);
		}
		else
		{
			element = null;
			System.err.println("getElement: index is out of bounds");
			System.exit(1);
		}
		
		return element;
	}
	
	/**
	 * The function returns a required element as a character, if this element is a string according to its class
	 * and this string has length one. Otherwise, the function issues an error message and stops the program.
	 * 
	 * @param idx: index of the required element
	 * @return element as a character
	 */
	public Character getElementAsChar(int idx)
	{
		Object    element_obj;//required elements as an object
		String    element_str;//required elements as a string
		Character element;//output value
		
		if(idx < _vector.size())
		{
			element_obj = _vector.get(idx);
			if(element_obj.getClass()==String.class)
			{
				element_str = (String)element_obj;
				if(element_str.length()==1)
				{
					element = element_str.charAt(0);
				}
				else
				{
					element = null;
				}
			}
			else
			{
				element = null; 
			}
		}
		else
		{
			element = null;
			System.err.println("getElementAsChar: index is out of bounds");
			System.exit(1);
		}
		
		return element;
	}
	
	/**
	 * The function returns a required element as a floating-point value, if the class of this element is Double.
	 * Otherwise, the function issues an error message and stops the program.
	 * 
	 * @param idx: index of the required element
	 * @return element as a floating-point value
	 */
	public Double getElementAsDouble(int idx)
	{
		Object  element_obj;//required elements as an object
		Double element;//output value
		
		if(idx < _vector.size())
		{
			element_obj = _vector.get(idx);
			if(element_obj.getClass()==Double.class)
			{
				element = (Double)element_obj;
			}
			else
			{
				element = null; 
			}
		}
		else
		{
			element = null;
			System.err.println("getElementAsDouble: index is out of bounds");
			System.exit(1);
		}
		
		return element;
	}
	
	/**
	 * The function returns a required element as an integer, if the class of this element is Integer.
	 * Otherwise, the function issues an error message and stops the program.
	 * 
	 * @param idx: index of the required element
	 * @return value of the required Integer element
	 */
	public Integer getElementAsInt(int idx)
	{
		Object  element_obj;//required elements as an object
		Integer element;//output value
		
		if(idx < _vector.size())
		{
			element_obj = _vector.get(idx);
			if(element_obj.getClass()==Integer.class)
			{
				element = (Integer)element_obj;
			}
			else
			{
				element = null; 
			}
		}
		else
		{
			element = null;
			System.err.println("getElementAsInt: index is out of bounds");
			System.exit(1);
		}
		
		return element;
	}
	
	/**
	 * The function returns a required element as a boolean value, if the class of this element is Boolean.
	 * Otherwise, the function issues an error message and stops the program.
	 * 
	 * @param idx: index of the required element
	 * @return value of the required Boolean element
	 */
	public Boolean getElementAsBoolean(int idx)
	{
		Object  element_obj;//required elements as an object
		String  element_str;//required elements as a string
		Boolean element;//output value
		
		if(idx < _vector.size())
		{
			element_obj = _vector.get(idx);
			element_str = (String)element_obj;
			element     = Boolean.valueOf(element_str);
		}
		else
		{
			element = null;
			System.err.println("getElementAsBoolean: index is out of bounds");
			System.exit(1);
		}
		
		return element;
	}
	
	/**
	 * return a number of elements in the vector
	 * @return, number of elements in the vector
	 */
	public int getSize()
	{
		return _vector.size();
	}
	
	/**
	 * append the provided element to the vector
	 * @param val, provided element to be appended to the vector
	 */
	public void storeValue(Object val)
	{
		_vector.add(val);
	}
	
	/**
	 * convert a set object to string
	 */
	public String toString()
	{
		int i;
		String str_out;
		
		str_out = "{";
		for(i=0; i<_vector.size()-1; i++)
		{
			str_out+=_vector.get(i).toString();
			str_out+= ",";
		}
		str_out+=_vector.get(_vector.size()-1).toString();
		str_out+= "}";
		
		return str_out;
	}
}
