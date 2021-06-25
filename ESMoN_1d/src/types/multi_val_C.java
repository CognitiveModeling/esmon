package types;

import java.util.Vector;

public class multi_val_C
{
	public Vector<Object> _vector;//array of values in the set
	
	public multi_val_C()
	{
		_vector = new Vector<Object>(0, 1);
	}
	public multi_val_C(Number a)
	{
		_vector = new Vector<Object>(0, 1);
		_vector.add(a);
	} 
	public multi_val_C(Number a, Number b)
	{
		interval_C interval;
		
		interval = new interval_C(a, b);
		_vector = new Vector<Object>(0, 1);
		_vector.add(interval);
	}
	public multi_val_C(String str)
	{
		_vector = new Vector<Object>(0, 1);
		_vector.add(str);
	}
	
	/**
	 * The function returns an interval content of the current object.
	 * @return: interval content of the current object
	 */
	public interval_C getInterval()
	{
		interval_C interval;
		
		//check whether a multi-value object contains an interval
		if(_vector.get(0).getClass()!=interval_C.class)
		{
			System.err.println("multi_val_C.getInterval: no interval in multi-value object");
			System.exit(1);
		}
		
		interval = (interval_C)_vector.get(0);
		
		return interval;
	}
	
	/**
	 * The function returns a vector content of the current object.
	 * @return: vector content of the current object
	 */
	public vector_C getVector()
	{
		vector_C vector;
		
		vector = new vector_C(_vector);
		
		return vector;
	}
	
	/**
	 * The function searches for the largest absolute value in a contained interval object.
	 * @return: largest absolute value of an interval limits
	 */
	public double getMaxAbsInterval()
	{
		double max_abs;
		double lower_lim, upper_lim;
		interval_C interval;
		
		//check whether a multi-value object contains an interval
		if(_vector.get(0).getClass()!=interval_C.class)
		{
			System.err.println("multi_val_C.getInterval: no interval in multi-value object");
			System.exit(1);
		}
		
		interval = (interval_C)_vector.get(0);
		
		lower_lim = interval.getLowerLimitAsDouble();
		upper_lim = interval.getUpperLimitAsDouble();
		if(Math.abs(lower_lim) > Math.abs(upper_lim))
		{
			max_abs = Math.abs(lower_lim);
		}
		else
		{
			max_abs = Math.abs(upper_lim);
		}
		    
		return max_abs;
	}
	
	/**
	 * The function converts a contained interval into an array of doubles.
	 * @return: interval as an array of doubles
	 */
	public double[] getArray()
	{
		double[] out = new double[2];
		interval_C interval;
		
		//check whether a multi-value object contains an interval
		if(_vector.get(0).getClass()!=interval_C.class)
		{
			System.err.println("multi_val_C.getInterval: no interval in multi-value object");
			System.exit(1);
		}
		
		interval = (interval_C)_vector.get(0);
		
		out[0] = interval.getLowerLimitAsDouble();
		out[1] = interval.getUpperLimitAsDouble();
		    
		return out;
	}
	
	/**
	 * The function indicates whether the current object contains an interval.
	 * @return TRUE: content of an interval, FALSE: otherwise
	 */
	public boolean isInterval()
	{
		boolean is_interval;//output variable
		
		//check whether a multi-value object contains an interval
		if(_vector.get(0).getClass()==interval_C.class)
		{
			is_interval = true;
		}
		else
		{
			is_interval = false;
		}
		
		return is_interval;
	}
}
