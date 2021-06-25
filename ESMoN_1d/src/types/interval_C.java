package types;

import java.util.StringTokenizer;

public class interval_C
{
	private Number le;//left boundary of interval
	private Number ri;//right boundary of interval
	
	public interval_C()
	{
		le = Double.NaN;
		ri = Double.NaN;
	}
	
	/**
	 * The function sets a provided value as the left border of an interval.
	 * 
	 * @param left: value to assign the left border
	 */
	public void setLeftBorder(Number left)
	{
		le = left;
	}
	
	/**
	 * The function sets a provided value as the right border of an interval.
	 * 
	 * @param right: value to assign the right border
	 */
	public void setRightBorder(Number right)
	{
		ri = right;
	}

	public interval_C(Number a, Number b)
	{
		le = a;
		ri = b;
	}
	
	public interval_C(interval_C orig)
	{
		le = orig.le;
		ri = orig.ri;
	}
	
	public interval_C(String str)
	{
		Double lower_lim, upper_lim;
		String str_le, str_ri;//strings with the left and right border
		
		StringTokenizer st = new StringTokenizer(str, "(,)");
		
		str_le = st.nextToken();
		str_ri = st.nextToken();
		if(st.hasMoreTokens())
		{
			System.err.println("interval_C: invalid string to convert into interval");
			System.exit(1);
		}

		//try to extract and set borders of an interval
		lower_lim = Double.NaN;
		try {
			lower_lim = Double.valueOf(str_le);
		} catch (NumberFormatException nfe) {
			//do nothing; initial value of "lower_lim" is kept; this can happen for dummy intervals "(-,-)"
		}
		le = lower_lim;
		
		upper_lim = Double.NaN;
		try {
			upper_lim = Double.valueOf(str_ri);
		} catch (NumberFormatException nfe) {
			//do nothing; initial value of "lower_lim" is kept; this can happen for dummy intervals "(-,-)"
		}
		ri = upper_lim;
		
		//check values if both extracted values are meaningful
		if(((Double)le).isNaN()==false && ((Double)ri).isNaN()==false)
		{
			if(lower_lim.doubleValue() > upper_lim.doubleValue())
			{
				System.err.println("interval_C: extracted lower limit is larger than the extracted upper limit");
				System.exit(1);
			}
		}
	}
	
	/**
	 * The function copies a provided interval to the host interval. 
	 */
	public void copy(interval_C source)
	{
		le = source.le;
		ri = source.ri;
	}
	
	/**
	 * obtain boundaries of interval as an array where element 0 is the left boundary and element 1 is the right
	 * boundary 
	 * @return, array of interval boundaries
	 */
	public double[] getArray()
	{
		double[] out = new double[2];
		
		out[0] = (Double)le;
		out[1] = (Double)ri;
		    
		return out;
	}
	
	/**
	 * The function indicates a lower limit of an interval as a double
	 * 
	 * @return lower limit as a double
	 */
	public double getLowerLimitAsDouble()
	{
		return le.doubleValue();
	}
	
	/**
	 * The function indicates an upper limit of an interval as a double
	 * 
	 * @return upper limit as a double
	 */
	public double getUpperLimitAsDouble()
	{
		return ri.doubleValue();
	}
	
	/**
	 * The function indicates whether the current interval is valid.
	 * An interval is invalid if one of its limits is Double.NaN.
	 * 
	 * @return "true" if interval is valid; "false" - otherwise
	 */
	public boolean isIntervalValid()
	{
		boolean is_valid;
		
		if(((Double)le).isNaN()==true || ((Double)ri).isNaN()==true)
		{
			is_valid = false;
		}
		else
		{
			is_valid = true;
		}
		
		return is_valid;
	}
	
	/**
	 * convert an interval object to string
	 */
	public String toString()
	{
		String str_out;
		
		str_out = "(";
		str_out+= le.toString();
		str_out+= ",";
		str_out+= ri.toString();
		str_out+= ")";
		
		return str_out;
	}
}
