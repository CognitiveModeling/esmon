package types;

import java.util.Vector;

import Jama.Matrix;

/**
 * This class performs conversions from one class to another.
 * 
 * @author Danil Koryakin
 */
public class conversion_C
{
	/**
	 * The function converts the provided one-dimensional array of the type Object into a one-dimensional array of
	 * the type "double".
	 * 
	 * @param obj_val: provided array of objects
	 * @return array of the type "double"
	 */
	public static double[] ObjToDouble1D(Object[] obj_val)
	{
		int i;
		double[] double_val;//array of output values
		
		double_val = new double[obj_val.length];
		
		for(i=0; i<obj_val.length; i++)
		{
			double_val[i] = ((Double)obj_val[i]).doubleValue();
		}
		
		return double_val;
	}
	
	/**
	 * The function converts the provided two-dimensional array of the type String into a two-dimensional array of
	 * the type "double".
	 * 
	 * @param str_val: provided array of strings
	 * @return array of the type "double"
	 */
	public static double[][] StrToDouble2D(String[][] str_val)
	{
		int i,j;
		double[][] double_val;//array of output values
		
		double_val = new double[str_val.length][];
		
		for(i=0; i<str_val.length; i++)
		{
			double_val[i] = new double[str_val[i].length];
			for(j=0; j<double_val[i].length; j++)
			{
				double_val[i][j] = new Double(str_val[i][j]);
			}
		}
		
		return double_val;
	}
	
	/**
	 * The function converts the provided two-dimensional array of the type String into a matrix.
	 * 
	 * @param obj_val: provided array of objects
	 * @return matrix of the provided values
	 */
	public static Matrix StrToMatrix2D(String[][] str_val)
	{
		double[][] double_val;//intermediate array
		Matrix matrix_val;//array of output values
		
		double_val = StrToDouble2D(str_val);
		matrix_val = new Matrix(double_val);
		
		return matrix_val;
	}
	
	/**
	 * The function converts the one-dimensional provided array of the type Object into a one-dimensional array of
	 * the type "int".
	 * 
	 * @param obj_val: provided array of objects
	 * @return array of the type "int"
	 */
	public static int[] ObjToInt1D(Object[] obj_val)
	{
		int i;
		int[] int_val;//array of output values
		
		int_val = new int[obj_val.length];
		
		for(i=0; i<obj_val.length; i++)
		{
			int_val[i] = ((Integer)obj_val[i]).intValue();
		}
		
		return int_val;
	}
	
	/**
	 * The function converts the provided one-dimensional array of the type Object into a one-dimensional array of
	 * the type "interval_C".
	 * 
	 * @param obj_val: provided array of objects
	 * @return array of the type "interval_C"
	 */
	public static interval_C[] ObjToInterval1D(Object[] obj_val)
	{
		int i;
		String tmp_str;//temporary string
		interval_C[] interval_val;//array of output values
		
		interval_val = new interval_C[obj_val.length];
		
		for(i=0; i<obj_val.length; i++)
		{
			tmp_str = (String)obj_val[i];
			interval_val[i] = new interval_C(tmp_str);
		}
		
		return interval_val;
	}
	
	/**
	 * The function converts the provided one-dimensional array of the type String into a two-dimensional array of
	 * the type "interval_C".
	 * 
	 * @param str_val: provided array of strings
	 * @return array of the type "interval_C"
	 */
	public static interval_C[][] StrToInterval2D(String[][] str_val)
	{
		int i, j;
		boolean is_finished;//extraction of the last non-dummy interval has been finished
		interval_C[][] interval_val;//array of output values
		interval_C     tmp_interval;
		Vector<interval_C> tmp_intervals;//temporary (extensible) array of intervals
		
		interval_val = new interval_C[str_val.length][];
		
		for(i=0; i<str_val.length; i++)
		{
			tmp_intervals = new Vector<interval_C>(0, 1);
			
			is_finished = false;
			for(j=0; j<str_val[i].length && is_finished==false; j++)
			{
				tmp_interval = new interval_C(str_val[i][j]);
				if(tmp_interval.isIntervalValid()==false)
				{
					is_finished = true;
				}
				//avoid adding dummy intervals that can be present in the end of a row
				if(is_finished==false)
				{
					tmp_intervals.add(tmp_interval);
				}
			}
			
			//assign elements from a temporary array
			interval_val[i] = new interval_C[tmp_intervals.size()];
			for(j=0; j<interval_val[i].length; j++)
			{
				interval_val[i][j] = new interval_C(tmp_intervals.get(j));
			}
		}
		
		return interval_val;
	}
}
