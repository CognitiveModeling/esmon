package MathStat;

import java.util.Vector;

import types.stat_seq_C;

/**
 * This class implements functions for computing different statistics. 
 * @author Danil Koryakin
 *
 */
public class StatCommon {

	/**
	 * interface class to return the output statistics like the median value or the best value
	 * which require an index of the corresponding array's element additionally to its value
	 * @author Danil Koryakin 
	 */
	public class idx_stat_C
	{
		public int    idx;
		public Number val;
	};
	
	/**
	 * compute a mean value of the values in the given array
	 * @param array, given values
	 * @return, computed mean value
	 */
	public static double computeMean(Number[] array)
	{
		int i;
		double mean;
		
		mean = 0.0;
		for(i=0; i<array.length; i++)
		{
			mean += array[i].doubleValue();
		}
		mean /= array.length;
			
		return mean;
	}
	
	/**
	 * The function computed the grand mean for all performance indicators in the provided vector.
	 * (Normally they are mean values; therefore, computed means are the grand means.)
	 * 
	 * @param stat_seq: vector of performance statistics
	 * @return: structure where all performance indicators are the grand means
	 */
	public static stat_seq_C computeGrandMean(Vector<stat_seq_C> stat_seq)
	{
		int i, j, k;
		int num_values;//number of values to compute the mean
		int num_out;//extracted number of outputs
		int num_sub;//extracted number of modules
		stat_seq_C grand_mean;//mean for each performance indicator (output variable)
		
		//create the output variable
		if(stat_seq.get(0).sel_config!=null)
		{
			num_sub = stat_seq.get(0).sel_config.length;
		}
		else
		{
			num_sub = 0;
		}
		num_out = stat_seq.get(0).mse.length;
		grand_mean = new stat_seq_C(num_out, num_sub);
		
		num_values = stat_seq.size();
		grand_mean.InitAllTo0();
		
		//means that are different for different outputs
		for(j=0; j<num_out; j++)
		{
			for(i=0; i<num_values; i++)
			{
				//sum of "mse"
				grand_mean.mse[j] += stat_seq.get(i).mse[j];

				//sum of "nrmse"
				grand_mean.nrmse[j] += stat_seq.get(i).nrmse[j];
				
				//sum of "rmse"
				grand_mean.rmse[j] += stat_seq.get(i).rmse[j];

				//sum of "sel"
				grand_mean.sel[j] += stat_seq.get(i).sel[j];

				//sum of "lel"
				grand_mean.lel[j] += stat_seq.get(i).lel[j];

				//sum of "sel_config"
				if(grand_mean.sel_config!=null)
				{
					for(k=0; k<num_sub; k++)
					{
						grand_mean.sel_config[k][j] += stat_seq.get(i).sel_config[k][j];
					}
				}
			}
			grand_mean.mse[j]   /= num_values;
			grand_mean.nrmse[j] /= num_values;
			grand_mean.rmse[j]  /= num_values;
			grand_mean.sel[j]   /= num_values;
			grand_mean.lel[j]   /= num_values;
			if(grand_mean.sel_config!=null)
			{
				for(k=0; k<num_sub; k++)
				{
					grand_mean.sel_config[k][j] /= num_values;
				}
			}
		}
		
		//component identification is independent of a number of outputs
		for(i=0; i<num_values; i++)
		{
			grand_mean.comp_ident_ok += stat_seq.get(i).comp_ident_ok;
		}
		grand_mean.comp_ident_ok /= num_values;
		
		//component identification (incl. error) is independent of a number of outputs
		for(i=0; i<num_values; i++)
		{
			grand_mean.comp_ident_ok_incl_error += stat_seq.get(i).comp_ident_ok_incl_error;
		}
		grand_mean.comp_ident_ok_incl_error /= num_values;
		
		return grand_mean;
	}
	
	/**
	 * The function computes a mean value among a specified portion of the smallest values in the given array.
	 * Before computing a mean value, the function sorts the values in the ascending order.
	 * 
	 * @param array: given values
	 * @param portion: specified portion 
	 * @return: computed mean value
	 */
	public double computeMeanOfSmallest(Number[] array, double portion)
	{
		int i, j;
		int num_total;//number of provided values
		int num_portion;//number of values to compute a mean
		double tmp_val;
		double mean;
		double[] local_array;//local array to store the provided values in order not to change the provided array
		
		//store the provided array locally
		num_total = array.length;
		local_array = new double[num_total];
		for(i=0; i<num_total; i++)
		{
			local_array[i] = array[i].doubleValue();
		}
		
		//sort a provided array
		for(i=0; i<num_total-1; i++)
		{
			for(j=i+1; j<num_total; j++)
			{
				if(local_array[j] < local_array[i])
				{
					tmp_val = local_array[i];
					local_array[i] = local_array[j];
					local_array[j] = tmp_val;
				}
			}
		}
		
		//compute a mean value
		num_portion = (int)(portion * num_total);
		mean = 0.0;
		for(i=0; i<num_portion; i++)
		{
			mean += local_array[i];
		}
		mean /= num_portion;
			
		return mean;
	}
	
	/**
	 * compute a variance of values in the given array
	 * @param array, given values
	 * @return, computed standard deviation
	 */
	public double computeStdDev(Number[] array)
	{
		double variance;
		double std_dev;
		
		variance = computeVariance(array);
		std_dev  = Math.sqrt(variance);
		
		return std_dev;
	}
	
	/**
	 * compute a sum of values in the given array
	 * @param array, given values
	 * @return, computed sum
	 */
	public double computeSum(Number[] array)
	{
		int i;
		double sum;
		
		sum = 0.0;
		for(i=0; i<array.length; i++)
		{
			sum += array[i].doubleValue();
		}
			
		return sum;
	}
	
	/**
	 * compute a variance of values in the given array
	 * @param array, given values
	 * @return, computed variance
	 */
	public static double computeVariance(Number[] array)
	{
		int i;
		double mean;//mean value used in the computation of variance
		double variance;
		
		mean = StatCommon.computeMean(array);
		
		variance = 0.0;
		for(i=0; i<array.length; i++)
		{
			variance += ((array[i].doubleValue() - mean) * (array[i].doubleValue() - mean));
		}
		variance /= (array.length - 1);
		
		return variance;
	}
	
	/**
	 * The function searches for the smallest element in a submitted array.
	 * 
	 * @param array: submitted array
	 * @return: value and index of the found smallest element
	 */
	public idx_stat_C findMin(Number[] array)
	{
		int i;
		idx_stat_C min = new idx_stat_C();
		
		//submitted array should not be empty
		if(array.length > 0)
		{
			//assign the 1st array's element as the smallest before the search 
			min.idx = 0;
			min.val = array[min.idx];
			for(i=1; i<array.length; i++)
			{
				if(min.val.doubleValue() > array[i].doubleValue())
				{
					min.idx = i;
					min.val = array[min.idx];
				}
			}
		}
		else
		{
			min.val = 0;
			min.idx = 0;
		}
		
		return min;
	}
	
	/**
	 * The function searches for the largest element in a submitted array.
	 * 
	 * @param array: submitted array
	 * @return: value and index of the found largest element
	 */
	public idx_stat_C findMax(Number[] array)
	{
		int i;
		idx_stat_C max = new idx_stat_C();
		
		//submitted array should not be empty
		if(array.length > 0)
		{
			//assign the 1st array's element as the smallest before the search 
			max.idx = 0;
			max.val = array[max.idx];
			for(i=1; i<array.length; i++)
			{
				if(max.val.doubleValue() < array[i].doubleValue())
				{
					max.idx = i;
					max.val = array[max.idx];
				}
			}
		}
		else
		{
			max.val = 0;
			max.idx = 0;
		}
		
		return max;
	}
	
	/**
	 * The function searches for a median element in a submitted array.
	 * 
	 * @param array: submitted array
	 * @return: value and index of the found median element
	 */
	public idx_stat_C findMedian(Number[] array)
	{
		int i, j;		
		int[]    idx_array = new int[array.length];//indices corresponding to elements in the given array
		Number[] val_array;//copy of the given array
		//index of the middle element
		int idx_median = array.length/2;
		//output variable
		idx_stat_C median = new idx_stat_C();
		Number tmp_double;
		int    tmp_int;
		
		//make a copy of the given array
		val_array = array.clone();
		
		//assign array of indices
		for(i=0; i<array.length; i++)
		{
			idx_array[i] = i;
		}

		//sort array of errors coupled with array of initial indices of elements in ascending order 
		for(i=0; i<val_array.length-1; i++)
		{
			for(j=i; j<val_array.length; j++)
			{
				//check the test error
				if(val_array[i].doubleValue() > val_array[j].doubleValue())
				{
					//exchange the array's values 
					tmp_double   = val_array[i];
					val_array[i] = val_array[j];
					val_array[j] = tmp_double;
					//exchange their indices
					tmp_int      = idx_array[i];
					idx_array[i] = idx_array[j];
					idx_array[j] = tmp_int;
				}
			}
		}

		median.val = val_array[idx_median];
		median.idx = idx_array[idx_median];
		
		return median;
	}
}
