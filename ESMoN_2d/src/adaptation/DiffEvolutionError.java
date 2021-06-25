package adaptation;

import java.util.Vector;

/**
 * This class keeps information about errors at previous evaluations of an individual.
 * @author Danil Koryakin
 */
public class DiffEvolutionError
{
	/**
	 * The class keeps errors for each element of the output vecetor.
	 * 
	 * @author Danil
	 */
	private class error_vector_C
	{
		private double[] _error;
		
		public error_vector_C(double[] deviation)
		{
			int i;
			
			_error = deviation.clone();
			for(i=0; i<_error.length; i++)
			{
				_error[i] *= _error[i];
			}
		}
	}
	
	public Vector<error_vector_C> all_single_squares;//all up-to-now collected squares of error deviations
	public double[] sum_error;//sum of errors at previous evaluations with an individual
	public double[] avg_mse;//average MSE
	public double[] _tmp_output;//temporary array to output values that are not directly stored within the class
	public int cnt_error;//counter of previous evaluations
	public boolean is_increase;//indicator whether an error is steadily increasing
	
	/**
	 * class constructor
	 * 
	 * error_length: number of elements in the output vector 
	 */
	public DiffEvolutionError(int error_length)
	{
		int i;
		
		all_single_squares = new Vector<error_vector_C>(0, 1);
		
		sum_error = new double[error_length];
		avg_mse = new double[error_length];
		_tmp_output = new double[error_length];
		
		for(i=0; i<error_length; i++)
		{
			sum_error[i] = 0;
			avg_mse[i] = Double.MAX_VALUE;
		}
		
		cnt_error  = 0;
		is_increase = true;
	}
	
	/**
	 * The function updates an indicator of whether an error is increasing.
	 */
	private void updateIncrease()
	{
		int i,j;
		int error_len;
		double[] error_cur, error_next;
		
		is_increase = false;
		error_len = avg_mse.length;
		for(j=0; j<error_len && is_increase==false; j++)//do not do the next error dimension if the error was increasing
		{                                               //  in the previous error dimension
			is_increase = true;
			for(i=0; i<all_single_squares.size()-1 && is_increase==true; i++)
			{
				error_cur  = all_single_squares.get(i)._error;
				error_next = all_single_squares.get(i+1)._error;
				if(error_cur[j] >= error_next[j])
				{
					is_increase = false;
				}
			}
		}
	}
	
	/**
	 * The function copies data of a provided error status to the host error status.
	 * 
	 * @param error: provided error status
	 */
	public void copy(DiffEvolutionError error)
	{
		int i;
		int num_errors;
		error_vector_C tmp_error;
		
		sum_error   = error.sum_error.clone();
		cnt_error   = error.cnt_error;
		avg_mse     = error.avg_mse.clone();
		is_increase = error.is_increase;
		
		all_single_squares = new Vector<error_vector_C>(0, 1);
		num_errors = error.all_single_squares.size();
		for(i=0; i<num_errors; i++)
		{
			tmp_error = new error_vector_C(error.all_single_squares.elementAt(i)._error);
			all_single_squares.add(tmp_error);
		}
	}

	/**
	 * The function updates an error status with a provided error from a previous evaluation of an individual.
	 * The error is provided as an array of deviations of each output of the network from the corresponding target
	 * value.
	 * 
	 * @param error: provided array of deviations for all network outputs
	 */
	public void update(double[] deviation)
	{
		int i;
		error_vector_C error_vector;
		
		error_vector = new error_vector_C(deviation);
		
		//update the sum by the provided array of deviations
		cnt_error ++;
		for(i=0; i<deviation.length; i++)
		{
			sum_error[i] += (deviation[i]*deviation[i]);
			avg_mse[i] = sum_error[i]/cnt_error;
		}
		
		//store deviations
		all_single_squares.add(error_vector);
		
		updateIncrease();
	}

	/**
	 * The function computes NRMSE for collected data.
	 * The formula for computing NRMSE is the same as in "roeschies09" and in "otte16".
	 * 
	 * The largest variance is used for computation.
	 * 
	 * @param variance: array of variances for each element of the target vector
	 * @return: computed NRMSE for each element of the target vector
	 */
	public double[] computeNrmse(double[] variance)
	{
		int i;
		double[] nrmse;//output array
		
		nrmse = new double[variance.length];
		
		//compute NRMSE for each element of the output vector
		for(i=0; i<variance.length; i++)
		{
			nrmse[i] = avg_mse[i] / variance[i];
			nrmse[i] = Math.sqrt(nrmse[i]);
		}
		
		return nrmse;
	}
	
	/**
	 * The function computes RMSE for collected data.
	 * 
	 * @return: computed RMSE for all elements of the output vector
	 */
	public double[] computeRmse()
	{
		int i;
		double[] rmse;//output array
		
		rmse = new double[avg_mse.length];
		
		//compute RMSE for each element of the output vector
		for(i=0; i<avg_mse.length; i++)
		{
			rmse[i] = Math.sqrt(avg_mse[i]);
		}
		
		return rmse;
	}
	
	/**
	 * The function removes all contents from the error object.
	 */
	public void clean()
	{
		int i;
		
		all_single_squares.clear();
		cnt_error  = 0;
		
		for(i=0; i<sum_error.length; i++)
		{
			sum_error[i] = 0;
			avg_mse[i] = Double.MAX_VALUE;
		}
		is_increase = true;
	}

	/**
	 * The function returns an average error over all errors that were computed for the given individual.
	 * @return: average error
	 */
	public double[] getAverageError()
	{
		return avg_mse;
	}
	
	/**
	 * The function indicates the largest element of the vector of the average MSE.
	 * 
	 * @return: largest element of the vector of the average MSE
	 */
	public double getMaxAverageMse()
	{
		int i;
		double avg_max;//largest average error
		
		avg_max = avg_mse[0];
		for(i=1; i<avg_mse.length; i++)
		{
			if(avg_max < avg_mse[i])
			{
				avg_max = avg_mse[i];
			}
		}
		
		return avg_max;
	}

	/**
	 * The function returns an average MSE over several first time steps after the given individual has been computed.
	 * The number of time steps is specified by the provided parameter.
	 * 
	 * @param num_time_steps: number of requested time steps
	 * @return: average MSE
	 */
	public double[] getAverageMseFirst(int num_time_steps)
	{
		int i,j;
		double[] error_cur;//currently considered error
		double[] average;//output variable

		if(all_single_squares.size()<num_time_steps)
		{
			System.err.println("getAverageErrorFirst: invalid input parameter");
			System.exit(1);
		}
		
		//*** initialization ***
		
		average = new double[avg_mse.length];
		for(i=0; i<average.length; i++)
		{
			average[i] = 0;
		}
		
		//*** summing up ***
		
		for(i=0; i<num_time_steps; i++)
		{
			error_cur = all_single_squares.get(i)._error;
			for(j=0; j<average.length; j++)
			{
				average[j] += error_cur[j];
			}
		}
		
		//*** computing the average
		
		for(i=0; i<average.length; i++)
		{
			average[i] /= num_time_steps;
		}

		return average;
	}
	
	/**
	 * The function returns the average MSE over several last time steps of the host individual.
	 * The requested number of time steps is specified by the provided parameter.
	 * 
	 * @param num_time_steps: requested number of time steps
	 * @return: average MSE
	 */
	public double[] getAverageMseLast(int num_time_steps)
	{
		int i,j;
		int total_num_errors;
		double[] error_cur;//currently considered error
		double[] average;//output variable

		total_num_errors = all_single_squares.size();
		if(total_num_errors<num_time_steps)
		{
			System.err.println("getAverageErrorLast: invalid input parameter");
			System.exit(1);
		}
		
		//*** initialization ***
		
		average = new double[avg_mse.length];
		for(i=0; i<average.length; i++)
		{
			average[i] = 0;
		}
		
		//*** summing up ***
		
		for(i=1; i<=num_time_steps; i++)
		{
			error_cur = all_single_squares.get(total_num_errors - i)._error;
			for(j=0; j<average.length; j++)
			{
				average[j] += error_cur[j];
			}
		}
		
		//*** computing the average
		
		for(i=0; i<average.length; i++)
		{
			average[i] /= num_time_steps;
		}

		return average;
	}
	
	/**
	 * The function indicates the largest element of the vector of RMSE.
	 * 
	 * @return: largest element of the vector of RMSE
	 */
	public double getMaxRmse()
	{
		int i;
		double max_rmse;//largest RMSE
		double[] rmse;//array of all RMSE
		
		rmse = computeRmse();
		
		max_rmse = rmse[0];
		for(i=1; i<rmse.length; i++)
		{
			if(max_rmse < rmse[i])
			{
				max_rmse = rmse[i];
			}
		}
		
		return max_rmse;
	}
	
	/**
	 * The function indicates the largest element of the vector of NRMSE.
	 * 
	 * @param variance: array of variances for each element of the target vector
	 * @return: largest element of the vector of NRMSE
	 */
	public double getMaxNrmse(double[] variance)
	{
		int i;
		double max_nrmse;//largest NRMSE
		double[] nrmse;//array of all NRMSE
		
		nrmse = computeNrmse(variance);
		
		max_nrmse = nrmse[0];
		for(i=1; i<nrmse.length; i++)
		{
			if(max_nrmse < nrmse[i])
			{
				max_nrmse = nrmse[i];
			}
		}
		
		return max_nrmse;
	}

	/**
	 * The function retrieves an error which is specified by the provided index.
	 * 
	 * @param idx: provided index
	 */
	public double[] getError(int idx)
	{
		if(idx>=all_single_squares.size())
		{
			System.err.println("error_C.getError: index is out of bounds");
			System.exit(1);
		}

		return all_single_squares.get(idx)._error;
	}
	
	/**
	 * The function returns the last stored deviation of the network outputs from their target values.
	 * 
	 * @return: last stored deviation of the host individual
	 */
	public double[] getDeviationLast()
	{
		int i;
		error_vector_C last_mse_vector;//last stored MSE vector
		
		if(all_single_squares.isEmpty()==true)
		{
			System.err.println("DiffEvolutionError.getDeviationLast: list of deviations is empty");
			System.exit(1);
		}
		
		last_mse_vector = all_single_squares.lastElement();
		
		for(i=0; i<avg_mse.length; i++)
		{
			_tmp_output[i] = Math.sqrt(last_mse_vector._error[i]);
		}
		
		return _tmp_output;
	}
	
	/**
	 * The function returns an increase indicator of the error.
	 *  
	 * @return
	 */
	public boolean isIncrease()
	{
		return is_increase;
	}
	
	/**
	 * The function indicates whether the host error is better than the submitted error.
	 * 
	 * Here "better" means at least one element is smaller and the others are not larger.
	 * 
	 * @param submitted_error: submitted error
	 * @return: TRUE if better; FALSE - otherwise
	 */
	public boolean isBetter(double[] submitted_error)
	{
		int i;
		boolean is_smaller;//indicator that at least one element of host vector is smaller than that of submitted one
		boolean is_larger;//indicator that at least one element of the host error is larger than that of submitted one
		boolean is_better;//output variable
		
		//compare elements of the host and submitted errors
		is_smaller = false;
		is_larger  = false;
		for(i=0; i<avg_mse.length && is_larger==false; i++)
		{
			if(avg_mse[i] > submitted_error[i])
			{
				is_larger = true;
			}
			else if(avg_mse[i] < submitted_error[i])
			{
				is_smaller = true;
			}
			else
			{
				//do nothing
			}
		}
		
		if(is_larger==true)
		{
			is_better = false;
		}
		else if(is_smaller==true)
		{
			is_better = true;
		}
		else
		{
			is_better = false;
		}
		
		return is_better;
	}
	
	/**
	 * The function indicates whether all elements of the host average error are smaller than the provided threshold.
	 * 
	 * @param thresh: provided threshold
	 * @return: TRUE if all elements are smaller; FALSE - otherwise
	 */
	public boolean isBetterThresh(double thresh)
	{
		int i;
		boolean is_better;//output variable
		
		//compare elements of the host and submitted errors
		is_better = true;
		for(i=0; i<avg_mse.length && is_better==true; i++)
		{
			if(avg_mse[i] > thresh)
			{
				is_better = false;
			}
		}
		
		return is_better;
	}
	
	/**
	 * The function indicates whether the host error is the same as the submitted error.
	 * 
	 * @param submitted_error: submitted error
	 * @return: TRUE if the same; FALSE - otherwise
	 */
	public boolean isSame(double[] submitted_error)
	{
		int i;
		boolean is_same;//output variable
		
		//compare elements of the host and submitted errors
		is_same = true;
		for(i=0; i<avg_mse.length && is_same==true; i++)
		{
			if(avg_mse[i]!=submitted_error[i])
			{
				is_same = false;
			}
		}
		
		return is_same;
	}
	
	/**
	 * The function indicates whether the host error is worse than the submitted error.
	 * 
	 * The error vector is worse if it has more elements that are larger than the corresponding elements of
	 * the submitted error vector.
	 * 
	 * @param submitted_error: submitted error
	 * @return: TRUE if worse; FALSE - otherwise
	 */
	public boolean isWorse(double[] submitted_error)
	{
		int i;
		int cnt_larger_host;//counter of larger elements of the host error vector
		int cnt_larger_submit;//counter of larger elements of the submitted error vector
		boolean is_worse;//output variable
		
		//compare elements of the host and submitted errors
		cnt_larger_host   = 0;
		cnt_larger_submit = 0;
		for(i=0; i<avg_mse.length; i++)
		{
			if(avg_mse[i] > submitted_error[i])
			{
				cnt_larger_host++;
			}
			else if(avg_mse[i] < submitted_error[i])
			{
				cnt_larger_submit++;
			}
			else
			{
				//do not count if elements are equal
			}
			
		}
		
		//host error vector is worse only if it has more larger elements
		if(cnt_larger_host > cnt_larger_submit)
		{
			is_worse = true;
		}
		else
		{
			is_worse = false;
		}
		
		return is_worse;
	}
}
