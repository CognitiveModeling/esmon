package adaptation;

import java.util.Vector;

/**
 * This class keeps information about errors at previous evaluations of an individual.
 * @author Danil Koryakin
 */
public class DiffEvolutionError
{
	public Vector<Double> all_single_squares;//all up-to-now collected squares of error deviations
	public double sum_error;//sum of errors at previous evaluations with an individual
	public int cnt_error;//counter of previous evaluations
	public double avg_mse;//average MSE
	public boolean is_increase;//indicator whether an error is steadily increasing

	/**
	 * class constructor
	 */
	public DiffEvolutionError()
	{
		all_single_squares    = new Vector<Double>(0, 1);
		sum_error  = 0;
		cnt_error  = 0;
		avg_mse  = Double.MAX_VALUE;
		is_increase = true;
	}
	
	/**
	 * The function updates an indicator of whether an error is increasing.
	 */
	private void updateIncrease()
	{
		int i;
		
		is_increase = true;
		for(i=0; i<all_single_squares.size()-1 && is_increase==true; i++)
		{
			if(all_single_squares.get(i).doubleValue() >= all_single_squares.get(i+1).doubleValue())
			{
				is_increase = false;
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
		Double tmp_error;
		
		sum_error   = error.sum_error;
		cnt_error   = error.cnt_error;
		avg_mse   = error.avg_mse;
		is_increase = error.is_increase;

		//copy a list of single squares
		all_single_squares = new Vector<Double>(0, 1);
		num_errors = error.all_single_squares.size();
		for(i=0; i<num_errors; i++)
		{
			tmp_error = error.all_single_squares.elementAt(i).doubleValue();
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
		double single_deviation;//average simple deviation over all outputs
		double single_square;//average squared deviation over all outputs
		
		//compute a single square for the provided array of deviations
		single_deviation = 0;
		single_square    = 0;
		for(i=0; i<deviation.length; i++)
		{
			single_deviation += deviation[i];
			single_square    += (deviation[i] * deviation[i]);
		}
		single_deviation /= deviation.length;
		single_square    /= deviation.length;
		
		//store deviations
		all_single_squares.add(single_square);
		
		sum_error += single_square;
		cnt_error ++;
		avg_mse = sum_error/cnt_error;
		
		updateIncrease();
	}
	
	/**
	 * The function computes NRMSE for collected data.
	 * The formula for computing NRMSE is the same as in "roeschies09" and in "otte16".
	 * 
	 * The largest variance is used for computation.
	 * 
	 * @param variance: array of variances for each element of the target vector
	 * @return: computed NRMSE
	 */
	public double computeNrmse(double[] variance)
	{
		int i;
		double variance_max;
		double nrmse;
		
		//search for the largest variance to use in computation
		variance_max = variance[0];
		for(i=1; i<variance.length; i++)
		{
			if(variance_max < variance[i])
			{
				variance_max = variance[i];
			}
		}
		
		nrmse = avg_mse / variance_max;
		nrmse = Math.sqrt(nrmse);
		
		return nrmse;
	}
	
	/**
	 * The function computes RMSE for collected data.
	 * 
	 * @return: computed RMSE
	 */
	public double computeRmse()
	{
		double rmse;
		
		rmse = Math.sqrt(avg_mse);
		
		return rmse;
	}
	
	/**
	 * The function removes all contents from the error object.
	 */
	public void clean()
	{
		all_single_squares.clear();
		sum_error  = 0;
		cnt_error  = 0;
		avg_mse  = Double.MAX_VALUE;
		is_increase = true;
	}

	/**
	 * The function returns an average error over all errors that were computed for the given individual.
	 * @return: average error
	 */
	public double getAverageError()
	{
		return avg_mse;
	}

	/**
	 * The function returns an average MSE over several first time steps after the given individual has been computed.
	 * The number of time steps is specified by the provided parameter.
	 * 
	 * @param num_errors: number of requested time steps
	 * @return: average MSE
	 */
	public double getAverageMseFirst(int num_time_steps)
	{
		int i;
		double average;//output variable

		if(all_single_squares.size()<num_time_steps)
		{
			System.err.println("getAverageErrorFirst: invalid input parameter");
			System.exit(1);
		}
		average = 0;
		for(i=0; i<num_time_steps; i++)
		{
			average += all_single_squares.get(i).doubleValue();
		}
		average /= num_time_steps;

		return average;
	}
	
	/**
	 * The function returns the average MSE over several last time steps of the host individual.
	 * The requested number of time steps is specified by the provided parameter.
	 * 
	 * @param num_time_steps: requested number of time steps
	 * @return: average MSE
	 */
	public double getAverageMseLast(int num_time_steps)
	{
		int i;
		int total_num_time_steps;
		double average;//output variable

		total_num_time_steps = all_single_squares.size();
		if(total_num_time_steps<num_time_steps)
		{
			System.err.println("getAverageErrorLast: invalid input parameter");
			System.exit(1);
		}
		average = 0;
		for(i=1; i<=num_time_steps; i++)
		{
			average += all_single_squares.get(total_num_time_steps - i).doubleValue();
		}
		average /= num_time_steps;

		return average;
	}
	
	/**
	 * The function returns the last stored deviation of the network outputs from their target values.
	 * 
	 * @return: last stored deviation of the host individual
	 */
	public double getDeviationLast()
	{ 
		if(all_single_squares.isEmpty()==true)
		{
			System.err.println("DiffEvolutionError.getDeviationLast: list of deviations is empty");
			System.exit(1);
		}
		
		return Math.sqrt(all_single_squares.lastElement().doubleValue());
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
}
