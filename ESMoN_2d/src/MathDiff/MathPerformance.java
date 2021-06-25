package MathDiff;

import experiment.ExpSeq.sample_C;

/**
 * This class implements functions for computing performance of the echo-state networks.
 * @author Danil Koryakin
 *
 */
public class MathPerformance
{
	/**
	 * compute the mean squared error (MSE) for the provided sequence
	 * The resulting MSE is computed as an average over all elements of the output vector.
	 * 
	 * @param output, sequence of the network output vectors which correspond to the samples of the provided sequence 
	 * @param target, provided sequence
	 * @return MSE for every element of the output vector 
	 */
	public static double computeMSE(double[][] output, sample_C[] target)
	{
		int i, j;
		int size_out;//length of the output vector
		double sum;
		double sum_i;//sum of deviations at one step
		double mse;//computed error to be returned
		
		size_out = target[0]._out.length;
		
		sum = 0.0;
		for(i=0; i < size_out; i++)
		{
			sum_i = 0.0;
			for(j=0; j < output.length; j++)
			{
				sum_i += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}
			sum_i /= output.length;
			sum += sum_i;
		}
		mse = sum / size_out;
		
		return mse;
	}
	
	/**
	 * The function computes a total mean squared error (MSE) over all vector's elements of the provided sequence.
	 * @param output: provided sequence of vectors (for example, the ESN output)
	 * @param target: provided sequence of target vectors
	 * @return: total MSE over all elements of the vector 
	 */
	public static double computeMseTotal(double[][] output, sample_C[] target)
	{
		int i, j;
		int size_out;//length of the output vector
		double mse_i;//MSE from one output element
		double mse;//computed error to be returned
		
		size_out = target[0]._out.length;

		mse = 0;
		for(i=0; i < size_out; i++)
		{
			mse_i = 0;
			for(j=0; j < output.length; j++)
			{
				mse_i += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}
			mse_i /= output.length;
			mse += mse_i;
		}
		mse = mse / size_out;
		
		return mse;
	}
	
	/**
	 * calculate the normalized root mean squared error (NRMSE) for the actual and target outputs;
	 * NRMSE is computed according to formula (33) in "roeschies09"
	 * The resulting NRMSE is computed as an average over all elements of the output vector.
	 *  
	 * @param output, sequence of activation values of the output nodes
	 * @param target, sequence of target values which should be predicted by ESN
	 * @param variance, array of variances for each element of the target vector
	 * @return NRMSE for every element of the output vector
	 */
	public static double computeNRMSE(double[][] output,
			                            sample_C[] target,
			                            double[]   variance)
	{
		int i, j;
		int size_out;//length of the output vector
		double nrmse_i;//MSE from one output element
		double nrmse;//computed error to be returned
		
		size_out = target[0]._out.length;
		
		nrmse = 0;
		for(i=0; i < size_out; i++)
		{
			nrmse_i = 0;
			for(j=0; j < output.length; j++)
			{
				nrmse_i += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}
			nrmse_i /= output.length;
			nrmse_i /= variance[i];
			nrmse_i  = Math.sqrt(nrmse_i);
			nrmse += nrmse_i;
		}
		nrmse = nrmse / size_out;
		
		return nrmse;
	}
	
	/**
	 * The function calculates the root mean squared error (RMSE) from the actual and target outputs.
	 * The resulting RMSE is computed as an average over all elements of the output vector.
	 *  
	 * @param output: sequence of activation values of the output nodes
	 * @param target: sequence of target values which should be predicted by ESN
	 * @return average RMSE over elements of the output vector
	 */
	public static double computeRMSE(double[][] output,
			                         sample_C[] target)
	{
		int i, j;
		int size_out;//length of the output vector
		double rmse;//computed average RMSE to be returned
		double rmse_i;//computed RMSE at the current output element
		
		size_out = target[0]._out.length;
		
		rmse = 0.0;
		for(i=0; i < size_out; i++)
		{
			rmse_i = 0.0;
			for(j=0; j < output.length; j++)
			{
				rmse_i += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}
			rmse_i /= output.length;
			rmse_i  = Math.sqrt(rmse_i);
			
			rmse += rmse_i;
		}
		rmse = rmse / size_out;
		
		return rmse;
	}
	
	/**
	 * compute the small error length (SEL) of the network output for the provided sequence; SEL is counted for each
	 * element of the output vector;
	 * SEL is a number of time steps where a deviation of the network output from the corresponding target value stays
	 * within the specified error threshold
	 * 
	 * The computed SEL is the smallest SEL among elements of the output vector. 
	 * 
	 * @param output, sequence of the network output vectors which correspond to the samples of the provided sequence
	 * @param target, sequence of target values which should be predicted by ESN
	 * @param err_thresh, error threshold
	 * @return smallest SEL over the output vector
	 */
	public static int computeSEL(double[][] output,
			                       sample_C[] target,
			                       double     err_thresh)
	{
		int i, j;
		int size_out;//length of the output vector
		int sel_min;
		int[] sel;//computed performance indicator for all elements of the output vector
		double dev;//deviation at the current sample
		boolean f_measure_done;//indicator that at the current sample the deviation exceeds the threshold
		
		size_out = target[0]._out.length;
		
		//allocate an array for saving the performance indicator for all elements of the output vector
		sel = new int[size_out];
		
        for(i=0; i<size_out; i++)
        {
        	sel[i] = 0;
        	f_measure_done = false;
        	for(j=0; j<target.length && f_measure_done==false; j++)
        	{
        		dev = Math.abs(output[j][i] - target[j]._out[i]);
        		if(dev > err_thresh)
        		{
        			f_measure_done = true;
        		}
        		else
        		{
        			sel[i]++;//increase the currently measured length 
        		}
        	}
        }//for i
        
        //find the smallest SEL
        sel_min = Integer.MAX_VALUE;
        for(i=0; i<size_out; i++)
        {
        	if(sel_min > sel[i])
        	{
        		sel_min = sel[i];
        	}
        }
        
        return sel_min;
	}
	
	/**
	 * The function computes the large error length. The large error length is a number of time steps where
	 * the deviation of the ESN output from its target dynamics stays larger than the provided threshold. The steps are
	 * always counted from the beginning of the sequence. After the last counted time step the largest deviation is
	 * lower than the provided threshold.
	 * The large error length is counted for each output neuron independently on the other output neurons.
	 * The large error length is an indicator which can be used to show a performance of the considered configuration
	 * method. Normally configuration methods should steadily reduce an initially large deviation.
	 * 
	 * The computed SEL is the largest SEL among elements of the output vector.
	 * 
	 * @param output: sequence of the output neurons' states which correspond to the samples of the target sequence
	 * @param target: provided target sequence
	 * @param lel_thresh: provided threshold of the large error length
	 * @return largest LEL over the output vector
	 */
	public static int computeLEL(double[][] output,
			                     sample_C[] target,
			                     double     lel_thresh)
	{
		int i, j;
		int size_out;//number of output neurons
		int lel_max;
		int[] lel;//output array: computed large error length for each output neuron
		double dev;//deviation at the current sample
		boolean f_measure_done;//indicator that at the current sample the deviation exceeds the threshold
		
		size_out = target[0]._out.length;
		
		//allocate the output array
		lel = new int[size_out];
		
        for(i=0; i<size_out; i++)
        {
        	lel[i] = 0;//LEL stays "0", if there is no time step where the deviation is larger than the threshold
        	f_measure_done = false;
        	for(j=target.length-1; j>=0 && f_measure_done==false; j--)
        	{	
        		dev = Math.abs(output[j][i] - target[j]._out[i]);
        		if(dev > lel_thresh)
        		{
        			f_measure_done = true;
        			lel[i] = j+1;
        		}
        	}
        }//for i
        
        //find the smallest SEL
        lel_max= Integer.MIN_VALUE;
        for(i=0; i<size_out; i++)
        {
        	if(lel_max < lel[i])
        	{
        		lel_max = lel[i];
        	}
        }
        
        return lel_max;
	}
}
