package MathDiff;

import types.sample_C;

/**
 * This class implements functions for computing performance of the echo-state networks.
 * @author Danil Koryakin
 *
 */
public class MathPerformance
{
	/**
	 * compute the mean squared error (MSE) for the provided sequence; MSE is computed for each element of the output
	 * vector
	 * @param output, sequence of the network output vectors which correspond to the samples of the provided sequence 
	 * @param target, provided sequence
	 * @return MSE for every element of the output vector 
	 */
	public static double[] computeMSE(double[][] output, sample_C[] target)
	{
		int i, j;
		int size_out;//length of the output vector
		double sum;
		double mse[];//computed error to be returned
		
		size_out = target[0]._out.length;
		
		//allocate an array for saving the MSE for each element of the output vector
		mse = new double[size_out];
		
		for(i=0; i < size_out; i++)
		{
			sum = 0.0;
			for(j=0; j < output.length; j++)
			{
				sum += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}
			mse[i] = sum / output.length;
		}		
		
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
		double mse;//computed error to be returned
		
		size_out = target[0]._out.length;

		mse = 0;
		for(i=0; i < size_out; i++)
		{
			for(j=0; j < output.length; j++)
			{
				mse += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}			
		}
		mse = mse / output.length;
		
		return mse;
	}
	
	/**
	 * calculate the normalized root mean squared error (NRMSE) for the actual and target outputs;
	 * NRMSE is computed according to formula (33) in "roeschies09" 
	 * @param output, sequence of activation values of the output nodes
	 * @param target, sequence of target values which should be predicted by ESN
	 * @param variance, array of variances for each element of the target vector
	 * @return NRMSE for every element of the output vector
	 */
	public static double[] computeNRMSE(double[][] output,
			                            sample_C[] target,
			                            double[]   variance)
	{
		int i, j;
		double sum;//sum of one element over all samples
		double nrmse[];//computed error to be returned
		
		//allocate an array of errors for all elements of the output vector
		nrmse = new double[variance.length];
		
		for(i=0; i < variance.length; i++)
		{	
			sum = 0.0;
			//go over all samples
			for(j=0; j < output.length; j++)
			{
				sum += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}
			sum /= variance[i];
			sum /= output.length;
			nrmse[i] = Math.sqrt(sum);
		}
		
		return nrmse;
	}
	
	/**
	 * calculate the root mean squared error (RMSE) for the actual and target outputs; 
	 * @param output: sequence of activation values of the output nodes
	 * @param target: sequence of target values which should be predicted by ESN
	 * @return RMSE for every element of the output vector
	 */
	public static double[] computeRMSE(double[][] output,
			                           sample_C[] target)
	{
		int i, j;
		double sum;//sum of one element over all samples
		double rmse[];//computed error to be returned
		
		//allocate an array of errors for all elements of the output vector
		rmse = new double[output.length];
		
		//"[0]" because all output vectors have the same dimensionality
		for(i=0; i < output[0].length; i++)
		{	
			sum = 0.0;
			//go over all samples
			for(j=0; j < output.length; j++)
			{
				sum += Math.pow((target[j]._out[i] - output[j][i]), 2.0);
			}
			sum /= output.length;
			rmse[i] = Math.sqrt(sum);
		}
		
		return rmse;
	}
	
	/**
	 * compute the small error length (SEL) of the network output for the provided sequence; SEL is counted for each
	 * element of the output vector;
	 * SEL is a number of time steps where a deviation of the network output from the corresponding target value stays
	 * within the specified error threshold
	 * @param output, sequence of the network output vectors which correspond to the samples of the provided sequence
	 * @param target, sequence of target values which should be predicted by ESN
	 * @param err_thresh, error threshold
	 * @return SEL for each element of the output vector
	 */
	public static double[] computeSEL(double[][] output,
			                          sample_C[] target,
			                          double     err_thresh)
	{
		int i, j;
		int size_out;//length of the output vector
		double[] sel;//computed performance indicator for all elements of the output vector
		double dev;//deviation at the current sample
		boolean f_measure_done;//indicator that at the current sample the deviation exceeds the threshold
		
		size_out = target[0]._out.length;
		
		//allocate an array for saving the performance indicator for all elements of the output vector
		sel = new double[size_out];
		
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
        
        return sel;
	}
	
	/**
	 * The function computes the large error length. The large error length is a number of time steps where
	 * the deviation of the ESN output from its target dynamics stays larger than the provided threshold. The steps are
	 * always counted from the beginning of the sequence. After the last counted time step the largest deviation is
	 * lower than the provided threshold.
	 * The large error length is counted for each output neuron independently on the other output neurons.
	 * The large error length is an indicator which can be used to show a performance of the considered configuration
	 * method. Normally configuration methods should steadily reduce an initially large deviation.
	 * @param output: sequence of the output neurons' states which correspond to the samples of the target sequence
	 * @param target: provided target sequence
	 * @param lel_thresh: provided threshold of the large error length
	 * @return LEL for each output neuron
	 *         (output is of "double" because this a performance indicator which is stored as "double")
	 */
	public static double[] computeLEL(double[][] output,
			                          sample_C[] target,
			                          double     lel_thresh)
	{
		int i, j;
		int size_out;//number of output neurons
		double[] lel;//output array: computed large error length for each output neuron
		double dev;//deviation at the current sample
		boolean f_measure_done;//indicator that at the current sample the deviation exceeds the threshold
		
		size_out = target[0]._out.length;
		
		//allocate the output array
		lel = new double[size_out];
		
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
        
        return lel;
	}
}
