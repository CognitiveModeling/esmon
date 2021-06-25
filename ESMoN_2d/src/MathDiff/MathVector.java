package MathDiff;

/**
 * This class provides functions for operations on vectors.
 * @author danil
 */
public class MathVector
{
	/**
	 * The function computes a vector whose elements are average values of the corresponding elements of the provided
	 * vectors.
	 * @param vect1: 1st provided vector
	 * @param vect2: 2nd provided vector
	 * @return: vector of averages
	 */
	public static double[] computeAverageVect(double[] vect1, double[] vect2)
	{
		int      i;
		double[] average;//output array of the averaged vector
		
		//check the input parameters
		if(vect1.length != vect2.length)
		{
			System.err.println("computeDist: provided vectors have different lengths");
			System.exit(1);
		}
		
		average = new double[vect1.length];
		for(i=0; i<vect1.length; i++)
		{
			average[i] = (vect1[i] + vect2[i])/2;
		}
		
		return average;
	}
	
	/**
	 * The function computes a distance between the provided vectors.
	 * @param vect1: 1st provided vector
	 * @param vect2: 2nd provided vector
	 * @return distance between the provided vectors
	 */
	public static double computeDist(double[] vect1, double[] vect2)
	{
		int    i;
		double diff;//difference between two double values
		double dist;//distance (output variable)
		
		//check the input parameters
		if(vect1.length != vect2.length)
		{
			System.err.println("computeDist: provided vectors have different lengths");
			System.exit(1);
		}
		
		dist = 0;
		for(i=0; i<vect1.length; i++)
		{
			diff  = vect1[i] - vect2[i];
			dist += Math.pow(diff, 2);
		}
		dist = Math.sqrt(dist);
		
		return dist;
	}
	
	/**
	 * The function normalizes elements of the provided vector to the largest of them.
	 * 
	 * @param vect: provided vector
	 */
	public static void normalizeToMax(double[] vect)
	{
		int i;
		double max_val;
		
		//find the largest value
		max_val = 0;
		for(i=0; i<vect.length; i++)
		{
			if(max_val < Math.abs(vect[i]))
			{
				max_val = Math.abs(vect[i]);
			}
		}
		
		if(max_val==0)
		{
			System.err.println("normalizeToMax: all elements of the provided vector are 0");
			System.exit(1);
		}
		
		//normalize vector's elements
		for(i=0; i<vect.length; i++)
		{
			vect[i] /= max_val;
		}
	}
}
