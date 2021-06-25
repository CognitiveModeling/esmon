package MathDiff;

import java.util.Random;

import Jama.Matrix;

/**
 * This class provides methods for simulation of different kinds of noise.
 * 
 * @author Johannes Lohmann, Danil Koryakin
 */
public class MathNoise
{
	/**
	 * Instances of this enumeration are used in the settings of an esn to indicate the
	 * intended type of noise.
	 */
	public enum noise_E
	{
		NOISE_UNIFORM,//uniformly distributed noise
		NOISE_UNIFORM_SYNC,//uniformly distributed synchronous noise
		NOISE_GAUSSIAN,//normally distributed noise
		NOISE_GAUSSIAN_SYNC,//normally distributed synchronous noise
		NOISE_NONE;//no noise should
		
		/**
		 * The function converts the provided string into an instance of the returned enumeration type. The function
		 * tries to match the name of each enumeration instance with the provided string.
		 * 
		 * @param value: provided string
		 * @return instance of the enumeration type, if the match was found; "null", otherwise
		 */
		public static noise_E fromString(String value)
		{
			noise_E match = null;
			
			for (noise_E typ : MathNoise.noise_E.values())
			{
				if (value.equals(typ.name()))
				{
					match = typ;
				}
			}
			
			return match;
		}
	}
	
	/**
	 * This method applies Gaussian noise at the values of a given Matrix. 
	 * @param states: a Matrix object containing the values the error should applied to.
	 * @param mean: the mean of the gaussian distribution.
	 * @param dev: the deviation of the gaussian distribution.
	 * @param noise_type: type of noise (it can be GAUSSIAN or GAUSSIAN_SYNC)
	 * @return states: the Matrix that was assigned to this method now containing the values
	 * with the induced noise.
	 */
	public static Matrix applyGaussianNoise(Matrix states, Number mean, Number dev, noise_E noise_type)
	{
		int i, j;
		double[][] noise = new double[states.getArray().length]
		                             [states.getArray()[0].length];
		Random rand = new Random(0);
		double m;
		double sigma;
		double noise_term;//noise term to be added to each element of the given matrix

		m     = (Double)mean;
		sigma = (Double)dev;
		noise_term = (rand.nextGaussian() * sigma + m);
		for(i=0; i < noise.length; i++)
		{
			for(j = 0; j < noise[i].length; j++)
			{
				switch(noise_type)
				{
					case NOISE_GAUSSIAN:
						noise[i][j] = rand.nextGaussian() * sigma + m;
						break;
					case NOISE_GAUSSIAN_SYNC:
						noise[i][j] = noise_term;
						break;
					default:
						System.err.println("applyGaussianNoise: unexpected noise type");
						System.exit(1);
						break;
				}
			}
		}
		
		return states.plus(new Matrix(noise));
	}
	
	/**
	 * This method applies equal distributed noise at the values of a given Matrix. 
	 * @param states: a Matrix object containing the values the error should applied to.
	 * @param lBound: the lowest possible value for the noise term.
	 * @param uBound: the highest possible value for the noise term.
	 * @param noise_type: type of noise (it can be UNIFORM or UNIFORM_SYNC)
	 * @return states: the Matrix that was assigned to this method now containing the values
	 * with the induced noise.
	 */
	public static Matrix applyUniformNoise(Matrix states, Number lBound, Number uBound, noise_E noise_type)
	{
		int i, j;
		double[][] noise = new double[states.getArray().length]
	                                 [states.getArray()[0].length];
		double upper;
		double lower;
		double noise_term;//noise term to be added to each element of the given matrix
		
		Random rand = new Random(0);
		
		upper = (Double)uBound;
		lower = (Double)lBound;
		noise_term = ((upper - lower) * rand.nextDouble() + lower);
		for(i = 0; i < noise.length; i++)
		{
			for(j = 0; j < noise[i].length; j++)
			{
				switch(noise_type)
				{
					case NOISE_UNIFORM:
						noise[i][j] = ((upper - lower) * rand.nextDouble() + lower);
						break;
					case NOISE_UNIFORM_SYNC:
						noise[i][j] = noise_term;
						break;
					default:
						System.err.println("applyUniformNoise: unexpected noise type");
						System.exit(1);
						break;
				}
			}
		}
		
		return states.plus(new Matrix(noise));
	}
}
