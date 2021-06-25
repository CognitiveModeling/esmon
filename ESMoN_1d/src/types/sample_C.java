package types;

public class sample_C
{
	public double[] _in;//vector of input values
	public double[] _out;//vector of output values
	
	/**
	 * create an object of a sample and allocate memory for the input and output vectors without their assignment
	 * @param len_in, length of the input vector
	 * @param len_out, length of the output vector
	 */
	public sample_C(int len_in, int len_out)
	{
		//allocate memory for the input vector, only if its length is larger than 0
		if(len_in > 0)
		{
			_in  = new double[len_in];
		}
		//allocate memory for the output vector, only if its length is larger than 0
		if(len_out > 0)
		{
			_out = new double[len_out];
		}
	}
	
	/**
	 * create an object of a sample and extract input and output vectors from given string
	 * @param str, given string
	 */
	public sample_C(String str)
	{
		int i;
		String[] vectors;//strings of input and output vectors at current time step
		String[] strInputVector;//strings of elements of input vector at current time step 
		String[] strOutputVector;//strings of elements of output vector at current time step
		
		vectors         = str.split("\t");
		strInputVector  = vectors[1].split(",");
		strOutputVector = vectors[2].split(",");
		
		_in  = new double[strInputVector.length];
		_out = new double[strOutputVector.length];
		
		try
		{
			for(i=0; i < _in.length; i++)
			{
				_in[i] = Double.valueOf(strInputVector[i]);
			}
			for(i=0; i < _out.length; i++)
			{
				_out[i] = Double.valueOf(strOutputVector[i]);
			}
		}
		catch (NumberFormatException nfe)
		{
			System.err.println("sample_C: invalid sample value");
			System.exit(1);
		}
	}
	
	/**
	 * The constructor creates a new sample as a copy of the provided sample. 
	 * @param sample: provided sample
	 */
	public sample_C(sample_C sample)
	{
		int i;
		
		_in  = new double[sample.getLenIn()];
		for(i=0; i<_in.length; i++)
		{
			_in[i] = sample._in[i];
		}
		
		_out = new double[sample.getLenOut()];
		for(i=0; i<_out.length; i++)
		{
			_out[i] = sample._out[i];
		}
	}
	
	/**
	 * return a length of the input vector
	 * @return, length of the input vector
	 */
	public int getLenIn()
	{
		return _in.length;
	}
	
	/**
	 * return a length of the output vector
	 * @return, length of the output vector
	 */
	public int getLenOut()
	{
		return _out.length;
	}
}