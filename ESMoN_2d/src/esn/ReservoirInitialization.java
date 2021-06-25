package esn;

import java.util.Random;

import Jama.Matrix;

public enum ReservoirInitialization {
	/**
	 * Indicates the default random initialization with a defined connectivity between zero
	 * and one.
	 */
	RANDOM,
	/**
	 * Indicates that the dynamic reservoir should be structured according to a defined
	 * weight matrix.
	 */
	EXPLICIT,
	/**
	 * Indicates a delay line reservoir. Units are organized in one line.
	 */
	DLR,
	/**
	 * Indicates a DLR with feedback-connections.
	 */
	DLRB,
	/**
	 * Indicates a simple cyclic reservoir.
	 */
	SCR,
	SRR;//self-recurrent reservoir (only self-recurrent connections are present at all its neurons)
	
	public static ReservoirInitialization fromString(String value)
	{
		for (ReservoirInitialization ri : ReservoirInitialization.values())
		{
			if (ri.name().equals(value))
			{
				return ri;
			}
		}
		
		return ReservoirInitialization.RANDOM;
	}
	
	public static Matrix initializeDLR (double[] bounds, int seedInternal, int rows, int cols) {
		double[][] matrix = new double[rows][cols];
		
		Random rand = new Random(seedInternal);
		
		//The internal units of a delay line reservoir are organized in one line. Hence only values on the lower sub-diagonal are different from zero.
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++)
			{
				if (i - 1 == j) {
						matrix[i][j] = bounds[0] + Math.abs(bounds[1] - bounds[0]) * rand.nextDouble();
				} else {
					matrix[i][j] = 0.0D;
				}
				
			}
		}
		
		return new Matrix(matrix);
	}
	
	public static Matrix initializeDLRB (double[] bounds, int seedInternal, int rows, int cols) {
		double[][] matrix = new double[rows][cols];
		
		Random rand = new Random(seedInternal);
		
		//The internal units of a delay line reservoir with feedback are organized in one line. Hence only values on the lower and the upper sub-diagonal are different from zero.
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++)
			{
				if (i - 1 == j) {
						matrix[i][j] = bounds[0] + Math.abs(bounds[1] - bounds[0]) * rand.nextDouble();
				} else if (i + 1 == j) {
						matrix[i][j] = bounds[0] + Math.abs(bounds[1] - bounds[0]) * rand.nextDouble();
				} else {
					matrix[i][j] = 0.0D;
				}
				
			}
		}
		
		return new Matrix(matrix);
	}
	
	public static Matrix initializeSCR (double[] bounds, int seedInternal, int rows, int cols) {
		double[][] matrix = new double[rows][cols];
		
		Random rand = new Random(seedInternal);
		
		//The internal units of a delay line reservoir with feedback are organized in one line. Hence only values on the lower and the upper sub-diagonal are different from zero.
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++)
			{
				if (i - 1 == j) {
						matrix[i][j] = bounds[0] + Math.abs(bounds[1] - bounds[0]) * rand.nextDouble();
				} else if (i == 0 && j == cols - 1) {
						matrix[i][j] = bounds[0] + Math.abs(bounds[1] - bounds[0]) * rand.nextDouble();
				} else {
					matrix[i][j] = 0.0D;
				}
				
			}
		}
		
		return new Matrix(matrix);
	}
	
	/**
	 * create a reservoir with only self-recurrent connections at its neurons
	 * @param bounds, interval of random weights assigned to the reservoir's connections
	 * @param seedInternal, value for seeding the generator of random numbers
	 * @param rows, number of rows of the connectivity matrix
	 * @param cols, number of columns of the connectivity matrx
	 * @return, connectivity matrix of the created reservoir
	 */
	public static Matrix initializeSRR (double[] bounds, int seedInternal, int rows, int cols)
	{
		int i, j;
		double[][] matrix = new double[rows][cols];
		
		Random rand = new Random(seedInternal);
		
		//weights of the self-recurrent connections are on the main diagonal of the connectivity matrix
		for(i=0; i < matrix.length; i++)
		{
			for(j=0; j < matrix[i].length; j++)
			{
				if(i == j)
				{
					matrix[i][j] = bounds[0] + Math.abs(bounds[1] - bounds[0]) * rand.nextDouble();
				}
				else
				{
					matrix[i][j] = 0.0D;
				}
				
			}
		}
		
		return new Matrix(matrix);
	}
}
