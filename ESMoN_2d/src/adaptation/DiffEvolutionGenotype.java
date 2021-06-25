package adaptation;

import java.util.Vector;

public class DiffEvolutionGenotype
{
	/**
	 * The class serves as a container for the ESN output at a single time step.
	 *  
	 * @author Danil
	 */
	public class esn_output_C
	{
		double[] _output;
		
		/**
		 * The constructor creates an object of the class to store the provided array as the ESN output.
		 * 
		 * @param output: provided array
		 */
		public esn_output_C(double[] output)
		{
			_output = output.clone();
		}
	};
	
	public int _time_step;//index of a time step where a genotype was created
	public double[][] _individual;//individuals of a genotype
	public double[][] _sub_output;//module outputs of individuals of a genotype
	public DiffEvolutionError _error;//error status of a genotype
	public DiffEvolutionError _error_config;//configuration error which a genotype had as it was stored
	public Vector<esn_output_C> _esn_output_history;//history of the total EPOS output over the whole life time
	
	public DiffEvolutionGenotype(double[][] individual, double[][] sub_output,
			                     DiffEvolutionError error_config, int time_step)
	{
		int i, j;
		
		_time_step = time_step;
		_individual = new double[individual.length][];
		_sub_output = new double[sub_output.length][];
		_esn_output_history = new Vector<esn_output_C>(0, 1);
		
		for(i=0; i<_individual.length; i++)
		{
			_individual[i] = new double[individual[i].length];
			for(j=0; j<_individual[i].length; j++)
			{
				_individual[i][j] = individual[i][j];
			}
		}
		for(i=0; i<_sub_output.length; i++)
		{
			_sub_output[i] = new double[sub_output[i].length];
			for(j=0; j<_sub_output[i].length; j++)
			{
				_sub_output[i][j] = sub_output[i][j];
			}
		}
		_error = new DiffEvolutionError(sub_output[0].length);
		_error.copy(error_config);

		_error_config = new DiffEvolutionError(sub_output[0].length);
		_error_config.copy(error_config);
	}
	
	/**
	 * The function stores the provided EPOS output as the last element in the EPOS output history.
	 * 
	 * @param epos_output_array: provided EPOS output
	 */
	public void storeCurrentEposOutput(double[] epos_output_array)
	{
		esn_output_C epos_output_obj;
		
		epos_output_obj = new esn_output_C(epos_output_array);
		_esn_output_history.add(epos_output_obj);
	}
}
