package adaptation;

public class DiffEvolutionGenotype
{
	public int _time_step;//index of a time step where a genotype was created
	public double[][] _individual;//individuals of a genotype
	public double[][] _sub_output;//module outputs of individuals of a genotype
	public DiffEvolutionError _error;//error status of a genotype
	public DiffEvolutionError _error_config;//configuration error which a genotype had as it was stored
	
	public DiffEvolutionGenotype(double[][] individual, double[][] sub_output,
			                     DiffEvolutionError error_config, int time_step)
	{
		int i, j;
		
		_time_step = time_step;
		_individual = new double[individual.length][];
		_sub_output = new double[sub_output.length][];
		
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
		_error = new DiffEvolutionError();
		_error.copy(error_config);

		_error_config = new DiffEvolutionError();
		_error_config.copy(error_config);
	}
}
