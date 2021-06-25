package adaptation;

/**
 * This class defines a structure of an individual.
 * @author Danil Koryakin
 */
public class DiffEvolutionIndividual
{
	public double[] genotype;
	public double[] sub_output;
	public DiffEvolutionError error;//error status of an individual

	/**
	 * Class constructor
	 * 
	 * @param genotype_orig: original genotype for initialization of the created individual 
	 * @param sub_output_orig: original module output for initialization of the created individual
	 * @param error_init: first deviations to be stored
	 */
	public DiffEvolutionIndividual(double[] genotype_init, double[] sub_output_init, double[] deviations_init)
	{
		int i;
		int len_genotype;
		int len_sub_output;

		len_genotype   = genotype_init.length;
		len_sub_output = sub_output_init.length;

		genotype   = new double[len_genotype];
		sub_output = new double[sub_output_init.length];

		for(i=0; i<len_genotype; i++)
		{
			genotype[i] = genotype_init[i];
		}
		for(i=0; i<len_sub_output; i++)
		{
			sub_output[i] = sub_output_init[i];
		}
		error = new DiffEvolutionError(sub_output_init.length);
		error.update(deviations_init);
	}
}
