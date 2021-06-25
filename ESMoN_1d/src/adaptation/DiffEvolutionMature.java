package adaptation;

import java.util.Vector;

import esn.Module.layer_type_E;
import esn.mESN;

public class DiffEvolutionMature
{
	private int _best_idx;//index of the best individual in the pool
	private int _time_best;//index of a time step where last assigned individual was previously saved
	private int _min_life_span;//minimum life-span of a genotype to be chosen as the best one
	private Vector<DiffEvolutionGenotype> _pool_of_bests;//pool of the best individuals found so far
	private mESN _esn;//optimized ESN. It is needed to get an access to a function for computing an error.
	
	private final int _size_pool_of_bests = 50;//size of a pool of the best individuals
	
	/**
	 * Class constructor
	 * 
	 * @param esn: object with an ESN to be synchronized
	 * @param fit_len: fitness length
	 * @param valid_len: validation length
	 */
	public DiffEvolutionMature(mESN esn, int fit_len, int valid_len)
	{
		_esn = esn;
		_pool_of_bests = new Vector<DiffEvolutionGenotype>(0, 1);
		_best_idx  = -1;
		_time_best = -1;
		_min_life_span = fit_len + valid_len;//life span is measured by a length of error history;
		                                     //since error window from evolution is accepted and continued
		                                     //in the maturity pool, the minimum life span is a fitness length plus
		                                     //a number of steps how long a genotype stays in the maturity pool
	}
	
	/**
	 * The function search for the best individual in the maturity pool.
	 */
	private void findBest()
	{
		int i;
		int size_mature_pool;
		int idx_current;//index of a currently checked genotype
		int num_errors;//number of errors of the current chromosome
		double min_error;//smallest error among the oldest genotypes
		Vector<Integer> idx_oldest;
		
		size_mature_pool = _pool_of_bests.size();
		
		idx_oldest = new Vector<Integer>(0, 1);
		
		//find all oldest genotypes
		for(i=0; i<size_mature_pool; i++)
		{
			num_errors = _pool_of_bests.get(i)._error.all_single_squares.size();
			if(num_errors>=_min_life_span)
			{
				idx_oldest.add(i);
			}
		}
		
		//find the best oldest gentypes with a span above a threshold
		_best_idx = -1;
		if(idx_oldest.size() > 0)
		{
			_best_idx = idx_oldest.get(0);
			min_error = _pool_of_bests.get(_best_idx)._error.getAverageError();
			for(i=1; i<idx_oldest.size(); i++)
			{
				idx_current = idx_oldest.get(i);
				if(min_error > _pool_of_bests.get(idx_current)._error.getAverageError())
				{
					_best_idx = idx_current;
					min_error = _pool_of_bests.get(_best_idx)._error.getAverageError();
				}
			}
		}
	}
	
	/**
	 * The function removes the worst individual from a pool of the best individuals.
	 * The worst individual is chosen after comparison of the oldest individual with the second oldest one.
	 * The other individual is marked as the best one.
	 * Only these two individuals are compared because they have the longest history.
	 */
	private void removeInvalidFromPool()
	{
		int i;
		int size_mature_pool;
		int idx_1st_oldest;//index of the oldest individual
		int idx_2nd_oldest;//index of the 2nd oldest individual
		int idx_to_remove;//index of an chromosome which has to be removed
		int idx_current;//index of currently checked genotype
		int num_errors;//number of errors of the current chromosome
		int max_num_errors;//maximum numbers of stored errors among checked chromosomes
		double error_1st_oldest;//average error of the oldest individual
		double error_2nd_oldest;//average error of the 2nd oldest individual
		double max_error;//largest error among the oldest individuals
		Vector<Integer> idx_oldest;//indices of all chromosomes with a life span larger than a predefined threshold
		
		size_mature_pool = _pool_of_bests.size();
		
		//if the maturity pool is full and none of the genotypes could be removed above because of module outputs
		//then remove one of the oldest individuals according to the errors
		if(_pool_of_bests.size()==_size_pool_of_bests)
		{
			idx_oldest = new Vector<Integer>(0, 1);
			
			//find all oldest genotypes
			for(i=0; i<size_mature_pool; i++)
			{
				num_errors = _pool_of_bests.get(i)._error.all_single_squares.size();
				if(num_errors>=_min_life_span)
				{
					idx_oldest.add(i);
				}
			}
			
			//1) find the worst oldest gentypes with a span above a threshold
			//2) do not remove a single genotype without comparison;
			//   it could have very high performance
			if(idx_oldest.size() > 1)
			{
				idx_to_remove = idx_oldest.get(0);
				max_error = _pool_of_bests.get(idx_to_remove)._error.getAverageError();
				for(i=idx_to_remove+1; i<idx_oldest.size(); i++)
				{
					idx_current = idx_oldest.get(i);
					if(max_error < _pool_of_bests.get(idx_current)._error.getAverageError())
					{
						idx_to_remove = idx_current;
						max_error = _pool_of_bests.get(idx_to_remove)._error.getAverageError();
					}
				}
			}
			else//remove the worst of two oldest genotypes if none of genotypes had span above the threshold
			{
				//find the oldest and 2nd oldest individuals
				idx_1st_oldest = -1;
				max_num_errors = 0;
				for(i=0; i<size_mature_pool; i++)
				{
					//"[0]" because the number of errors is equal for all sub-individuals of the same individual
					num_errors = _pool_of_bests.get(i)._error.all_single_squares.size();
					if(max_num_errors < num_errors)
					{
						max_num_errors = num_errors;
						idx_1st_oldest = i;
					}
				}
				//find the 2nd oldest individuals
				idx_2nd_oldest = -1;
				if(size_mature_pool > 1)//the 2nd oldest individual exists if there are 2 or more individuals
				{
					max_num_errors = 0;
					for(i=0; i<size_mature_pool; i++)
					{
						if(i!=idx_1st_oldest)//the oldest individual should be ignored
						{
							//"[0]" because the number of errors is equal for all sub-individuals of the same individual
							num_errors = _pool_of_bests.get(i)._error.all_single_squares.size();
							if(max_num_errors < num_errors)
							{
								max_num_errors = num_errors;
								idx_2nd_oldest = i;
							}
						}
					}
				}

				//compare performances of both oldest genotypes using their average errors;
				error_1st_oldest = _pool_of_bests.get(idx_1st_oldest)._error.getAverageError();
				if(idx_2nd_oldest!=-1)
				{
					error_2nd_oldest = _pool_of_bests.get(idx_2nd_oldest)._error.getAverageError();

					if(error_1st_oldest < error_2nd_oldest)
					{
						idx_to_remove = idx_2nd_oldest;
					}
					else
					{
						idx_to_remove = idx_1st_oldest;
					}
				}
				else//a single individual must always be replaced
				{
					idx_to_remove = idx_1st_oldest;
				}
			}//are there genotypes with life span above a threshold?
			
			_pool_of_bests.remove(idx_to_remove);
		}//is maturity pool full?
	}
	
	/**
	 * The function advances existing genotypes using the provided sample.
	 * The function also updates an error status of all individuals in the pool.
	 * 
	 * @param sample_in: input values of the provided sample
	 * @param sample_out: output values of the provided sample
	 */
	public void advance(double[] sample_in, double[] sample_out)
	{
		int i,j;
		int num_sub;
		int pool_size;
		double[] deviations;
		
		pool_size = _pool_of_bests.size();
		if(pool_size > 0)
		{
			num_sub = _pool_of_bests.get(0)._individual.length;
			for(i=0; i<pool_size; i++)
			{
				//assign states to be advanced
				for(j=0; j<num_sub; j++)
				{					
					_esn.configDiffEvolutionDecodeIndividual(j, _pool_of_bests.get(i)._individual[j]);
					_esn.setModuleNodes(j, layer_type_E.LT_OUTPUT, _pool_of_bests.get(i)._sub_output[j]);
				}
				
				//it is enough to compute mESN outputs only when the best genotype is restored
				_esn.advance(sample_in, false);
				_esn.calculateEsnOutput();
				
				//store an updated individual and its module output
				for(j=0; j<num_sub; j++)
				{
					_esn.configDiffEvolutionEncodeIndividual(j, _pool_of_bests.get(i)._individual[j]);
					_esn.getModuleNodes(j, layer_type_E.LT_OUTPUT, _pool_of_bests.get(i)._sub_output[j]);
				}
				
				deviations = _esn.computeError(sample_out);
				//update errors of the genotypes
				_pool_of_bests.get(i)._error.update(deviations);
			}
		}
	}
	
	/**
	 * The function stores an array with provided individuals as a new genotype in the maturity pool.
	 * The provided individuals, their module outputs and an error are used for initialization of the genotype.
	 * If the maturity pool is already full then the worst mature genotype is removed to get place for a new one.
	 * 
	 * @param individual: array to be stored as a genotype of the best individual
	 * @param sub_output: array to be stored as a module output of the best individual
	 * @param error_config: error that a submitted genotype had when it must be stored
	 * @param time_step: time step where the stored individuals were obtained
	 */
	public void storeBestIndividual(double[][] individual, double[][] sub_output, DiffEvolutionError error_config, int time_step)
	{
		DiffEvolutionGenotype new_best;//new best individual

		new_best = new DiffEvolutionGenotype(individual, sub_output, error_config, time_step);
		
		//remove genotypes whose module output are invalid
		removeInvalidFromPool();
		_pool_of_bests.add(new_best);
	}
	
	/**
	 * The function assigns ESNs states the previously found best genotype.
	 * If no best individual was previously found then ESN keeps states that it had before this function.
	 * 
	 * @return: "true" if new states were assigned; "false" otherwise
	 */
	public boolean restoreBestIndividual()
	{
		int i;
		int num_sub;
		boolean is_assigned;//output variable
		
		//search for the best genotype
		findBest();
		
		if(_best_idx!=-1)
		{
			_time_best = _pool_of_bests.get(_best_idx)._time_step;
			num_sub = _pool_of_bests.get(_best_idx)._individual.length;
			for(i=0; i<num_sub; i++)
			{
				_esn.configDiffEvolutionDecodeIndividual(i, _pool_of_bests.get(_best_idx)._individual[i]);
				_esn.setModuleNodes(i, layer_type_E.LT_OUTPUT, _pool_of_bests.get(_best_idx)._sub_output[i]);
			}
			_esn.calculateEsnOutput();
			is_assigned = true;
		}
		else
		{
			is_assigned = false;
		}
		return is_assigned;
	}
	
	/**
	 * The function returns an index of a time step where the best previously assigned genotype was obtained.
	 * For reasons of compatibility, the function returns the index as an array where the time step is copied
	 *    in each element of the array.
	 * 
	 * @return: array where each element contains the same value of the best previously assigned genotype
	 *          (output is of "double" because this a performance indicator which is stored as "double")
	 */
	public double[] getTimeBest()
	{
		int i;
		double[] time_step; 

		time_step = new double[_esn.getNumOutputNeurons()];//create an element for each output neuron
		for(i=0; i<time_step.length; i++)
		{
			time_step[i] = _time_best;
		}
		
		return time_step;
	}
	
	/**
	 * The function returns an average configuration MSE of the best genotype.
	 * For reasons of compatibility, the function returns the index as an array where the average error is copied
	 *    in each element of the array.
	 * 
	 * @return array of the configuration MSE for each output element
	 */
	public double[] getConfigMse()
	{
		int i;
		double[] avg_error; 
		
		avg_error = new double[_esn.getNumOutputNeurons()];
		if(_best_idx!=-1)
		{
			for(i=0; i<avg_error.length; i++)
			{
				avg_error[i] = _pool_of_bests.get(_best_idx)._error_config.avg_mse;
			}
		}
		else
		{
			for(i=0; i<avg_error.length; i++)
			{
				avg_error[i] = Double.MAX_VALUE;
			}
		}
		
		return avg_error;
	}
	
	/**
	 * The function returns an average configuration RMSE of the best genotype.
	 * For reasons of compatibility, the function returns the index as an array where the average error is copied
	 *    in each element of the array.
	 * 
	 * @return array of the configuration RMSE for each output element
	 */
	public double[] getConfigRmse()
	{
		int i;
		double[] avg_error; 
		
		avg_error = new double[_esn.getNumOutputNeurons()];
		if(_best_idx!=-1)
		{
			for(i=0; i<avg_error.length; i++)
			{
				avg_error[i] = _pool_of_bests.get(_best_idx)._error_config.computeRmse();
			}
		}
		else
		{
			for(i=0; i<avg_error.length; i++)
			{
				avg_error[i] = Double.MAX_VALUE;
			}
		}
		
		return avg_error;
	}
	
	/**
	 * The function returns an average configuration NRMSE of the best genotype.
	 * For reasons of compatibility, the function returns the index as an array where the average error is copied
	 *    in each element of the array.
	 * 
	 * @param variance: array of variances for each element of the target vector
	 * @return array of the configuration NRMSE for each output element
	 */
	public double[] getConfigNrmse(double[] variance)
	{
		int i;
		double[] avg_error; 
		
		avg_error = new double[_esn.getNumOutputNeurons()];
		if(_best_idx!=-1)
		{
			for(i=0; i<avg_error.length; i++)
			{
				avg_error[i] = _pool_of_bests.get(_best_idx)._error_config.computeNrmse(variance);
			}
		}
		else
		{
			for(i=0; i<avg_error.length; i++)
			{
				avg_error[i] = Double.MAX_VALUE;
			}
		}
		
		return avg_error;
	}
}
