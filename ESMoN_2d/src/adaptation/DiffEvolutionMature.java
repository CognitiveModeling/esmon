package adaptation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

import adaptation.DiffEvolutionGenotype.esn_output_C;

import esn.Module.layer_type_E;
import esn.mESN;

public class DiffEvolutionMature
{
	private int _best_idx;//index of the best individual in the pool
	private int _best_idx_previous;//index of the previously found best genotype
	                               //(it cannot be removed if there was no new best genotype)
	private int _time_best;//index of a time step where last assigned individual was previously saved
	private int _min_life_span;//minimum life-span of a genotype to be chosen as the best one
	private double _max_best_error;//largest acceptable error for the found best individual
	private Vector<DiffEvolutionGenotype> _pool_of_bests;//pool of the best individuals found so far
	private mESN _esn;//optimized ESN. It is needed to get an access to a function for computing an error.
	
	private final int _size_pool_of_bests = 50;//size of a pool of the best individuals
	
	/**
	 * Class constructor
	 * 
	 * @param esn: object with an ESN to be synchronized
	 * @param fit_len: fitness length
	 * @param valid_len: validation length
	 * @param max_best_error: largest acceptable error for the found best individual
	 */
	public DiffEvolutionMature(mESN esn, int fit_len, int valid_len, double max_best_error)
	{
		_esn = esn;
		_pool_of_bests = new Vector<DiffEvolutionGenotype>(0, 1);
		_best_idx  = -1;
		_best_idx_previous = -1;
		_time_best = -1;
		_min_life_span = fit_len + valid_len;//life span is measured by a length of error history;
		                                     //since error window from evolution is accepted and continued
		                                     //in the maturity pool, the minimum life span is a fitness length plus
		                                     //a number of steps how long a genotype stays in the maturity pool
		_max_best_error = max_best_error;
	}
	
	/**
	 * The function search for the best genotype in the maturity pool.
	 * The best genotype is the one which has the smallest average error among genotypes with life-time larger
	 * than the validation length.
	 */
	private void findBest()
	{
		int i;
		int size_mature_pool;
		int num_errors;//number of errors of the current chromosome
		double min_error;//smallest error
		double error_of_oldest;//error of the oldest genotype
		
		size_mature_pool = _pool_of_bests.size();
		
		//check all genotypes in the pool until one with a suitable error has been found
		_best_idx = -1;
		min_error = Double.MAX_VALUE;
		for(i=0; i<size_mature_pool; i++)
		{
			num_errors = _pool_of_bests.get(i)._error.all_single_squares.size();
			if(num_errors >= _min_life_span)
			{
				error_of_oldest = _pool_of_bests.get(i)._error.getMaxAverageMse();
				if(min_error > error_of_oldest)
				{
					_best_idx = i;
					_best_idx_previous = _best_idx;
					min_error = error_of_oldest;
				}
			}
		}
	}
	
	/**
	 * The function removes the worst genotype from the maturity pool. It must have the error above
	 *    the largest allowed error threshold "_max_best_error".
	 * The removed genotype must have the longest life. 
	 * (see a reason for this in the short remark to implementation of the version
	 *  "RC_Experiment063_20180624_correctMatureRemoval")
	 */
	private void removeInvalidFromPool()
	{
		int i;
		int size_mature_pool;
		int idx_to_remove;//index of an chromosome which has to be removed
		DiffEvolutionError error_cur;//object of the current error
		double[] max_error;
		
		size_mature_pool = _pool_of_bests.size();
		
		//if the maturity pool is full and none of the genotypes could be removed above because of module outputs
		//then remove one of the oldest individuals according to the errors
		if(_pool_of_bests.size()==_size_pool_of_bests)
		{
			//simply take the worst genotype if all have errors below the threshold
			if(_best_idx_previous!=0)
			{
				idx_to_remove = 0;
				max_error = _pool_of_bests.get(idx_to_remove)._error.getAverageError();
			}
			else
			{
				idx_to_remove = 1;
				max_error = _pool_of_bests.get(idx_to_remove)._error.getAverageError();
			}
			for(i=0; i<size_mature_pool; i++)
			{
				if(i!=_best_idx_previous)//previously found best genotype cannot be removed
				{
					error_cur = _pool_of_bests.get(i)._error;
					if(error_cur.isWorse(max_error)==true)
					{
						max_error = error_cur.getAverageError();
						idx_to_remove = i;
					}
				}
			}
			
			//check consistency
			if(idx_to_remove==_best_idx_previous)
			{
				System.err.println("DiffEvolutionMature.findBest: best previous genotype cannot be removed");
				System.exit(1);
			}
			else if(idx_to_remove < _best_idx_previous)//correct index of the previous best genotype
			{
				_best_idx_previous--;
			}
			else
			{
				//everything is fine
			}
			
			_pool_of_bests.remove(idx_to_remove);
		}//is maturity pool full?
	}
	
	/**
	 * The function saves the output of the best genotype from its whole life time.
	 * 
	 * @param time_step: current time step
	 */
	private void saveOutputBestGenotype(int time_step)
	{
		int i, j;
		int size_out;
		int life_time;
		int curr_time_step;
		int first_time_step;
		double[] avg_error;
		File     file;
		String   str_out;//string for data output
		String   filename;//name of output file
		DiffEvolutionGenotype best_genotype;
		esn_output_C epos_output;//EPOS output at the current time step
		
		best_genotype = _pool_of_bests.get(_best_idx);
		try {
			
			filename = "EPOS_output_best_genotype_time_"+ time_step + ".dat";
			
			file = new File("./output/", filename);
			
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			
			//check consistency of the data
			if(best_genotype._error.all_single_squares.size() != best_genotype._esn_output_history.size())
			{
				life_time = 0;
				System.err.println("DiffEvolutionMature.saveOutputBestGenotype: inconsistent data");
				System.exit(1);
			}
			else
			{
				life_time = best_genotype._error.all_single_squares.size();
			}
			first_time_step = time_step - life_time + 1;
			
			str_out = "#current time step: " + time_step;
			str_out += "\n";
			bw.write(str_out);
			str_out = "#life time: " + life_time;
			str_out += "\n";
			bw.write(str_out);
			str_out = "#average config error:";
			str_out += "\n";
			bw.write(str_out);
			avg_error = best_genotype._error_config.getAverageError();
			for(i=0; i<avg_error.length; i++)
			{
				str_out = "#   output " + i + ": " + avg_error[i];
				str_out += "\n";
				bw.write(str_out);
			}
			str_out = "#average error over life time:";
			str_out += "\n";
			bw.write(str_out);
			avg_error = best_genotype._error.getAverageError();
			for(i=0; i<avg_error.length; i++)
			{
				str_out = "#   output " + i + ": " + avg_error[i];
				str_out += "\n";
				bw.write(str_out);
			}
			str_out = "\n\n";
			bw.write(str_out);
			
			//write headers of the columns
			str_out = "time step";
			size_out = best_genotype._esn_output_history.get(0)._output.length;
			curr_time_step = first_time_step;
			for(i=0; i<life_time; i++)
			{
				epos_output = best_genotype._esn_output_history.get(i);
				str_out = "";
				str_out += curr_time_step;
				str_out += "\t";
				for(j=0; j<size_out; j++)
				{
					str_out += epos_output._output[j];
					str_out += "\t";
				}
				str_out += "\n";
				bw.write(str_out);
				
				curr_time_step++;
			}
			
			bw.close();
			bw = null;
			//assign a copy "ReadOnly" access
            file.setReadOnly();
		} catch(FileNotFoundException fnf) {
			fnf.printStackTrace();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
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
		double[] epos_output;
		
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
				//"sample_in = null" because the input vector is a product of synchronization
				_esn.advance(null, false);
				_esn.calculateEsnOutput();
				
				epos_output = _esn.getOutput();
				_pool_of_bests.get(i).storeCurrentEposOutput(epos_output);
				
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
	 * @param sub_output_history: array of the module's output
	 */
	public void storeBestIndividual(double[][] individual, double[][] sub_output, DiffEvolutionError error_config, int time_step, double[][][] sub_output_history)
	{
		int i,j;
		double[] epos_output;
		DiffEvolutionGenotype new_best;//new best individual		
		
		new_best = new DiffEvolutionGenotype(individual, sub_output, error_config, time_step);
		
		//go over the life time
		for(i=0; i<sub_output_history[0].length; i++)
		{
			for(j=0; j<sub_output_history.length; j++)
			{
				//assign the module responsibility and the module output to be used for computing the output of the whole EPOS
				_esn.configDiffEvolutionDecodeIndividual(j, new_best._individual[j]);
				_esn.setModuleNodes(j, layer_type_E.LT_OUTPUT, sub_output_history[j][i]);
				
				_esn.calculateEsnOutput();
				
				epos_output = _esn.getOutput();
				new_best.storeCurrentEposOutput(epos_output);
			}
		}
		
		//remove genotypes whose module output are invalid
		removeInvalidFromPool();
		_pool_of_bests.add(new_best);
	}
	
	/**
	 * The function assigns ESNs states the previously found best genotype.
	 * If no best individual was previously found then ESN keeps states that it had before this function.
	 * The function also saves the output of the best genotype for the current time step. The latter is specified
	 * as an input parameter.
	 * 
	 * @param time_step: current time step
	 * @return: "true" if new states were assigned; "false" otherwise
	 */
	public boolean restoreBestIndividual(int time_step)
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
			
			//save the output of the best genotype from its whole life time
			//commented out because of a big run of 02.08.2018 saveOutputBestGenotype(time_step);
		}
		else
		{
			is_assigned = false;
		}
		return is_assigned;
	}
	
	/**
	 * The function returns an index of a time step where the best previously assigned genotype was obtained.
	 * 
	 * @return: index of a time step where the best previously assigned genotype was obtained
	 */
	public int getTimeBest()
	{
		return _time_best;
	}
	
	/**
	 * The function returns the largest average configuration MSE of the best genotype.
	 * The largest MSE is chosen among elements of the output vector.
	 * 
	 * @return largest average configuration MSE
	 */
	public double getConfigMse()
	{
		double avg_error;
		
		if(_best_idx!=-1)
		{
			avg_error = _pool_of_bests.get(_best_idx)._error_config.getMaxAverageMse();
		}
		else
		{
			avg_error = Double.MAX_VALUE;
		}
		
		return avg_error;
	}
	
	/**
	 * The function returns a total average MSE of the best genotype.
	 * The total error is the config MSE together the error at time steps as the genotype stayed in the maturity pool
	 *    without evolution.
	 * 
	 * @return total average MSE
	 */
	public double getTotalMse()
	{
		double avg_error; 
		
		if(_best_idx!=-1)
		{
			avg_error = _pool_of_bests.get(_best_idx)._error.getMaxAverageMse();
		}
		else
		{
			avg_error = Double.MAX_VALUE;
		}
		
		return avg_error;
	}
	
	/**
	 * The function returns the largest configuration RMSE of the best genotype.
	 * The largest RMSE is chosen among elements of the output vector.
	 * 
	 * @return largest configuration RMSE
	 */
	public double getConfigRmse()
	{
		double avg_error;
		
		if(_best_idx!=-1)
		{
			avg_error = _pool_of_bests.get(_best_idx)._error_config.getMaxRmse();
		}
		else
		{
			avg_error = Double.MAX_VALUE;
		}
		
		return avg_error;
	}
	
	/**
	 * The function returns a total average RMSE of the best genotype.
	 * The total error is the config RMSE together the error at time steps as the genotype stayed in the maturity pool
	 *    without evolution.
	 * The largest RMSE is chosen from among all dimensions.
	 * 
	 * @return total largest RMSE
	 */
	public double getTotalRmse()
	{
		double avg_error; 
		
		if(_best_idx!=-1)
		{
			avg_error = _pool_of_bests.get(_best_idx)._error.getMaxRmse();
		}
		else
		{
			avg_error = Double.MAX_VALUE;
		}
		
		return avg_error;
	}
	
	/**
	 * The function returns the largest configuration NRMSE of the best genotype.
	 * The largest RMSE is chosen among elements of the output vector.
	 * 
	 * @param variance: array of variances for each element of the target vector
	 * @return largest configuration NRMSE
	 */
	public double getConfigNrmse(double[] variance)
	{
		double avg_error;
		
		if(_best_idx!=-1)
		{
			avg_error = _pool_of_bests.get(_best_idx)._error_config.getMaxNrmse(variance);
		}
		else
		{
			avg_error = Double.MAX_VALUE;
		}
		
		return avg_error;
	}
	
	/**
	 * The function returns a total largest NRMSE of the best genotype.
	 * The total error is the config NRMSE together the error at time steps as the genotype stayed in the maturity pool
	 *    without evolution.
	 * The largest NRMSE is chosen from among all dimensions.
	 * 
	 * @return total largest NRMSE
	 */
	public double getTotalNrmse(double[] variance)
	{
		double avg_error; 
		
		if(_best_idx!=-1)
		{
			avg_error = _pool_of_bests.get(_best_idx)._error.getMaxNrmse(variance);
		}
		else
		{
			avg_error = Double.MAX_VALUE;
		}
		
		return avg_error;
	}
	
	/**
	 * The function return the life-time of the best genotype in the maturity pool.
	 * Reason:
	 *    To know a portion of the configuration sequence which is perfectly modeled by the found best genotype.
     *    This is important because, for identification of dynamics (recognition of symbols), modeling of
     *       the configuration sequence has a higher priority over generalization.
     *    (introduced in "RC_Experiment063_20180715_ConfigLel")
	 * 
	 * @return life-time of the best genotype in the maturity pool
	 */
	public int getLifeTimeBest()
	{
		int life_time; 
		
		if(_best_idx!=-1)
		{
			life_time = _pool_of_bests.get(_best_idx)._error.all_single_squares.size();
		}
		else
		{
			life_time = 0;
		}
		
		return life_time;
	}
}
