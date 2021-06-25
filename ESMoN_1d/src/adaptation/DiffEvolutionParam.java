package adaptation;

import java.util.Random;

import types.vector_C;

import esn.mESN.config_ea_mode_E;

public class DiffEvolutionParam
{
	/**
	 * types of initialization at every time step under evolutionary adaptation
	 * @author Danil Koryakin
	 */
	public enum config_ea_init_E
	{
		CFG_EA_INIT_NONE,  //population is not initialized
		CFG_EA_INIT_RANDOM,//each element is randomly initialized with values from the whole valid range
		CFG_EA_INIT_HYPER_UNIFORM,//each element is initialized with uniform hyper-mutation at the best individual
		CFG_EA_INIT_HYPER_GAUSS,//initialization with normally distributed hyper-mutation at best individual
		CFG_EA_INIT_UNKNOWN;//unknown initialization mode
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CFG_EA_MODE_UNKNOWN", if the provided string was not found in any of the enumeration values
		 */
		public static config_ea_init_E fromString(String str)
		{
			config_ea_init_E init;//output variable
			
			init = CFG_EA_INIT_UNKNOWN;
			for(config_ea_init_E value : config_ea_init_E.values())
			{
				if(value.name().contains(str)==true)
				{
					init = value;
				}
			}
			if(init==CFG_EA_INIT_UNKNOWN)
			{
				System.err.println("config_ea_init_E.fromString: invalid string");
				System.exit(1);
			}
			
			return init;
		}
	};
	
	/**
	 * possible order of modules under co-evolution
	 * @author Danil Koryakin
	 */
	public enum config_ea_order_method_E
	{
		CFG_EA_ORDER_NORMAL,//1st evolved population is of module 1, 2nd evolved population is of module 2 and so on
		CFG_EA_ORDER_RANDOM,//at every time steps an order of modules is chosen at random
		CFG_EA_ORDER_UNKNOWN;//unknown order of modules under co-evolution
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CFG_EA_ORDER_UNKNOWN", if the provided string was not found in any of the enumeration values
		 */
		public static config_ea_order_method_E fromString(String str)
		{
			config_ea_order_method_E init;//output variable
			
			init = CFG_EA_ORDER_UNKNOWN;
			for(config_ea_order_method_E value : config_ea_order_method_E.values())
			{
				if(value.name().contains(str)==true)
				{
					init = value;
				}
			}
			if(init==CFG_EA_ORDER_UNKNOWN)
			{
				System.err.println("config_ea_order_E.fromString: invalid string");
				System.exit(1);
			}
			
			return init;
		}
	};
	
	public DiffEvolutionParam(int seed)
	{
		_rand = new Random(seed);
		coevol_method = config_ea_order_method_E.CFG_EA_ORDER_UNKNOWN;
	}
	
	/**
	 * The function computes indices of modules for their provided arrangement.
	 * The function needs the module's largest possible index which is given as an input parameter.
	 *
	 * @param max_index: module's largest possible index
	 * @return: array of modules' indices
	 */
	public int[] getPopOrder(int max_index)
	{
		int i,j;
		int num_idx;//number of indices
		int[] order;//output variable
		boolean is_present;//indicator that an index is already present among already generated module indices
		
		num_idx = max_index+1;
		order = new int[num_idx];
		switch(coevol_method)
		{
			case CFG_EA_ORDER_NORMAL:
				for(i=0;i<num_idx;i++)
				{
					order[i] = coevol_order.getElementAsInt(i);
				}
				break;
			case CFG_EA_ORDER_RANDOM:
				for(i=0;i<num_idx;i++)
				{
					do{
						order[i] = _rand.nextInt(num_idx);
						//check whether the current index had already been generated
						is_present = false;
						for(j=0; j<i && is_present==false; j++)
						{
							if(order[i]==order[j])
							{
								is_present = true;
							}
						}
					}while(is_present==true);
				}
				break;
			default:
				System.err.println("getPopOrder: unknown arrangement is required");
				System.exit(1);
				break;
		}
		
		return order;
	}
	
	/**
	 * The function computes a number of generations for a specified time step of a configuration sequence.
	 * The current number of generations is computed assuming that 1 generation must be done in the beginning
	 * of a configuration sequence and the maximum number of generations - in the end.
	 * 
	 * @param idx_cur: current time step of configuration sequence
	 * @param len_config: length of configuration sequence
	 * @return: computed number of generations
	 */
	public int computeNumGen(int idx_cur, int len_config)
	{
		int len_effective_config;//number of different time steps along a config sequence
		int len_interval;//number of time steps where a number of generations stay constant
		int idx_last_rest_step;//index of the last time step which is added to the 1st interval with 1 generation
		int num_gen_cur;//output variable
		
		len_effective_config = len_config - (fitness_length - 1);
		len_interval = len_effective_config / num_gen;
		
		/*if there are rest time steps, they are added to the 1st interval;
		  thus the 1st interval can be larger than the other intervals*/
		idx_last_rest_step = (len_config-1) - (len_interval * num_gen);
		if(idx_cur <= idx_last_rest_step)
		{
			num_gen_cur = 1;
		}
		else
		{
			num_gen_cur = (idx_cur - idx_last_rest_step - 1)/len_interval;
			num_gen_cur++;//1 generation in the 1st interval
		}
		
		return num_gen_cur;
	}
	
	public config_ea_mode_E  force_mode;
	public config_ea_init_E  init_mode;
	public config_ea_order_method_E coevol_method;//method to arrange modules under co-evolution
	public vector_C          coevol_order;//required order of modules (relevant only for NORMAL method)
	public vector_C          margin_module;//max allowed exaggeration of a valid range of module states
	public double F;
	public double CR;
	public double ciok_level;//largest allowed config MSE to accept component identification to be correct
	public double activ_thresh;//magnitude of module output to reach to start active suppression of that module
	public int sort_min_num;//minimum number of individuals to be sorted
	public int validation_len;//min number of steps for a genotype to stay in the maturity pool
	public int pop_size;
	public int num_gen;
	public int fitness_length;//number of consecutive time steps to compute fitness
	
	private Random _rand;//generator of random numbers
}
