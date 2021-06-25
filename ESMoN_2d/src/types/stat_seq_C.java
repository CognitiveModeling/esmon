package types;

/**
 * The class provides a list of statistics which shall be computed after running an ESN on the given sequence.
 * Each statistics is computed for all output neurons.
 * @author Danil Koryakin
 *
 */
public class stat_seq_C
{
	public int    comp_ident_ok_incl_error;//indicator that component identification succeeded and MSE is small
	public int    comp_ident_ok;//CIOK = 1 if active oscillator has same parameter lists as an applied sequence;
	                                  //   and MSE is irrelevant
	public double mse;//mean square error
	public double nrmse;//normalized root mean square error
	public double rmse;//root mean square error
	public int    sel;//small error length
	public int    lel;//large error length
	public int[]  sel_config;//small error length under the configuration of the sub-reservoirs
	
	/**
	 * Constructor which allocates memory for the statistics of each output neuron and of each sub-reservoir
	 * If the provided number of sub-reservoirs is 0 then no statistics of the sub-reservoirs should be saved.
	 * @param num_out: number of output neurons
	 * @param num_sub: number of sub-reservoirs
	 */
	public stat_seq_C(int num_sub)
	{
		comp_ident_ok_incl_error       = 0;
		comp_ident_ok = 0;
		mse   = Double.MAX_VALUE;
		nrmse = Double.MAX_VALUE;
		rmse  = Double.MAX_VALUE;
		sel   = Integer.MAX_VALUE;
		lel   = Integer.MAX_VALUE;
		if(num_sub!=0)
		{
			sel_config = new int[num_sub];
		}
		else
		{
			sel_config = null;
		}
	}
	
	/**
	 * The function assigns the data of the provided object to the corresponding elements of the host object.
	 * @param stat_data: provided object
	 */
	public void assignData(stat_seq_C stat_data)
	{
		int i;
		int num_sub;//number of sub-reservoirs in the provided object
		
		//prevent any assignment, if there are no data to be assigned
		if((sel_config!=null && stat_data.sel_config==null) ||
		   (sel_config==null && stat_data.sel_config==null)
		  )
		{
			num_sub = 0;
		}
		//allocate the array, if it is not available for the data to be assigned
		else if(sel_config==null && stat_data.sel_config!=null)
		{
			//allocate an array before the assignment
			num_sub = stat_data.sel_config.length;
			sel_config = new int[num_sub];
		}
		else
		{
			num_sub = stat_data.sel_config.length;
			if(num_sub!=sel_config.length)
			{
				System.err.println("assignData: different sizes of the arrays");
				System.exit(1);
			}
		}
		comp_ident_ok_incl_error       = stat_data.comp_ident_ok_incl_error;
		comp_ident_ok = stat_data.comp_ident_ok;
		mse   = stat_data.mse;
		nrmse = stat_data.nrmse;
		rmse  = stat_data.rmse;
		sel   = stat_data.sel;
		lel   = stat_data.lel;

		for(i=0; i<num_sub; i++)
		{
			sel_config[i] = stat_data.sel_config[i];
		}
	}
}
