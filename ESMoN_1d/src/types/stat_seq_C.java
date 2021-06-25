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
	public double[] mse;//mean square error
	public double[] nrmse;//normalized root mean square error
	public double[] rmse;//root mean square error
	public double[] sel;//small error length
	                    //(normally "int" is enough but the grand mean needs "double")
	public double[] lel;//large error length
	                    //(normally "int" is enough but the grand mean needs "double")
	public double[][] sel_config;//small error length under the configuration of the modules
	                             //(normally "int" is enough but the grand mean needs "double")
	
	/**
	 * Constructor which allocates memory for the statistics of each output neuron and of each sub-reservoir
	 * If the provided number of sub-reservoirs is 0 then no statistics of the sub-reservoirs should be saved.
	 * @param num_out: number of output neurons
	 * @param num_sub: number of sub-reservoirs
	 */
	public stat_seq_C(int num_out, int num_sub)
	{
		comp_ident_ok_incl_error       = 0;
		comp_ident_ok = 0;
		mse   = new double[num_out];
		nrmse = new double[num_out];
		rmse  = new double[num_out];
		sel   = new double[num_out];
		lel   = new double[num_out];
		if(num_sub!=0)
		{
			sel_config = new double[num_sub][num_out];
		}
		else
		{
			sel_config = null;
		}
	}
	
	/**
	 * The function sets all members of the class to "0".
	 */
	public void InitAllTo0()
	{
		int i, j;
		
		for(i=0; i<mse.length; i++)
		{
			mse  [i] = 0;
			nrmse[i] = 0;
			rmse [i] = 0;
			sel  [i] = 0;
			lel  [i] = 0;
			
			if(sel_config!=null)
			{
				for(j=0; j<sel_config[i].length; j++)
				{
					sel_config[i][j] = 0;
				}
			}
		}
	}
	
	/**
	 * The function assigns the data of the provided object to the corresponding elements of the host object.
	 * @param stat_data: provided object
	 */
	public void assignData(stat_seq_C stat_data)
	{
		int i,j;
		int num_out;//number of the elements in the array of each statistics
		int num_sub;//number of sub-reservoirs in the provided object
		
		num_out = stat_data.mse.length;
		if(num_out!=mse.length)
		{
			System.err.println("assignData: different sizes of the arrays");
			System.exit(1);
		}
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
			sel_config = new double[num_sub][];
			for(i=0; i<num_sub; i++)
			{
				sel_config[i] = new double[num_out];
			}
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
		
		for(i=0; i<num_out; i++)
		{
			mse[i]   = stat_data.mse[i];
			nrmse[i] = stat_data.nrmse[i];
			rmse[i]  = stat_data.rmse[i];
			sel[i]   = stat_data.sel[i];
			lel[i]   = stat_data.lel[i];
		}
		for(i=0; i<num_sub; i++)
		{
			for(j=0; j<num_out; j++)
			{
				sel_config[i][j] = stat_data.sel_config[i][j];
			}
		}
	}
}
