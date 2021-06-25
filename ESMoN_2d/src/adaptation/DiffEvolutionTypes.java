package adaptation;

public class DiffEvolutionTypes {
	/**
	 * This enumeration defines a list of possible parameter values for fetching individuals at random.
	 * @author Danil Koryakin
	 */
	public enum param_rand_fetch_E 
	{
		EVOL_PRF_ANY,      //any individual
		EVOL_PRF_VALID_OUT //individuals that provide the module outputs with a valid range
	};
	
	/**
	 * This enumeration defines classes of the trend of a prediction error.
	 * @author Danil Koryakin
	 */
	public enum error_trend_E
	{
		ET_DECREASE,                     //error decreased
		ET_INCREASE_LESS_THAN_1PERCENT,  //error increased by less than 1% 
		ET_INCREASE_LESS_THAN_10PERCENT, //error increased by less than 10% but more than 1% 
		ET_INCREASE_LESS_THAN_100PERCENT,//error increased by less than 100% but more than 10%
		ET_INCREASE_MORE_THAN_100PERCENT,//error increased by more than 100%
		ET_UNKNOWN                       //error trend has not been computed yet
	};
}
