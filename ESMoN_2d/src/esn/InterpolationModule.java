package esn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import types.interval_C;
import Jama.Matrix;
import experiment.ExpOutput;
import experiment.ExpParam;

public class InterpolationModule
{
	/**
	 * Enumeration of identifiers of the parts of an interpolation module
	 * @author Danil Koryakin
	 */
	public enum interpolation_part_save_E
	{
		IPS_MODULE_TYPE,//module type
		IPS_ARGUMENT_MIN_MAX,//minimum and maximum values of the argument
		IPS_SLOPE,//slope of linear interpolation
		IPS_OFFSET//offset of linear interpolation
	};
	
	/**
	 * The class keeps formating information for loading parts of an interpolation module.
	 * @author Danil Koryakin
	 */
	private class interpolation_part_C
	{
		private String name;//name of the primary state variable
	}
	
	/**
	 * Interpolation part whose values are of the type "Double"
	 * 
	 * @author Danil
	 */
	private class part_double_C
	{
		Vector<Double> value;
		
		public part_double_C()
		{
			value = new Vector<Double>(0, 1);
		}
		
		public void AddValue(Double new_value)
		{
			value.add(new_value);
		}
		
		/**
		 * The function indicates a value at the specified element of the storage.
		 * 
		 * @param idx: specified index
		 * @return value at the specified element
		 */
		public double GetValue(int idx)
		{
			return value.get(idx).doubleValue();
		}
	}
	
	/**
	 * Interpolation part whose values are of the type "interval_C"
	 * 
	 * @author Danil
	 */
	private class part_interval_C
	{
		Vector<interval_C> value;
		
		public part_interval_C()
		{
			value = new Vector<interval_C>(0, 1);
		}
		
		public void AddValue(interval_C new_value)
		{
			value.add(new_value);
		}
		
		/**
		 * The function indicates a value at the specified element of the storage.
		 * 
		 * @param idx: specified index
		 * @return value at the specified element
		 */
		public interval_C GetValue(int idx)
		{
			return value.get(idx);
		}
		
		/**
		 * The function gets the number of intervals.
		 * 
		 * @return: number of intervals
		 */
		public int GetNumElements()
		{
			return value.size();
		}
	}
	
	/**
	 * methods for configuration of the sub-reservoirs
	 * @author Danil Koryakin
	 */
	public enum config_method_E
	{
		CM_DirectTeacherForce,//to call the function "configDirectTeacherForce()"
		CM_DiffEvolution,     //to call the function "configDiffEvolution()"
		CM_None,              //no configuration is required
		CM_Unknown;           //unknown configuration method is required
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CM_None", if the provided string was not found in any of the enumeration values
		 */
		public static config_method_E fromString(String str)
		{
			config_method_E method;//output variable
			
			method = CM_Unknown;
			for(config_method_E value : config_method_E.values())
			{
				if(value.name().contains(str)==true)
				{
					method = value;
				}
			}
			if(method==CM_Unknown)
			{
				System.err.println("ConfigMethod.fromString: invalid string to convert into configuration method");
				System.exit(1);
			}
			
			return method;
		}
	};
	
	/**
	 * This enumeration defines possible types of configuration of ESN modules.
	 * @author Danil Koryakin
	 */
	public enum config_type_E
	{
		CT_Decomposition,  //active ESN modules are not known a priori
		CT_Synchronization,//active ESN modules are known a priori
		CT_None;           //enumeration value which corresponds to none of available types of configuration 
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CT_None", if the provided string was not found in any of the enumeration values
		 */
		public static config_type_E fromString(String str)
		{
			config_type_E type;//output variable
			
			type = CT_None;
			for(config_type_E value : config_type_E.values())
			{
				if(value.name().contains(str)==true)
				{
					type = value;
				}
			}
			if(type==CT_None)
			{
				System.err.println("ConfigType.fromString: invalid string to convert into configuration type");
				System.exit(1);
			}
			
			return type;
		}
	};
	
	/**
	 * modes of teacher-forcing under evolutionary adaptation
	 * @author Danil Koryakin
	 */
	public enum config_ea_mode_E
	{
		CFG_EA_MODE_PERMANENT, //evolutionary algorithm is started at every time step
		CFG_EA_MODE_INTERLEAVE,//evolutionary algorithm is started every N-th time step; free-run between these steps
		CFG_EA_MODE_UNKNOWN;//not known how to apply the evolutionary algorithm
		
		/**
		 * This method is used to retrieve a value of this enumeration from a provided string.
		 * @param str: string which should be a sub-sequence of characters in one of the enumeration's values
		 * @return: value of the enumeration where the provided string is found;
		 *          "CFG_EA_MODE_UNKNOWN", if the provided string was not found in any of the enumeration values
		 */
		public static config_ea_mode_E fromString(String str)
		{
			config_ea_mode_E mode;//output variable
			
			mode = CFG_EA_MODE_UNKNOWN;
			for(config_ea_mode_E value : config_ea_mode_E.values())
			{
				if(value.name().contains(str)==true)
				{
					mode = value;
				}
			}
			if(mode==CFG_EA_MODE_UNKNOWN)
			{
				System.err.println("config_ea_mode_E.fromString: invalid string");
				System.exit(1);
			}
			
			return mode;
		}
	};

	private interpolation_part_C[]  _interpolation_part;//array with formatting parameters of interpolation parts to be saved
	private part_interval_C[] _arg_interval;//storages with intervals of linear interpolation for all output dimensions
	private part_double_C[]   _k;//storages with slope of linear interpolation for all output dimensions
	private part_double_C[]   _offset;//storages with offset of linear interpolation for all output dimensions
	private Matrix            _output;//outputs of the interpolation module
	private double            _input;//argument vale for which the output was computed

	/**
	 * Constructor of the class
	 * 
	 * @param module_idx: index of loaded module (needed under loading the module)
	 * @param esn_load_file: file with parameters of ESN module
	 * @param exp_param: required parameters of ESN module
	 * @param exp_output: object of a class responsible for the file output
	 * @param commonPathCurrRun: common incomplete path to save algorithm specific data
	 * @param size_out: expected dimensionality of interpolation
	 */
	public InterpolationModule(int module_idx, String esn_load_files, ExpParam exp_param, ExpOutput exp_output,
			                   String commonPathCurrRun, int size_out)
	{
		int i;
		int idx_related;
		
		//define names for all interpolation parts in the input file
		_interpolation_part = new interpolation_part_C[interpolation_part_save_E.values().length];
		for(i=0; i<interpolation_part_save_E.values().length; i++)
		{
			_interpolation_part[i] = new interpolation_part_C();
		}
		idx_related = interpolation_part_save_E.IPS_MODULE_TYPE.ordinal();
		_interpolation_part[idx_related].name = "TYPE";
		idx_related = interpolation_part_save_E.IPS_ARGUMENT_MIN_MAX.ordinal();
		_interpolation_part[idx_related].name = "MINIMUM_AND_MAXIMUM_OF_ARGUMENT";
		idx_related = interpolation_part_save_E.IPS_SLOPE.ordinal();
		_interpolation_part[idx_related].name = "SLOPE";
		idx_related = interpolation_part_save_E.IPS_OFFSET.ordinal();
		_interpolation_part[idx_related].name = "OFFSET";
		
		_arg_interval = new part_interval_C[size_out];
		_k = new part_double_C[size_out];
		_offset = new part_double_C[size_out];
		for(i=0; i<size_out; i++)
		{
			_arg_interval[i] = new part_interval_C();
			_k[i] = new part_double_C();
			_offset[i] = new part_double_C();
		}
		
		_output = new Matrix(1, size_out);
		_input = Double.NaN;
		        
		//check whether a module must be loaded
		if(esn_load_files.contains("*")==false)
		{
			//interpolation modules have always index "0"
			module_idx = 0;
			loadInterpolationModule(module_idx, esn_load_files, size_out);
		}
		else
		{
			System.err.println("FfannModule: missing names of the FF-ANN modules");
			System.exit(1);
		}
	}
	
	/**
	 * The function determines a number of columns in a matrix whose rows are given by the provided array of strings.
	 * In the provided array, values of the same row shall be separated with spaces.
	 * 
	 * @param str_vals: array of strings with values of a matrix
	 * @return: number of columns
	 */
	private int getNumCols(Vector<String> str_vals)
	{
		int i, j;
		int num_cols;//number of columns in the 1st row (returned value)
		int num_cols_i;//number of columns in the 2nd and further rows
		String[] str_dummy;//dummy array with values of the same row
		
		i = 0;
		str_dummy = str_vals.get(i).split(" ");
		//number of columns is a number of not-empty strings in the same row
		num_cols = 0;
		for(j=0; j<str_dummy.length; j++)
		{
			if(str_dummy[j].isEmpty()==false)
			{
				num_cols++;
			}
		}
		//check the number of columns in all rows;
		//it must be equal
		for(i=1; i<str_vals.size(); i++)
		{
			str_dummy = str_vals.get(i).split(" ");
			num_cols_i = 0;
			for(j=0; j<str_dummy.length; j++)
			{
				if(str_dummy[j].isEmpty()==false)
				{
					num_cols_i++;
				}
			}
			
			if(num_cols!=num_cols_i)
			{
				System.err.println("getNumCols: provided strings conain different numbers of values");
				System.exit(1);
			}
		}
		
		return num_cols;
	}
	
	/**
	 * The function assigns the array of the argument intervals from the provided array of strings.
	 * 
	 * @param str_array: provided array of strings
	 */
	private void assignArgIntervalFromStrArray(String[][] str_array)
	{
		int i, j;
		boolean is_dummy;
		interval_C tmp_interval;
		
		if(_arg_interval.length!=str_array[0].length)
		{
			System.err.println("Interpolation.assignArgIntervalFromStrArray: mismatch in arrays' dimensions");
			System.exit(1);
		}
		
		for(i=0; i<_arg_interval.length; i++)
		{
			is_dummy = false;
			//add valid values until the next dummy symbol
			for(j=0; j<str_array.length && is_dummy==false; j++)
			{
				//check for the dummy symbol
				if(str_array[j][i].contains("*")==true)
				{
					is_dummy = true;
				}
				else
				{
					tmp_interval = new interval_C(str_array[j][i]);
					_arg_interval[i].AddValue(tmp_interval);
				}
			}
		}
	}
	
	/**
	 * The function stores strings from the provided array as double values in the provided storage.
	 *
	 * @param part: provided storage
	 * @param str_array: provided array of strings
	 */
	private void assign2DdoubleFromStrArray(part_double_C[] part, String[][] str_array)
	{
		int i, j;
		boolean is_dummy;
		Double tmp_double;
		
		if(part.length!=str_array[0].length)
		{
			System.err.println("Interpolation.assign2DdoubleFromStrArray: mismatch in arrays' dimensions");
			System.exit(1);
		}
		
		for(i=0; i<part.length; i++)
		{
			is_dummy = false;
			//add valid values until the next dummy symbol
			for(j=0; j<str_array.length && is_dummy==false; j++)
			{
				//check for the dummy symbol
				if(str_array[j][i].contains("*")==true)
				{
					is_dummy = true;
				}
				else
				{
					tmp_double = Double.valueOf(str_array[j][i]);
					part[i].AddValue(tmp_double);
				}
			}
		}
	}
	
	/**
	 * The function extracts values of the provided type from an array of strings. Each string can contain one or
	 * several values. All value of the same string are stored in one row of the output array.
	 * 
	 * @param str_vals: array of strings with values to be extracted
	 * @return: array of extracted values
	 */
	private String[][] get2DArrayFromStr(Vector<String> str_vals)
	{
		int i,j;
		int idx_col;//index of a column of the currently assigned element
		int num_cols;//number of columns
		String[][] vals;
		String[] str_row;//string with values of the same row
		
		vals = new String[str_vals.size()][];
		num_cols = getNumCols(str_vals);

		for(i=0; i<vals.length; i++)
		{
			str_row = str_vals.get(i).split(" ");

			//extract values of the same row
			vals[i] = new String[num_cols];
			idx_col = 0;
			for(j=0; j<str_row.length; j++)
			{
				if(str_row[j].isEmpty()==false)
				{
					vals[i][idx_col] = str_row[j];
					idx_col++;
				}
			}
		}
		
		return vals;
	}
	
	/**
	 * The function extracts a specified part of the specified interpolation module from the provided set of strings.
	 * The interpolation part is specified by its identifier.
	 * The interpolation module is specified by its index.
	 * The extracted values are returned as an array of strings where values of the same row are separated
	 * with spaces.
	 * A row with the column names is not included in the output data.
	 * A column with the row names is not included in the output data either.
	 * 
	 * @param data: loaded data of an interpolation module without comments
	 * @param esn_part: identifier of the interpolation module
	 * @param module_idx: specified module index
	 * @return: set of strings with values of the specified part
	 */
	private Vector<String> getInterpolationPart(Vector<String> data, interpolation_part_save_E esn_part, int module_idx)
	{
		int i;
		int idx_name;//index of a string with a name of an ESN part
		int idx_space;//index of a space in a string
		String part_name;//name of an ESN part to be extracted
		String str_tmp;//temporary string
		Vector<String> str_vals;//output variables
		
		str_vals = new Vector<String>(0, 1);
		
		//find the beginning of the specified ESN part
		part_name = _interpolation_part[esn_part.ordinal()].name + "_MODULE_" + module_idx;
		idx_name = data.indexOf(part_name);
		
		if(idx_name!=-1)
		{
			i = idx_name + 2;//"+2" in order to get rid of a name of the ESN part 
			str_tmp = data.get(i);
			
			while(str_tmp.isEmpty()==false)//empty string must be at the end of each ESN part
			{
				//remove a name of the row
				idx_space = str_tmp.indexOf(" ");
				str_tmp = str_tmp.substring(idx_space+1);
				
				//remove possible spaces at the beginning and at the end
				str_tmp = str_tmp.trim();
				
				//store the string for the output
				str_vals.add(str_tmp);

				i++;
				//previously analyzed string could be the last one in the provided array
				if(i < data.size())
				{
					str_tmp = data.get(i);
				}
				else
				{
					str_tmp = "";
				}
			}
		}
		
		return str_vals;
	}
	
	/**
	 * This function loads an interpolation module which was specified by its index.
	 * Parameters are loaded from the file which is specified by the provided path.
	 * It is necessary that the interpolation module file contains a module with the provided index.
	 * Otherwise, it issues an error message.
	 * 
	 * @param module_idx: index of a module to be loaded
	 * @param interpoation_path: path to the interpolation file to be loaded
	 * @param size_out: expected dimensionality of the loaded interpolation
	 */
	private void loadInterpolationModule(int module_idx, String interpolation_path, int size_out)
	{
		File esn_file;//file object
		String[][] str_vals2D;//two-dimensional array of extracted values as the type "String"
		Vector<String> data;//vector of strings loaded from the file
		Vector<String> interpolation_part;//vector of strings with loaded values of a certain part
		
		data = new Vector<String>(0,1);
		esn_file = new File(interpolation_path);
		try {
			BufferedReader reader = new BufferedReader(new FileReader(esn_file));
			
			while (reader.ready())
			{
				String line = reader.readLine();
				
				//filter out lines with comments
				if(!line.startsWith("#"))
				{
					data.add(line);
				}
			}
			
			reader.close();
			reader = null;
		} catch (FileNotFoundException fnf) {
			data = null;
		} catch (IOException ioe) {
			data = null;
		}
		
		//parse the loaded data
		if(data!=null)
		{			
			//determine number of neurons in specified module
			interpolation_part = getInterpolationPart(data, interpolation_part_save_E.IPS_MODULE_TYPE, module_idx);			
			//check whether the specified module is available in the file
			if(interpolation_part.get(0).toString().matches("INTERPOLATION")==false)
			{
				System.err.println("Interpolation.loadInterpolationModule: specified module is not available in the file");
				System.exit(1);
			}
			
			//*** INTERVALS OF ARGUMENT ***

			interpolation_part = getInterpolationPart(data, interpolation_part_save_E.IPS_ARGUMENT_MIN_MAX, module_idx);
			str_vals2D = get2DArrayFromStr(interpolation_part);
			assignArgIntervalFromStrArray(str_vals2D);
			
			//*** SLOPES OF LINES ON INTERVALS
			
			interpolation_part = getInterpolationPart(data, interpolation_part_save_E.IPS_SLOPE, module_idx);
			str_vals2D = get2DArrayFromStr(interpolation_part);
			assign2DdoubleFromStrArray(_k, str_vals2D);
			
            //*** OFFSETS OF LINES ON INTERVALS
			
			interpolation_part = getInterpolationPart(data, interpolation_part_save_E.IPS_OFFSET, module_idx);
			str_vals2D = get2DArrayFromStr(interpolation_part);
			assign2DdoubleFromStrArray(_offset, str_vals2D);
		}
		else
		{
			System.err.println("InterpolationModule.loadInterpolationModule: specified file does not exist");
			System.exit(1);
		}
	}
	
	/**
	 * The function searches for the interval which is matching the provided argument value in the specified dimension.
	 * 
	 * @param arg: provided argument
	 * @param dim: specified dimension
	 * @return index of matching interval
	 */
	private int FindInterval(double arg, int dim)
	{
		int idx_interval;//index of the found interval
		int idx_interval_le;//index of the left checked interval
		int idx_interval_ri;//index of the right checked interval
		int idx_interval_mid;//index of the interval in the middle between the checked intervals
		int num_intervals;
		double upper_border_mid;//upper border of the middle interval
		
		num_intervals = _arg_interval[dim].GetNumElements();
		
		//check whether the argument is outside the range which is covered by all available intervals
		if(_arg_interval[dim].GetValue(0).IsBelow(arg)==true)
		{
			idx_interval = 0;
		}
		else if(_arg_interval[dim].GetValue(num_intervals-1).IsAbove(arg)==true)
		{
			idx_interval = num_intervals - 1;
		}
		else
		{
			idx_interval_le = 0;
			idx_interval_ri = num_intervals-1;
			while((idx_interval_ri - idx_interval_le) > 0)
			{
				idx_interval_mid = (idx_interval_ri + idx_interval_le)/2;
				
				//find out whether the argument is to the left or to the right from the middle interval
				upper_border_mid = _arg_interval[dim].GetValue(idx_interval_mid).getUpperLimitAsDouble();
				if(arg < upper_border_mid)
				{
					idx_interval_ri = idx_interval_mid;
				}
				else
				{
					idx_interval_le = idx_interval_mid+1;
				}
			}
			
			//it does not matter which interval index to assign, left or right; they are equal
			idx_interval = idx_interval_ri;
		}
		
		return idx_interval;
	}
	
	/**
	 * This function computes interpolation for each dimension at the provided argument value.
	 * The function starts a search for new output values if it is requested.
	 * Normally, it is not necessary to search for new output values if the input of the interpolation module
	 *    did not change.
	 * 
	 * @param do_search: request to find new output values
	 */
	public void calculateOutputVector(boolean do_search)
	{
		int i;
		int idx_interval;
		double output;//computed interpolation for the current dimension
		
		if(do_search==true)
		{
			for(i=0; i<_output.getColumnDimension(); i++)
			{
				idx_interval = FindInterval(_input, i);
				output = _k[i].GetValue(idx_interval)*_input + _offset[i].GetValue(idx_interval);

				_output.set(0, i, output);
			}
		}
	}
	
	/**
	 * The function in the interface to compute interpolation for the provided input sample.
	 * 
	 * @param sample_in: provided input sample
	 */
	public void advance(double[] sample_in)
	{
		boolean do_search;//request to search for new output value if the input changed
		
		//check the input
		if(sample_in.length==1)
		{
			if(_input != sample_in[0])
			{
				_input = sample_in[0];
				do_search = true;
			}
			else
			{
				do_search = false;
			}
		}
		else
		{
			System.err.println("InterpolationModule.advance: invalid size of the input vector");
			System.exit(1);
			do_search = false;
		}
		
		//compute interpolation for every dimension
		calculateOutputVector(do_search);
	}
	
	/**
	 * The function returns the output vector for the provided argument.
	 * 
	 * @param arg: provided argument
	 * @return: Matrix of the output values
	 */
	public Matrix GetOutputVector(double arg)
	{
		//check the input
		if(_input!=arg)
		{
			System.err.println("InterpolationMOdule.GetOutputVector: no computed output for this argument");
			System.exit(1);
		}
		
		return _output;
	}
}
