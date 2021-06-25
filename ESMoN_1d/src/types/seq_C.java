package types;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Vector;

import MathStat.StatCommon;

public class seq_C
{
	enum leading_word_E
	{
		LW_PARAMETER_SETS,//leading word before a list of parameter sets of present oscillators
		LW_SAMPLES//leading word before a list of target samples
	};
	
	public String _filename;//name of the loaded file
	private seq_parameter_C _seq_param;//list of identifiers of target components that constitute the sequence
	private Vector<sample_C> _sample;//loaded sequence of samples
	
	private final int _idx_period_stationarity = 1;//index of input element with an index of period of stationarity
	
	/**
	 * constructor of a sequence class
	 */
	public seq_C()
	{
		_seq_param = new seq_parameter_C();
		_sample = new Vector<sample_C>(0, 1);
	}
	
	/**
	 * The copy constructor creates a sequence object which includes an interval of a provided sequences
	 * between the provided 1st and last time steps.
	 *  
	 * @param seq_init: provided sequence
	 * @param idx_first: 1st sample
	 * @param idx_last_sample: last sample
	 */
	public seq_C(seq_C seq_init, int idx_first, int idx_last)
	{
		int i;
		
		_seq_param = new seq_parameter_C();
		_sample = new Vector<sample_C>(0, 1);
		
		//copy the required interval from the entire loaded sequence
		setFilename(seq_init.getFilename());
		setSeqParam(seq_init.getSeqParam());
		for(i=idx_first; i<=idx_last; i++)
		{
			addSample( seq_init.getSample(i) );
		}
	}
	
	/**
	 * extract a value of sample from the given string and add it to the end of the sequence 
	 * @param str_sample: given string which specifies the input and output values of the sample
	 */
	private void addSampleFromStr(String str_sample)
	{
		sample_C sample;//sample to be added
		
		//create a new element for the sample
		sample = new sample_C(str_sample);
		_sample.add(sample);
	}
	
	/**
	 * The function stores a provided string as a set of parameters of a target component.
	 * 
	 * @param param_str: provided string with parameters of a target component (oscillator)
	 */
	private void addSeqParamFromStr(String param_str)
	{
		_seq_param.addOscillatorParam(param_str);
	}
	
	/**
	 * The function adds the given sample to the end of the current sequence.
	 * @param sample: given sample 
	 */
	public void addSample(sample_C sample)
	{
		_sample.add(sample);
	}
	
	/**
	 * The function returns a filename of the sequence.
	 * @return filename of the sequence
	 */
	public String getFilename()
	{
		return _filename;
	}
	
	/**
	 * The function returns a sample which required by its index.
	 * @param idx_sample: index of the required sample
	 * @return required sample
	 */
	public sample_C getSample(int idx_sample)
	{
		sample_C sample;//output object
		
		sample = new sample_C(_sample.get(idx_sample));
		
		return sample;
	}
	
	/**
	 * The function returns the input vector of a sample whose index is specified as the input parameter.
	 * @param idx, index of sample
	 * @return, input vector of the specified sample
	 */
	public double[] getSampleIn(int idx)
	{
		int      i;
		int      len_sample_in;
		double[] sample_in;//output array
		
		len_sample_in = _sample.get(idx)._in.length;
		sample_in = new double[len_sample_in];
		
		for(i=0; i<len_sample_in; i++)
		{
			sample_in[i] = _sample.get(idx)._in[i];
		}
		
		return sample_in;
	}
	
	/**
	 * The function returns the output vector of a sample whose index is specified as an input parameter.
	 * @param idx, index of sample
	 * @return, output vector of the specified sample
	 */
	public double[] getSampleOut(int idx)
	{
		int      i;
		int      len_sample_out;
		double[] sample_out;//output array
		
		len_sample_out = _sample.get(idx)._out.length;
		sample_out = new double[len_sample_out];
		
		for(i=0; i<len_sample_out; i++)
		{
			sample_out[i] = _sample.get(idx)._out[i];
		}
		
		return sample_out;
	}
	
	/**
	 * The function stores an output vector of a sample in a provided array.
	 * An index of the sample is specified as an input parameter.
	 * 
	 * @param idx: index of sample
	 */
	public void getSampleOut(double[] storage, int idx)
	{
		int      i;
		int      len_sample_out;
		
		len_sample_out = _sample.get(idx)._out.length;
		
		for(i=0; i<len_sample_out; i++)
		{
			storage[i] = _sample.get(idx)._out[i];
		}
	}
	
	/**
	 * The function indicates an index of period of stationarity for a specified sample.
	 * 
	 * Assumption: Indices of periods of stationarity are stored as the 2nd input element.
	 *  
	 * @param idx: index of the specified sample
	 * @return: index of period of stationarity
	 */
	public int getIndexOfStationarityPeriod(int idx)
	{
		return (int)_sample.get(idx)._in[_idx_period_stationarity];
	}
	
	/**
	 * return an sequence's samples as an array
	 * @return, array of sequence's samples
	 */
	public sample_C[] getSeq()
	{
		int i;
		sample_C[] sample_array;//output array
		
		sample_array = new sample_C[_sample.size()];
		for(i=0; i<sample_array.length; i++)
		{
			sample_array[i] = _sample.get(i);
		}
		
		return sample_array;
	}
	
	/**
	 * The function returns a set of parameters of all target components (oscillators).
	 * 
	 * @return: set of parameters of all target components (oscillators)
	 */
	public seq_parameter_C getSeqParam()
	{
		return _seq_param;
	}
	
	/**
	 * return number of samples in the sequence
	 * @return, sequence length
	 */
	public int getSeqLen()
	{
		return _sample.size();
	}
	
	/**
	 * The function searches for the largest absolute value of each output element over all samples.
	 * 
	 * @return: array of largest absolute value of each output element
	 */
	public double[] getMaxAbsOut()
	{
		int i;
		int num_out;//number of output elements
		double[] min_val;//array of smallest values for each output element
		double[] max_val;//array of largest values for each output element
		double[] max_abs;//output arrays

		num_out = getSampleOut(0).length;
		min_val = getMinOut();
		max_val = getMaxOut();
		max_abs = new double[num_out];
		for(i=0; i<num_out; i++)
		{
			if(Math.abs(min_val[i]) > Math.abs(max_val[i]))
			{
				max_abs[i] = Math.abs(min_val[i]);
			}
			else
			{
				max_abs[i] = Math.abs(max_val[i]);
			}
		}
		return max_abs;
	}
	
	/**
	 * The function searches for the largest value of each output's element over all samples of the sequence.
	 * @return: array of the largest values of each output's element
	 */
	public double[] getMaxOut()
	{
		int i,j;
		int len_sample;//number of elements in one sample
		int len_seq;//sequence length
		double[] cur_sample;//currently checked sample
		double[] max;//output array
		
		cur_sample = _sample.get(0)._out;
		len_sample = cur_sample.length;
		len_seq = _sample.size();
		
		//initialize the output array from the very 1st sample
		max = new double[len_sample];
		for(i=0; i<len_sample; i++)
		{
			max[i] = cur_sample[i];
		}
		
		//find the largest value for each element of the output vector starting with the 2nd sample
		for(i=1; i<len_seq; i++)
		{
			cur_sample = _sample.get(i)._out;
			for(j=0; j<len_sample; j++)
			{
				if(cur_sample[j] > max[j])
				{
					max[j] = cur_sample[j];
				}
			}
		}
		
		return max;
	}
	
	/**
	 * The function searches for the smallest value of each output's element over all samples of the sequence.
	 * @return: array of the smallest values of each output's element
	 */
	public double[] getMinOut()
	{
		int i,j;
		int len_sample;//number of elements in one sample
		int len_seq;//sequence length
		double[] cur_sample;//currently checked sample
		double[] min;//output array
		
		cur_sample = _sample.get(0)._out;
		len_sample = cur_sample.length;
		len_seq = _sample.size();
		
		//initialize the output array from the very 1st sample
		min = new double[len_sample];
		for(i=0; i<len_sample; i++)
		{
			min[i] = cur_sample[i];
		}
		
		//find the smallest value for each element of the output vector starting with the 2nd sample
		for(i=1; i<len_seq; i++)
		{
			cur_sample = _sample.get(i)._out;
			for(j=0; j<len_sample; j++)
			{
				if(cur_sample[j] < min[j])
				{
					min[j] = cur_sample[j];
				}
			}
		}
		
		return min;
	}
	
	/**
	 * The function arranges the indices of the sequence samples in the ascending order according to the distances
	 * between the output vectors of these samples and the provided vector.
	 * @param vect: provided vector
	 * @return ordered array of the sequence values
	 */
	public int[] getSortDistOut(double[] vect)
	{
		int   i, j;
		int   tmp_int;//temporary integer value for sorting
		int[] idx;//indices of the sequence samples (output variable)
		double tmp_double;//temporary double value for sorting
		double[] dist;//array of distances between the samples and the provided vector
		
		//initialize the array of indices and compute the distances
		idx  = new int[_sample.size()];
		dist = new double[idx.length];
		for(i=0; i<idx.length; i++)
		{
			idx[i]  = i;
			dist[i] = MathDiff.MathVector.computeDist(_sample.get(i)._out, vect);
		}
		
		//sort the distances
		for(i=0; i<idx.length-1; i++)
		{
			for(j=i+1; j<idx.length; j++)
			{
				if(dist[i] > dist[j])
				{
					tmp_double = dist[i];
					dist[i]    = dist[j];
					dist[j]    = tmp_double;
					tmp_int = idx[i];
					idx[i]  = idx[j];
					idx[j]  = tmp_int;
				}
			}
		}
		return idx;
	}
	
	/**
	 * The function computes variances for all elements of the input vector on the whole sequence.
	 * @return array of variances for all elements of the input vector
	 */
	public double[] computeVarianceIn(int idx_first, int idx_last)
	{
		int i, j;
		Number[] seq_in;//array with values of a single input
		double[] variance;//array of variances
		
		//allocate declared arrays
		seq_in   = new Number[_sample.size()];
		variance = new double[_sample.get(0).getLenIn()];
		
		//go over all elements of the input vector
		for(i=0; i<variance.length; i++)
		{
			//collect all values of one output element in one array
			for(j=0; j<seq_in.length; j++)
			{
				seq_in[j] = _sample.get(j)._in[i];
			}
			variance[i] = StatCommon.computeVariance(seq_in);
		}
		
		return variance;
	}
	
	/**
	 * The function computes variances for all elements of the output vector on the whole sequence.
	 * @return array of variances for all elements of the output vector
	 */
	public double[] computeVarianceOut()
	{
		int i, j;
		Number[] seq_out;//array with values of a single input
		double[] variance;//array of variances
		
		//allocate declared arrays
		seq_out  = new Number[_sample.size()];
		variance = new double[_sample.get(0).getLenOut()];
		
		//go over all elements of the input vector
		for(i=0; i<variance.length; i++)
		{
			//collect all values of one output element in one array
			for(j=0; j<seq_out.length; j++)
			{
				seq_out[j] = _sample.get(j)._out[i];
			}
			variance[i] = StatCommon.computeVariance(seq_out);
		}
		
		return variance;
	}
	
	/**
	 * The function loads a sequence of the current sequence object.
	 * 
	 * @param dir_path: path to the directory with the sequence file
	 * @return: none
	 */
	public void loadSeq(String dir_path)
	{
		File           file;//file object of sequence file
		FileReader     file_reader;
		BufferedReader reader;
		LinkedList<String> raw_data = new LinkedList<String>();
		LinkedList<String> params   = new LinkedList<String>();
		String         seq_path;//path to the sequence file
		String         new_line;
		String         cur_str;//current string for extraction
		boolean do_load_ids;//indicator to load identifiers of target components
		boolean do_load_samples;//indicator to load samples
		
		//assign path to sequence file
		seq_path = dir_path + File.separator + _filename;
		
		//initialize requests for loading data
		do_load_ids     = false;
		do_load_samples = false;
		
		try {
			file        = new File(seq_path);
			file_reader = new FileReader(file);
			reader      = new BufferedReader(file_reader);
			
			while (reader.ready())
			{
				//read new line
				new_line = reader.readLine();
				//filter out lines with comments
				if(!new_line.startsWith("#"))
				{
					//store new line
					if(do_load_ids==true)
					{
						params.add(new_line);
					}
					else if(do_load_samples==true)
					{
						raw_data.add(new_line);
					}
					else
					{
						System.err.println("ExpSeq.loadSeq: unexpected data field");
						System.exit(1);
					}
				}
				else
				{
					//reset all requests when a new comment row was loaded
					do_load_ids     = false;
					do_load_samples = false;
					//check for all possible leading words
					for(leading_word_E lw : leading_word_E.values())
					{
						//keyword is present in a loaded string
						if(new_line.contains( lw.toString() )==true)
						{
							switch(lw)
							{
								case LW_PARAMETER_SETS:
									do_load_ids = true;
									break;
								case LW_SAMPLES:
									do_load_samples = true;
									break;
								default:
									System.err.println("ExpSeq.loadSeq: unrealized keyword");
									System.exit(1);
							}
						}
					}
				}
			}//load separate rows from a file
			
			reader.close();
			reader = null;

			//extract samples from loaded strings
			if(raw_data.isEmpty()==true)
			{
				System.err.println("ExpSeq.loadSeq: no samples in the opened file");
				System.exit(1);
			}
			else
			{
				while(raw_data.isEmpty()==false)
				{
					//get string with input and output values of current sample 
					cur_str = raw_data.pollFirst();
					//extract values of current sample from string
					addSampleFromStr(cur_str);
				}
			}
			//extract identifiers from loaded strings
			if(params.isEmpty()==true)
			{
				System.err.println("ExpSeq.loadSeq: opened file does not contain IDs of target components");
				System.exit(1);
			}
			else
			{
				while(params.isEmpty()==false)
				{
					//get a single loaded string with a target identifier
					cur_str = params.pollFirst();
					//extract an identifier of a target component
					addSeqParamFromStr(cur_str);
				}
			}
			
		} catch (FileNotFoundException fnf) {
			System.err.println("loadSeq: invalid path to sequence file");
			System.exit(1);
		} catch (IOException io) {
			System.err.println("loadSeq: cannot read sequence file");
			System.exit(1);
		}
	}
	
	/**
	 * set a name of file where the sequence was loaded from
	 * @param filename
	 */
	public void setFilename(String filename)
	{
		_filename = filename;
	}
	
	/**
	 * The function copies sequence parameters from a provided parameter set.
	 * 
	 * @param seq_param: provided parameter set
	 */
	public void setSeqParam(seq_parameter_C seq_param)
	{
		_seq_param.addOscillatorParam(seq_param);
	}
}
