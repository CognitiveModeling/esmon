package experiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Vector;

import types.seq_parameter_C;

import MathStat.StatCommon;


/**
 * This class is responsible for loading the sequences and their partitioning into training, test and washout ones.
 * @author Danil Koryakin
 *
 */
public class ExpSeq
{
	enum leading_word_E
	{
		LW_PARAMETER_SETS,//leading word before a list of parameter sets of present oscillators
		LW_SAMPLES//leading word before a list of target samples
	};
	
	public class seq_C
	{
		private String _filename;//name of the loaded file
		private seq_parameter_C _seq_param;//list of identifiers of target components that constitute the sequence
		private Vector<sample_C> _sample;//loaded sequence of samples
		
		/**
		 * constructor of a sequence class
		 */
		public seq_C()
		{
			_seq_param = new seq_parameter_C();
			_sample = new Vector<sample_C>(0, 1);
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
		 * The function returns an array of strings. Each of them contains parameters parameters of the corresponding
		 * dynamics in the sequence.
		 *  
		 * @return: dynamics' parameters as an array of strings
		 */
		public Vector<String> getSeqParametersAsStr()
		{
			return _seq_param.getSeqParametersAsStr();
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
		
		/**
		 * The function subtracts a provided bias from each output vector of the sequence.
		 * 
		 * @param bias: provided bias
		 */
		public void subtractBias(double[] bias)
		{
			int i, j;
			int seq_len;
			double[] output;//output vector of the current sample
			
			seq_len = _sample.size();
			for(i=0; i<seq_len; i++)
			{
				output = _sample.get(i)._out;
				for(j=0; j<output.length; j++)
				{
					output[j] -= bias[j];
				}
			}
		}
		
		/**
		 * The function adds a provided bias to each output vector of the sequence.
		 * 
		 * @param bias: provided bias
		 */
		public void addBias(double[] bias)
		{
			int i, j;
			int seq_len;
			double[] output;//output vector of the current sample
			
			seq_len = _sample.size();
			for(i=0; i<seq_len; i++)
			{
				output = _sample.get(i)._out;
				for(j=0; j<output.length; j++)
				{
					output[j] += bias[j];
				}
			}
		}
		
		/**
		 * The function returns a constant bias for each element of the output vector of the first component
		 * in the sequence.
		 * The function issues an error if there are multiple components in teh sequence.
		 */
		public double[] getBias()
		{
			int idx_osci;
			double[] bias;//array to fill up with bias values

			//the function should be called only for sequences that contain only a single oscillator
			if(_seq_param.getNumOscilators()!=1)
			{
				System.err.println("ExpSeq.getBias: sequence must contain a single oscillator");
				System.exit(1);
			}
			
			//"0" because a size of an output vector is equal for all samples in the sequence
			bias = new double[_sample.get(0)._out.length];
			
			idx_osci = 0;//output bias is returned for the first component
			_seq_param.getSeqOutputBiasAsDouble(idx_osci, bias);
			
			return bias;
		}
	};
	
	public class sample_C
	{
		public double[] _in;//vector of input values
		public double[] _out;//vector of output values
		
		/**
		 * create an object of a sample and allocate memory for the input and output vectors without their assignment
		 * @param len_in, length of the input vector
		 * @param len_out, length of the output vector
		 */
		public sample_C(int len_in, int len_out)
		{
			//allocate memory for the input vector, only if its length is larger than 0
			if(len_in > 0)
			{
				_in  = new double[len_in];
			}
			//allocate memory for the output vector, only if its length is larger than 0
			if(len_out > 0)
			{
				_out = new double[len_out];
			}
		}
		
		/**
		 * create an object of a sample and extract input and output vectors from given string
		 * @param str, given string
		 */
		public sample_C(String str)
		{
			final int size_input = 1;
			final int idx_input_element = 4;//index of extracted element keeping Phi
			
			int i;
			String[] vectors;//strings of input and output vectors at current time step
			String[] strInputVector;//strings of elements of input vector at current time step 
			String[] strOutputVector;//strings of elements of output vector at current time step
			
			vectors         = str.split("\t");
			strInputVector  = vectors[1].split(",");
			strOutputVector = vectors[2].split(",");
			
			_in  = new double[size_input];
			//_in  = new double[strInputVector.length];
			_out = new double[strOutputVector.length];
			
			try
			{
				/*for(i=0; i < _in.length; i++)
				{
					_in[i] = Double.valueOf(strInputVector[i]);
				}*/
				_in[0] = Double.valueOf(strInputVector[idx_input_element]);
				for(i=0; i < _out.length; i++)
				{
					_out[i] = Double.valueOf(strOutputVector[i]);
				}
			}
			catch (NumberFormatException nfe)
			{
				System.err.println("sample_C: invalid sample value");
				System.exit(1);
			}
		}
		
		/**
		 * The constructor creates a new sample as a copy of the provided sample. 
		 * @param sample: provided sample
		 */
		public sample_C(sample_C sample)
		{
			int i;
			
			_in  = new double[sample.getLenIn()];
			for(i=0; i<_in.length; i++)
			{
				_in[i] = sample._in[i];
			}
			
			_out = new double[sample.getLenOut()];
			for(i=0; i<_out.length; i++)
			{
				_out[i] = sample._out[i];
			}
		}
		
		/**
		 * return a length of the input vector
		 * @return, length of the input vector
		 */
		public int getLenIn()
		{
			return _in.length;
		}
		
		/**
		 * return a length of the output vector
		 * @return, length of the output vector
		 */
		public int getLenOut()
		{
			return _out.length;
		}
	};
	
	private boolean _is_multifile;//indicator that loaded files belong to the same multi-file sequence 
	private String  _dir_path;//path to directory with sequences
	private seq_C[] _sequence;//array of sample sequences

	/**
	 * constructor of the class "ExpSeq"
	 * @param dir_path: path to directory with sequences
	 * @param seq_name: array of names of the sequence files to be loaded
	 */
	public ExpSeq(String dir_path, String[] seq_name)
	{
		int i;
		int seq_len;//length of loaded sequence
		File file;
		
		//get a number of loaded sequences
		//Since it was already checked in ExpParam, that there are no multi-file and single-file sequences
		//simultaneously, it is enough to check the 1st sequence whether it is a directory or a single file.
		file = new File(dir_path + File.separator + seq_name[0]);
		if(file.isDirectory()==true)
		{
			_is_multifile = true;
			dir_path = dir_path + File.separator + seq_name[0];
			seq_name = file.list();
		}
		else
		{
			_is_multifile = false;
		}
		
		_sequence = new seq_C[seq_name.length];
		_dir_path = dir_path;
		
		seq_len = 0;
		for(i=0; i<seq_name.length; i++)
		{
			_sequence[i] = new seq_C();
			_sequence[i].setFilename(seq_name[i]);
			_sequence[i].loadSeq(dir_path);
			//check that all loaded sequences are of the same length
			if(i==0)
			{
				seq_len = _sequence[i].getSeqLen();//assign length of the 1st sequence
			}
			else
			{
				if(seq_len!=_sequence[i].getSeqLen())//all other sequences should have the same length as the first one
				{
					System.err.println("ExpSeq: loaded sequences have different lengths");
					System.exit(1);
				}
			}
		}
	}
	
	/**
	 * This is a constructor of the class ExpSeq.
	 * It constructs a new object as a part of a provided sequence object.
	 * The resulting object contains only a specified sequence from the provided object. 
	 * The sequence is specified by its index
	 * 
	 * @param orig_exp_seq: original sequence object
	 * @param idx_seq: index of a specified sequence
	 */
	public ExpSeq(ExpSeq orig_exp_seq, int idx_seq)
	{
		int      i;
		int      seq_len;//length of a specified sequence
		sample_C sample;//current sample
		
		_is_multifile = true;
		_dir_path = orig_exp_seq._dir_path;
		
		//resulting object contains only one sequence
		_sequence = new seq_C[1];
		_sequence[0] = new seq_C();
		
		//copy contents of the specified sequence
		_sequence[0]._filename = orig_exp_seq._sequence[idx_seq]._filename;
		_sequence[0].setSeqParam(orig_exp_seq._sequence[idx_seq].getSeqParam());
		seq_len = orig_exp_seq._sequence[idx_seq].getSeqLen();
		for(i=0; i<seq_len; i++)
		{
			sample = new sample_C( orig_exp_seq._sequence[idx_seq].getSample(i) );
			_sequence[0].addSample(sample);
		}
	}
	
	/**
	 * The function converts the output values of the provided sequences to their differences which can be used for
	 * training the output weights of separate sub-reservoirs. In the sequence with index 0 the output values stay as
	 * they are. The input value of the sequences do not change.
	 * @param seq: array of provided sequences
	 * @return: array of differences between the provided sequences
	 */
	public seq_C[] convertSeqToDiff(seq_C[] seq)
	{
		int i, j, k;
		int seq_len;//length of sequence
		sample_C sample;//currently created sample
		seq_C[] seq_new;//output array 
		
		seq_new = new seq_C[seq.length];
		seq_len = seq[0].getSeqLen();
		
		//go over the whole sequence
		for(i=0; i<seq_len; i++)
		{
			//allocate each sequence in the beginning of the process
			if(i==0)
			{
				for(j=0; j<seq.length; j++)
				{
					seq_new[j] = new seq_C();
				}
				//sequence parameters are assigned only for the 1st sequence 
				seq_new[0].setSeqParam(seq[0].getSeqParam());
			}
			
			//assign the first sequence as it is
			sample = new sample_C( seq[0].getSample(i) );
			seq_new[0].addSample( sample );
			
            //assign further sequences
			for(j=1; j<seq.length; j++)
			{
				sample = new sample_C(seq[j].getSample(i).getLenIn(),
									  seq[j].getSample(i).getLenOut());
				//assign the input values
				for(k=0; k<sample.getLenIn(); k++)
				{
					sample._in[k] = seq[j].getSample(i)._in[k];
				}
				//assign the output values
				for(k=0; k<sample.getLenOut(); k++)
				{
					sample._out[k] = seq[j  ].getSample(i)._out[k];
					sample._out[k]-= seq[j-1].getSample(i)._out[k];
				}
				seq_new[j].addSample(sample);
			}
		}
		
		return seq_new;
	}
	
	/**
	 * This function returns an interval of the sequence which is required by its index. The start and end samples of
	 * the interval are given with their indices.
	 * @param idx_seq: given sequence index
	 * @param idx_first: index of the 1st sample of the interval
	 * @param idx_last: index of the last sample of the interval
	 * @return array of sequence's samples
	 */
	public seq_C getSeq(int idx_seq, int idx_first, int idx_last)
	{
		int   i;
		seq_C seq;
		
		if(idx_seq >= _sequence.length)
		{
			System.err.println("getSeq: sequence index exceeds the total number of sequences");
			System.exit(1);
			seq = null;
		}
		else
		{
			//check the indices of samples
			if(idx_first > idx_last)
			{
				System.err.println("getSeq: index of 1st sample can be only smaller or equal to index of last sample");
				System.exit(1);
			}
			if(idx_last >= _sequence[idx_seq].getSeqLen())
			{
				System.err.println("getSeq: index of last sample exceeds the sequence length");
				System.exit(1);
			}
			
			//allocate a sequence for the output
			seq = new seq_C();
			
			//copy the required interval from the entire loaded sequence
			seq.setFilename(_sequence[idx_seq].getFilename());
			seq.setSeqParam(_sequence[idx_seq].getSeqParam());
			for(i=idx_first; i<=idx_last; i++)
			{
				seq.addSample( _sequence[idx_seq].getSample(i) );
			}
		}
		
		return seq;
	}
	
	/**
	 * The function returns the number of samples in each sequence. The function returns only one value because all
	 * sequences should consist of the same number of samples.
	 * @return number of samples in each sequence
	 */
	public int getSeqLen()
	{
		return _sequence[0].getSeqLen();
	}
	
	/**
	 * return a filename of the sequence which was loaded for the given index
	 * @param idx_seq: given sequence index
	 * @return: filename of loaded sequence
	 */
	public String getSeqName(int idx_seq)
	{
		String filename;

		if(idx_seq >= _sequence.length)
		{
			System.err.println("getSeqName: sequence index exceeds the total number of sequences");
			System.exit(1);
			filename = null;
		}
		else
		{
			filename = _sequence[idx_seq].getFilename();
		}
		return filename;
	}
	
	/**
	 * The function shows all sequence names with a single string.
	 * 
	 * @return: string with all sequence names
	 */
	public String getAllSeqNamesAsStr()
	{
		int    i;
		int    num_seq;//number of sequences
		String tmp_str;//temporary value
		String str_all_names;
		
		num_seq = _sequence.length;
		str_all_names = "{";
		for(i=0; i<num_seq; i++)
		{
			tmp_str = _sequence[i].getFilename();
			str_all_names += tmp_str;
			if(i!=num_seq-1)//no comma is needed after the last element
			{
				str_all_names += ",";
			}
		}
		str_all_names += "}";
		
		return str_all_names;
	}
	
	/**
	 * The function returns a total number of loaded sequences.
	 * @return total number of loaded sequences
	 */
	public int getSeqNum()
	{
		return _sequence.length;
	}
	
	/**
	 * The function returns a path the directory where the sequences were loaded from.
	 * @return path to the directory with sequences
	 */
	public String getSeqPath()
	{
		return _dir_path;
	}
	
	/**
	 * The function indicates that a host object contains data from different files of the same multi-file sequence.
	 * 
	 * @return: TRUE if data belong to a multi-file sequence
	 *          FALSE if data belong to several single-file sequences
	 */
	public boolean isMiltifileSeq()
	{
		return _is_multifile;
	}
}
