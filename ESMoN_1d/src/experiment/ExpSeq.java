package experiment;

import java.io.File;

import types.sample_C;
import types.seq_C;


/**
 * This class is responsible for loading the sequences and their partitioning into training, test and washout ones.
 * @author Danil Koryakin
 *
 */
public class ExpSeq
{	
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
			seq = new seq_C(_sequence[idx_seq], idx_first, idx_last);
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
