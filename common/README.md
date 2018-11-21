# Common

These are all the functions and utilities that are common to all the models. 

## Files

This folder has following files:

1. `split_data.py`: The file to split and dump the data in smaller files of input number of lines. Following is the list of arguments:

```
  --file-path:     path to training file
  --output-name:   prefix for output file name, training file is output_name_train.txt
  --num-sentences: number of sentences in output training file
  --ratio:         split ratio (val/total)
  --randomize:     to randomise data. if True random points are selected
```

2. `tex2ctf_mod.py`: The file to dump the data in Microsoft CNTK format. This is the modified version of file in `/baselines`. Following is the list of arguments:

```
--mode:          operation mode, FULL for complete dump, SAMPLE for first 3000 lines
--train-file:    path to training file
--valid-file:    path to validation file
--eval-file:     path to evaluation file
--glove-file:    path to glove emdedding file
--max-query-len: maximum length of query to be processed
--max-pass-len:  maximum length of passage to be processed
--prefix:        prefix for this dump iteration
--verbose:       verbosity, (True for yes)
```
