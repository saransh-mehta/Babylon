# Babylon

## Get Glove dataset
Download Glove Embeddings from [here](http://nlp.stanford.edu/data/glove.6B.zip)

## Run
To create the training and validation data run the following commands from `/common/split_data.py` (this process may take some time). This will create two new files `path/to/output/file/name_train.txt` and `path/to/output/file/name_validation.txt`:

```bash
$ python3 split_data.py --input-file=path/to/input/file --output-file=path/to/output/file/name --ratio=0.1 --randomize=True
```

Once the two files are generated to convert them to CNTK format run the file `/common/text2ctf_mod.py`. `--mode=SAMPLE` gives first 3000 lines, to get the complete data use `--mode=FULL`:

```bash
$ python3 text2ctf_mod.py --mode=SAMPLE --train-file=train.txt --valid-file=validation.txt --eval-file=eval.txt --glove-file=glove.txt --prefix=new 
```

## Tests
