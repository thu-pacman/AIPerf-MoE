MoE Bench
===

MoE Bench is a benchmark for systems to train large AI models.
It aims at measuring both computation and communication power of a system.

## Benchmarking guide

### Install prerequisites

**TOOD**

### Prepare the data

The training data requires preprocessing. An example script using `enwik8` to prepare data for is [scripts/prep_enwik8.sh](./scripts/prep_enwik8.sh).

If you want to preprocess text data of your own, you may first use [tools/text2json.py](./tools/text2json.py) to split a large text document into samples in a loose json format. For Example:

```json
{"text": "The quick brown fox"}
{"text": "jumps over the lazy dog"}
```

Of course, you may create your own json file in different ways. Note that only the `text` field of the json will be used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use [tools/preprocess_data.py](./tools/preprocess_data.py). Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for training is:

```bash
python3 tools/preprocess_data.py \
    --input my-corpus.json \
    --tokenizer-type GPT2BPETokenizer \
    --output-prefix my-corpus \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt
```

The output will be two files named, in this case, `my-corpus_text_sentence.bin` and `my-corpus_text_sentence.idx`. The `--data-path` specified in later training is the full path and new filename, but without the file extension.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

### Configure and start a test

**TODO**

### Scoring
