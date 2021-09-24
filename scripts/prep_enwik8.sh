#!/bin/bash

DATASET='enwik8'
if [[ ! -d $DATASET ]]; then
    mkdir -p $DATASET
    cd $DATASET
    wget --continue http://mattmahoney.net/dc/enwik8.zip
    wget --continue https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget --continue https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    unzip enwik8.zip
    cd ..

    python3 tools/text2json.py \
        --input-file "$PWD/$DATASET/enwik8" \
        --output-prefix "$PWD/$DATASET/$DATASET"

    cp tools/preprocess_data.py ./
    python3 preprocess_data.py \
        --input "$PWD/$DATASET/$DATASET.json" \
        --tokenizer-type GPT2BPETokenizer \
        --output-prefix "$PWD/$DATASET/$DATASET" \
        --vocab-file "$PWD/$DATASET/gpt2-vocab.json" \
        --merge-file "$PWD/$DATASET/gpt2-merges.txt"
    rm preprocess_data.py
fi
