BPE_DIR=./data/subword-nmt
# helper function that encodes a file with bpe
encode () {
    if [ ! -f "$3" ]; then
        echo "Encoding $3..."
        $PYTHON $BPE_DIR/apply_bpe.py -c $1 < $2 > $3 
    else
        echo "$3 exists, pass"
    fi
}

SRC_BPE_DIR=data/wmt15-de-en/de.bpe_code
TGT_BPE_DIR=data/wmt15-de-en/en.bpe_code
DATA_DIR=data/sup_critic

# encode data with BPE
encode $SRC_BPE_DIR $DATA_DIR/train.src $DATA_DIR/train.src.bpe
encode $SRC_BPE_DIR $DATA_DIR/valid.src $DATA_DIR/valid.src.bpe
encode $SRC_BPE_DIR $DATA_DIR/all_train.src $DATA_DIR/all_train.src.bpe
encode $SRC_BPE_DIR $DATA_DIR/all.src $DATA_DIR/all.src.bpe

encode $TGT_BPE_DIR $DATA_DIR/train.tgt $DATA_DIR/train.tgt.bpe
encode $TGT_BPE_DIR $DATA_DIR/valid.tgt $DATA_DIR/valid.tgt.bpe
encode $TGT_BPE_DIR $DATA_DIR/all_train.tgt $DATA_DIR/all_train.tgt.bpe
encode $TGT_BPE_DIR $DATA_DIR/all.tgt $DATA_DIR/all.tgt.bpe
