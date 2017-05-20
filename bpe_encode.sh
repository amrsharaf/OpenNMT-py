NAME=$1

: ${PRE_PROCESS_DIR:=.}
: ${TRAIN_DATA_DIR:=./data/amazon}
: ${VALID_DATA_DIR:=./data/wmt15-de-en}
: ${SCRIPT_DIR:=./scripts}
: ${BPE_DIR:=./data/subword-nmt}
: ${SRC_BPE_CODE:=./data/wmt15-de-en/de.bpe_code}
: ${TGT_BPE_CODE:=./data/wmt15-de-en/en.bpe_code}

python $BPE_DIR/apply_bpe.py -c $SRC_BPE_CODE < $TRAIN_DATA_DIR/de_$NAME.txt > $TRAIN_DATA_DIR/de_$NAME.bpe
python $BPE_DIR/apply_bpe.py -c $TGT_BPE_CODE < $TRAIN_DATA_DIR/en_$NAME.txt > $TRAIN_DATA_DIR/en_$NAME.bpe

python $PRE_PROCESS_DIR/preprocess.py \
    -train_src $TRAIN_DATA_DIR/de_$NAME.bpe \
    -train_tgt $TRAIN_DATA_DIR/en_$NAME.bpe \
    -valid_src $VALID_DATA_DIR/valid.de.tok.bpe \
    -valid_tgt $VALID_DATA_DIR/valid.en.tok.bpe \
    -save_data $TRAIN_DATA_DIR/all_${NAME}.bpe -lower
