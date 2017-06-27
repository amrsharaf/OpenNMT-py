DATA_DIR=data/wmt15-de-en
S='de'
T='en'
python preprocess.py -train_src $DATA_DIR/train.${S}.tok.bpe -train_tgt $DATA_DIR/train.${T}.tok.bpe -valid_src $DATA_DIR/valid.${S}.tok.bpe -valid_tgt $DATA_DIR/valid.${T}.tok.bpe -save_data $DATA_DIR/all_bpe -lower
