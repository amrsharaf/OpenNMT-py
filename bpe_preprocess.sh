DATA_DIR=data/baseline-1M-ende
SCRIPT_DIR=./scipts
BPE_DIR=data/subword-nmt
# BPE size
SRC_CODE_SIZE=20000
TRG_CODE_SIZE=20000
S='de'
T='en'

PYTHON=python
#if [ -z $PYTHON ]; then
#    if [ -n `which python3` ]; then
#        export PYTHON=python3
#    else
#        if [ -n `which python`]; then
#            export PYTHON=python
#        fi
#    fi 
#fi
echo "Using $PYTHON"

# get byte pair encoding
BPE_REPO=https://github.com/rsennrich/subword-nmt
if [ ! -d "${BPE_DIR}" ]; then
    echo "Cloning BPE central ..."
    mkdir -p ${BPE_DIR}
    git clone ${BPE_REPO} ${BPE_DIR}
fi

encode () {
    if [ ! -f "$3" ]; then
        $PYTHON $P2/apply_bpe.py -c $1 < $2 > $3 
    else
        echo "$3 exists, pass"
    fi
}

# tokenize
if [ ! -f $DATA_DIR/train.${S}.tok ]; then
    echo "Tokenizing train source..."
    perl $SCRIPT_DIR/tokenizer.perl -threads 5 -l ${S} < $DATA_DIR/train.${S} > $DATA_DIR/train.${S}.tok
fi

if [ ! -f $DATA_DIR/train.${T}.tok ]; then
    echo "Tokenizing train target..."
    perl $SCRIPT_DIR/tokenizer.perl -threads 5 -l ${T} < $DATA_DIR/train.${T} > $DATA_DIR/train.${T}.tok
fi

if [ ! -f $DATA_DIR/valid.${S}.tok ]; then
    echo "Tokenizing valid source..."
    perl $SCRIPT_DIR/tokenizer.perl -threads 5 -l ${S} < $DATA_DIR/valid.${S} > $DATA_DIR/valid.${S}.tok
fi

if [ ! -f $DATA_DIR/valid.${T}.tok ]; then
    echo "Tokenizing valid target..."
    perl $SCRIPT_DIR/tokenizer.perl -threads 5 -l ${T} < $DATA_DIR/valid.${T} > $DATA_DIR/valid.${T}.tok
fi

if [ ! -f $DATA_DIR/test.${S}.tok ]; then
    echo "Tokenizing test source..."
    perl $SCRIPT_DIR/tokenizer.perl -threads 5 -l ${S} < $DATA_DIR/test.${S} > $DATA_DIR/test.${S}.tok
fi

if [ ! -f  $DATA_DIR/test.${T}.tok ]; then
    echo "Tokenizing test target..."
    perl $SCRIPT_DIR/tokenizer.perl -threads 5 -l ${T} < $DATA_DIR/test.${T} > $DATA_DIR/test.${T}.tok
fi

# learn BPE coding from training set
if [ ! -f "$DATA_DIR/${S}.bpe_code" ]; then
    echo "Learning source BPE..."
    $PYTHON $BPE_DIR/learn_bpe.py -s 20000 < $DATA_DIR/train.${S}.tok > $DATA_DIR/${S}.bpe_code
fi
if [ ! -f "$DATA_DIR/${T}.bpe_code" ]; then
    echo "Learning target BPE..."
    $PYTHON $BPE_DIR/learn_bpe.py -s 20000 < $DATA_DIR/train.${T}.tok > $DATA_DIR/${T}.bpe_code
fi

# helper function that encodes a file with bpe
encode () {
    if [ ! -f "$3" ]; then
        echo "Encoding $3..."
        $PYTHON $BPE_DIR/apply_bpe.py -c $1 < $2 > $3 
    else
        echo "$3 exists, pass"
    fi
}

# encode data with BPE
encode $DATA_DIR/${S}.bpe_code $DATA_DIR/train.${S}.tok $DATA_DIR/train.${S}.tok.bpe
encode $DATA_DIR/${T}.bpe_code $DATA_DIR/train.${T}.tok $DATA_DIR/train.${T}.tok.bpe
encode $DATA_DIR/${S}.bpe_code $DATA_DIR/valid.${S}.tok $DATA_DIR/valid.${S}.tok.bpe
encode $DATA_DIR/${T}.bpe_code $DATA_DIR/valid.${T}.tok $DATA_DIR/valid.${T}.tok.bpe
encode $DATA_DIR/${S}.bpe_code $DATA_DIR/test.${S}.tok $DATA_DIR/test.${S}.tok.bpe
encode $DATA_DIR/${T}.bpe_code $DATA_DIR/test.${T}.tok $DATA_DIR/test.${T}.tok.bpe

# create dictionaries and stuff
python preprocess.py -train_src $DATA_DIR/train.${S}.tok.bpe -train_tgt $DATA_DIR/train.${T}.tok.bpe -valid_src $DATA_DIR/valid.${S}.tok.bpe -valid_tgt $DATA_DIR/valid.${T}.tok.bpe -save_data $DATA_DIR/all_bpe -lower
