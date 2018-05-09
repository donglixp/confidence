PWD_DIR=$(pwd)
DATASET=$1

mkdir -p $PWD_DIR/tmp
$PWD_DIR/toolkit/kenlm/build/bin/lmplz -o 4 -T $PWD_DIR/tmp <$PWD_DIR/data_model/$DATASET/train.src >$PWD_DIR/data_model/$DATASET/lm.arpa
