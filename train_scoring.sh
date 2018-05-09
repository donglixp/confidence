PWD_DIR=$(pwd)
DATASET=$1


python utils/analysis.py -dataset $DATASET -metric spearmanr -eval_type "omit_non_eng" -model "$PWD_DIR/data_model/$DATASET/model.pt" -conf_model lr -group nomodel,nodata,noinput,all
