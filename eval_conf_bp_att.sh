PWD_DIR=$(pwd)
DATASET=$1

python eval_conf_bp.py -model "$PWD_DIR/data_model/$DATASET/model.pt" -src "$PWD_DIR/data_model/$DATASET/test.src" -metric intersection2
python eval_conf_bp.py -model "$PWD_DIR/data_model/$DATASET/model.pt" -src "$PWD_DIR/data_model/$DATASET/test.src" -metric intersection4
