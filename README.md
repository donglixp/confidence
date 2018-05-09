# [Confidence Modeling for Neural Semantic Parsing](http://homepages.inf.ed.ac.uk/s1478528/acl18-confidence.pdf)

## Setup

### Requirements

- Python 2.7
- [PyTorch 0.1.12.post2](https://pytorch.org/previous-versions/) (GPU)

### Install Python dependency

```sh
pip install -r requirements.txt
```

### Install KenLM

```sh
./install_kenlm.sh
```

### Download data and pretrained model

Download the zip file from [Google Drive](https://drive.google.com/file/d/1g1uogoj8Aw2f1RYOxCC7thwloWfe4sjO/view?usp=sharing), and copy it to the folder of code.

```sh
unzip acl18confidence_data_model.zip
```

## Precompute Confidence Metrics

### Train a language model

```sh
./train_lm.sh [ifttt|django]
```

### Precompute all the confidence metrics, and perform uncertainty backpropagation

(This step can be skipped. The precomputed data have been cached in the zip file.)

```sh
# arg2: use gpu 0
./compute_metric.sh [ifttt|django] 0
# The intermediate results are saved to "data_model/*/*.eval".
```

## Usage

### Train a confidence scoring model, and compute spearman's rho between confidence scores and F1

```sh
# Evaluate full model and ablation models (w/o model uncertainty, w/o data uncertainty, and w/o input uncertainty).
./train_scoring.sh [ifttt|django]
```

### Evaluate confidence backpropagation and attention-based method against inferred ground truth on the development set

```sh
./eval_conf_bp_att.sh [ifttt|django]
```

## Acknowledgments

- The implementation is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- [XGBoost](https://github.com/dmlc/xgboost): gradient tree boosting model.
- [KenLM](https://github.com/kpu/kenlm): language model.
