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
./train_lm.sh ifttt
./train_lm.sh django
```

### Precompute all the confidence metrics, and perform uncertainty backpropagation

(This step can be skipped. The precomputed data have been cached in the zip file.)

```sh
# arg2: gpu id
./compute_metric.sh ifttt 0
./compute_metric.sh django 0
```

The intermediate results are saved to "data_model/[ifttt|django]/*.eval".

## Usage

### Confidence Estimation

- Train a confidence scoring model.
- Compute spearman's rho between confidence scores and F1 for the full model and ablation models (w/o model uncertainty, w/o data uncertainty, and w/o input uncertainty).

```sh
./train_scoring.sh ifttt
./train_scoring.sh django
```

### Uncertainty Interpretation

Evaluate confidence backpropagation and attention-based method against inferred ground truth.

```sh
./eval_conf_bp_att.sh ifttt
./eval_conf_bp_att.sh django
```

## Acknowledgments

- The implementation is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- [XGBoost](https://github.com/dmlc/xgboost): gradient tree boosting model.
- [KenLM](https://github.com/kpu/kenlm): language model.
