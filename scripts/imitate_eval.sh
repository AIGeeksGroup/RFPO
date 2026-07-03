#!/bin/bash
algo=${1}
model_dir=${2}
shift
python tools/imitate_eval.py --config-name=${algo} checkpoint.eval_dir=${model_dir} ${@:2}
