#!/bin/bash
algo=${1}
seq_name=${2}
shift
python tools/imitate_train.py --config-name=${algo} task.name=${seq_name} ${@:2}
