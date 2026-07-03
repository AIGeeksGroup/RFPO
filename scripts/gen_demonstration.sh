checkpoint_dir=${1}
num_trajs=${2}

python tools/generate_expert_trajs.py --checkpoint_dir=${checkpoint_dir} --num_trajs=${num_trajs}
