<div align="center">

# ViViDex: Learning Vision-based Dexterous Manipulation from Human Videos

[Zerui Chen](https://zerchen.github.io/)<sup>1</sup> &emsp; [Shizhe Chen](https://cshizhe.github.io/)<sup>1</sup> &emsp; [Etienne Arlaud](https://scholar.google.com/citations?user=-0kdc5cAAAAJ&hl=fr)<sup>1</sup> &emsp; [Ivan Laptev](https://www.di.ens.fr/~laptev/)<sup>2</sup> &emsp; [Cordelia Schmid](https://cordeliaschmid.github.io/)<sup>1</sup>

<sup>1</sup>WILLOW, INRIA Paris, France <br>
<sup>2</sup>MBZUAI

<a href='https://zerchen.github.io/projects/vividex.html'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2404.15709'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
</div>

This is the implementation of **[ViViDex](https://zerchen.github.io/projects/vividex.html)** under the SAPIEN simulator, a novel system for learning dexterous manipulation skills from human videos:
![teaser](assets/teaser.png)

## Installation 👷
```
git clone https://github.com/zerchen/vividex_sapien.git

conda create -n rl python=3.10
conda activate rl
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usuage 🚀
```bash
cd tools
# Train the state-based policy
python train.py env.name=seq_name env.norm_traj=True
```
Available seq_name can be found at: `norm_trajectories`. You can also download trained checkpoints [here](https://drive.google.com/drive/folders/130JTBsDv4I7NytXLMlo3ehxfJI4I75pB) and check their config files for a reference. When state-based policies are trained, rollout these policies with `generate_expert_trajs.py` and train the visual policy with `imitate_train.py` using either BC or diffusion policy.

## Real robot 🤖
Please refer to our UR5 ROS [code](https://github.com/inria-paris-robotics-lab/prl_ur5_robot) and Allegro hand ROS [code](https://github.com/inria-paris-robotics-lab/allegro_hand_ros_v4) as an example to set up the real robot experiment.

## Acknowledgements
Parts of the code are based on [DexArt](https://github.com/Kami-code/dexart-release), [DexPoint](https://github.com/yzqin/dexpoint-release) and [3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy). We thank the authors for sharing their excellent work!

## Citation 📝
If you find ViViDex useful for your research, please consider citing our paper:
```bibtex
@inproceedings{chen2025vividex,
  title={{ViViDex}: Learning Vision-based Dexterous Manipulation from Human Videos},
  author={Chen, Zerui and Chen, Shizhe and Arlaud, Etienne and Laptev, Ivan and Schmid, Cordelia},
  booktitle={ICRA},
  year={2025}
}
```
