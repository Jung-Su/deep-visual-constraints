# Deep Visual Constraints (DVC)

This is a pytorch implimentation of the paper: "Deep Visual Constraints: Neural Implicit Models for Manipulation Planning from Visual Input".

[[Project Page]](https://sites.google.com/view/deep-visual-constraints) [[Paper]](https://arxiv.org/abs/2112.04812) [[Video]](https://youtu.be/r__mIGTu6Jg)

## Requirements
- Pytorch
- Torchvision
- [H5py](https://docs.h5py.org/en/stable/quick.html)
- [Trimesh](https://trimsh.org/trimesh.html)
- Scipy
- Pyglet


## Instruction
1. Download [the pretrained network](https://drive.google.com/drive/folders/1RcjmbazIrejv6QT8cSJ9V62KSbA2ip5k?usp=sharing) files into the folder './network'
2. Download [the dataset](https://drive.google.com/file/d/12Ycx9oJkd8lape1SuQ2k0w75yp0IQ1pF/view?usp=sharing) and extract it to the folder './data' 
3. Run 'visualize_*.ipynb' to visualize data, learend SDFs (& mesh reconstruction), PCAs on learned features, or tasks (optimized grasp/hang poses)
4. Run 'train_PIFO.ipynb' to train the whole framework

## Citation
```
@article{ha2022dvc,
  title={Deep Visual Constraints: Neural Implicit Models for Manipulation Planning from Visual Input},
  author={Ha, Jung-Su and Driess, Danny and Toussaint, Marc},
  journal={arXiv preprint arXiv:2112.04812},
  year={2022}
}
```
