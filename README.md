# Custom I3D models

Forked from the original repo of  "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" paper by Joao Carreira and Andrew
Zisserman.

Take a look at the original [README.md](https://github.com/quental96/kinetics-i3d/blob/master/ORIGINAL.md) here.

## Custom files

This repo aims at widening the original use of Deepmind's I3D repo which gives only logits and probabilies on sample videos. Furthermore, it tests a continuous run of the I3D on entire datasets (note that this can be greatly improved).

## Example use

**Input**

```bash
python custom.py --imagenet_pretrained true --eval_type joint --final_endpoint Mixed_4f --path path --flow_path path --save_path path --frames 99
```
