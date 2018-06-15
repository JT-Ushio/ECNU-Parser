# Dependency Parsing

### Stanford Biaffine Parser

We implement the parser described in the paper [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734).

This code only acheived UAS 95.27% on the standard PTB dataset, though the original code acheived UAS 95.59%. We still need to continue to improve performance.

#### Usage (by examples)

##### Train

```bash
cd src
python train_autobatch.py --config_file ../configs/biaffine.cfg --model biaffine --gpu gpu_id
```

##### Test

```bash
cd src
python test.py --config_file ../configs/biaffine.cfg --model biaffine
```



All configuration options (see in `src/myutils/config.py`) can be specified on the command line, but it's much easier to instead store them in a configuration file like `configs/biaffine.cfg`.