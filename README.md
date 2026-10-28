# TimeHAT

Code for **Learning from Points and Patches: A Pre-trained Hybrid Attention Transformer for Time Series Analysis**.

TimeHAT combines point-and-patch masking with a Hybrid Attention Encoder for time-series representation learning. This update provides the model, pre-training, and long-term forecasting implementations. 

## Setup

Use Python 3 and a CUDA-enabled PyTorch 2.x installation suitable for your GPU. Install the remaining dependencies:

```bash
pip install numpy pandas scikit-learn scipy matplotlib tqdm
```

Run commands from the repository root. The current training entry point requires an NVIDIA GPU; pre-training uses BF16 by default (`--no_bf16` disables it).

## Data

Datasets are not included. Place the relevant CSV file under `--root_path` (for example, `dataset/ETT-small/ETTh1.csv`). The filename is derived from `--data` (for example, `--data weather` loads `weather.csv`). Forecasting CSV files should contain a `date` column and numeric series columns.


## Usage

Example pre-training command:

```bash
python run.py --task_name pre_training --data ETTh1 \
  --root_path ./dataset/ETT-small/ --seq_len 672 --patch_len 48 \
  --batch_size 32 --train_epochs 10 --gpu 0
```

Keep `seq_len` divisible by `patch_len`. The current pre-training sampler requires at least nine patches. Pre-training checkpoints are saved under `checkpoints/` as `checkpoint_step_*.pth`; training automatically resumes when matching checkpoints exist.

Example forecasting command:

```bash
python run.py --task_name long_term_forecast --data ETTh1 \
  --root_path ./dataset/ETT-small/ --seq_len 672 --patch_len 48 \
  --pred_len 96 --batch_size 32 --train_epochs 10 --gpu 0
```

To initialize forecasting from a pre-trained checkpoint, append:

```bash
--adaptation --pretrain_model_path /path/to/checkpoint_step_N.pth
```

Use matching model dimensions for pre-training and adaptation. Forecasting saves its best checkpoint under `checkpoints/` and reports MSE and MAE. 

## Main files

- `models/TimeHAT.py`: model and masking objectives.
- `layers/encoder.py`: attention and encoder layers.
- `exp/exp_pre_train.py`: pre-training loop and checkpointing.
- `exp/exp_long_term_forecasting.py`: forecasting and adaptation.
- `data_provider/`: data loading and preprocessing.
- `run.py`: command-line entry point.

## Contact

Peiwang Tang: [tpw@mail.ustc.edu.cn](mailto:tpw@mail.ustc.edu.cn).

