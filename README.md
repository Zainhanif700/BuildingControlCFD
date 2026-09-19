# BuildingControlCFD

This repository goes with the paper:
**Data-driven operator learning for energy-efficient building control**
[arXiv:2504.21243](https://arxiv.org/abs/2504.21243) -- Yuexin Bian, Yuanyuan Shi

The code has three parts: CFD airflow simulation, a neural operator model that learns from that simulation data, and a control strategy that uses the trained model.

<p align="center">
  <img src="images/framework.png" alt="Framework" width="600">
</p>
<p align="center"><em>Figure 1. Our Framework</em></p>

<p align="center">
  <img src="images/predict2.png" alt="1" width="600">
</p>
<p align="center"><em>Figure 2. Predicted indoor airflow distribution from the learned operator model.</em></p>

## Step 1: Clone and Install

```bash
git clone https://github.com/Zainhanif700/BuildingControlCFD.git
cd BuildingControlCFD
pip install -r requirements.txt
```

`requirements.txt` has the exact package versions this project was built with, including `torch`/`dgl` for CUDA 11.8. If your machine has a different CUDA version, check https://github.com/HaoZhongkai/GNOT for how to install matching versions instead.

## Step 2: Download the Dataset

This code needs two data files that are too large to store on GitHub (about 2.7GB together). Download them and put them at these exact paths inside the folder you just cloned:

```
learning/dataset/train_data_norm.pkl   (2,100,505,182 bytes)
learning/dataset/test_data_norm.pkl    (607,608,040 bytes)
```

```bash
mkdir -p learning/dataset
curl -L -o learning/dataset/train_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/train_data_norm.pkl
curl -L -o learning/dataset/test_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/test_data_norm.pkl
```

After downloading, check the file sizes match the numbers above:
```bash
ls -l learning/dataset/
```
If a size doesn't match, the download didn't finish properly -- delete the file and try again.

The trained model checkpoints are already included in this repo (`local/models/`, `learning/data/checkpoints/ensemble_5/`), so you don't need to download those separately.

## Step 3: Check the Results

This step does not train anything -- it just loads our already-trained models and runs them against the test data, so it's quick.

```bash
./run_evaluation.sh        # Linux / macOS
run_evaluation.bat         # Windows
```

By default this checks our own retrained 5-model ensemble (`learning/data/checkpoints/ensemble_5/`) and prints each model's error plus the ensemble's error. The paper reports 10.90% for the ensemble on this same test set (Table 3), so you can compare directly.

To check the paper authors' own checkpoints instead:
```bash
./run_evaluation.sh ../local/models/*.pt
```

Both commands are just a shortcut for `python learning/evaluate_ensemble.py <ckpt1.pt> ... <ckpt5.pt>`, which you can also run directly with any set of checkpoint files.

## Optional: Train From Scratch

Training builds a new model from the data instead of using the checkpoints already in this repo. The paper reports this takes about 16 GPU-hours in total for all 5 models (on 2x RTX 2080 Ti) -- your hardware may be faster or slower.

This trains one model per run. We used seeds 2023, 2024, 2025, 2026, and 2027 for our 5-model ensemble:

```bash
cd learning
python train.py --seed 2023
python train.py --seed 2024
python train.py --seed 2025
python train.py --seed 2026
python train.py --seed 2027
```

Then put the 5 resulting checkpoint files together in one folder.

**Important:** your new checkpoints are saved as new files in `learning/data/checkpoints/`, separate from the `ensemble_5/` folder already in this repo. The commands above and below default to `ensemble_5/`, which is our result, not yours. To check or visualize your own trained model instead of ours, point the commands at your new files (run from the repo root, then `cd learning` for the last one):

```bash
ls -t learning/data/checkpoints/*.pt | head -5
./run_evaluation.sh data/checkpoints/<your_files>.pt
cd learning
python visualize_prediction.py data/checkpoints/<your_file>.pt --out my_prediction.png
```

## Optional: Visualize a Prediction (Uses Our Model by Default)

This produces one image showing the true CO2 levels next to the model's predicted CO2 levels for one test sample, along with the error between them -- similar to Figure 5 in the paper. By default it uses our included model (`ensemble_5`), not a model you trained yourself -- see the note above if you trained your own and want to see that instead.

```bash
cd learning
python visualize_prediction.py data/checkpoints/ensemble_5/*.pt --out prediction.png
```

## Optional: Run the CFD Simulation

We already provide simulated data for seeds 0 to 300. You only need this step if you want more data:
```bash
python simulation/transient_simulation.py --seed 0
```

## Optional: Control Strategy

Uses the trained model to plan ventilation control, and lets you look at the results:
```bash
control/control_optimization.ipynb
control/visualize.ipynb
```

## Dataset Details

The full dataset (including raw simulation data, not just the two files from Step 2) is on [Hugging Face](https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset).

- Simulation tool: ANSYS FLUENT 2023R2
- Data types: steady-state and transient flow simulations
- Domain: indoor airflow and CO2 concentration in ventilated buildings

## License and Citation

The dataset and code are for research use only.

```bibtex
@article{bian2025data,
  title={Data-driven operator learning for energy-efficient building control},
  author={Bian, Yuexin and Shi, Yuanyuan Shi},
  journal={arXiv preprint arXiv:2504.21243},
  year={2025}
}
```

## Contact

Original authors: [Email](yubian@ucsd.edu)
