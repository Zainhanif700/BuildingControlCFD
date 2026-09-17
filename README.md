# BuildingControlCFD

This repository accompanies the paper:
**_Data-driven operator learning for energy-efficient building control_**
[arXiv:2504.21243](https://arxiv.org/abs/2504.21243) -- Yuexin Bian, Yuanyuan Shi

The code integrates CFD-based airflow simulation, operator learning (neural operator models), and optimization-based control with neural operators to enable energy-efficient and ventilation control.

<p align="center">
  <img src="images/framework.png" alt="Framework" width="600">
</p>
<p align="center"><em>Figure 1. Our Framework</em></p>

<p align="center">
  <img src="images/predict2.png" alt="1" width="600">
</p>
<p align="center"><em>Figure 2. Predicted indoor airflow distribution from the learned operator model. Proposed Ensemble Neural Operator increases accuracy and facilitates downstream control</em></p>

## ⚙️ Step 1: Clone and Install

```bash
git clone https://github.com/Zainhanif700/BuildingControlCFD.git
cd BuildingControlCFD
pip install -r requirements.txt
```

`requirements.txt` includes the exact pinned `torch`/`dgl` CUDA builds this project was developed with (CUDA 11.8) -- if your machine uses a different CUDA version, see https://github.com/HaoZhongkai/GNOT for how to pick matching builds instead.

## ⚠️ Step 2: Download the Dataset

Run this from inside the `BuildingControlCFD` folder you just cloned (i.e., right after Step 1, same terminal / same directory). This repo's code needs two files (~2.7GB combined -- too large for GitHub itself) at exactly these paths, relative to the repo root:

Download and place them there:

```bash
mkdir -p learning/dataset
curl -L -o learning/dataset/train_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/train_data_norm.pkl
curl -L -o learning/dataset/test_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/test_data_norm.pkl
```

Verify the download by checking file sizes match exactly (above) -- a truncated/corrupted download is the most common cause of a failed run:
```bash
ls -l learning/dataset/
```

The pretrained checkpoints needed for evaluation are already included in this repo (`local/models/`, `learning/data/checkpoints/ensemble_5/`) -- no separate download needed for those.

## 🚀 Step 3: Reproduce Ensemble Results (No Training Needed)

Evaluates our own retrained 5-model ensemble (`learning/data/checkpoints/ensemble_5/`) on the held-out test set and prints each model's test error plus the ensemble average -- same metric as the paper's Table 3 (reported ensemble test error: **10.90%**). No CFD, no training (~30 GPU-hours for 5 models) -- just evaluation, minutes not hours.

```bash
./run_evaluation.sh        # Linux / macOS
run_evaluation.bat         # Windows
```

To instead evaluate the paper authors' own pretrained checkpoints (`local/models/`):
```bash
./run_evaluation.sh ../local/models/*.pt
```

Both wrap `python learning/evaluate_ensemble.py <ckpt1.pt> ... <ckpt5.pt>`, which you can also call directly with any 5 checkpoint paths.

## Full Pipeline (Optional)

### 1. Run CFD Simulation
We already provide the dataset for seed = 0 to 300. To generate additional transient airflow fields for a building geometry:
```bash
python simulation/transient_simulation.py --seed 0
```

### 2. Train Neural Operator
Train data-driven surrogate models on CFD data from scratch:
```bash
python learning/train.py
```

### 3. Optimize Control Strategy and Visualize
```bash
control/control_optimization.ipynb
control/visualize.ipynb
```

## 📂 Dataset Access

Full dataset (including raw simulation data and steady-state cases, not just the normalized files above) is on [Hugging Face 🤗 Datasets](https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset).

- **Simulation Tool:** ANSYS FLUENT 2023R2
- **Data Types:** Steady-state and transient (time-dependent) flow simulations
- **Domain:** Indoor air flow and CO₂ concentration in ventilated building environments
- **Applications:** Neural operator learning, spatiotemporal modeling, model-based HVAC control

## 📜 License & Citation

The dataset and code are released for **research purposes only**.

```bibtex
@article{bian2025data,
  title={Data-driven operator learning for energy-efficient building control},
  author={Bian, Yuexin and Shi, Yuanyuan Shi},
  journal={arXiv preprint arXiv:2504.21243},
  year={2025}
}
```

## 📫 Contact

Original authors: [Email](yubian@ucsd.edu)
