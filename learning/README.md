# Code of the paper (Bian & Shi 2025)

The folders `learning/`, `simulation/`, `control/`, `local/` and `scripts/` are the code of Bian & Shi (2025),
*Data-driven operator learning for energy-efficient building control* (arXiv:2504.21243). To reproduce their results I
added evaluation and plotting scripts (`evaluate_ensemble.py`, `evaluate_ensemble_strict.py`, `visualize_prediction.py`,
`../scripts/run_evaluation.sh`) and my own retrained 5-model ensemble (`data/checkpoints/ensemble_5/`).

They need the authors' dataset (about 2.7 GB), run from the main folder of the repository:

```bash
mkdir -p learning/dataset
curl -L -o learning/dataset/train_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/train_data_norm.pkl
curl -L -o learning/dataset/test_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/test_data_norm.pkl
```

Then `scripts/run_evaluation.sh` evaluates the included 5-model ensemble (the paper reports 5.9 % train and 10.9 % test
error; my reproduction of the test error is 11.46 %), `learning/train.py --seed <n>` trains a model,
`simulation/transient_simulation.py` runs their simulation and `control/control_optimization.ipynb` the control
strategy. More details are in the authors' repository
([github.com/alwaysbyx/BuildingControlCFD](https://github.com/alwaysbyx/BuildingControlCFD)).

```bibtex
@article{bian2025data,
  title={Data-driven operator learning for energy-efficient building control},
  author={Bian, Yuexin and Shi, Yuanyuan},
  journal={arXiv preprint arXiv:2504.21243},
  year={2025}
}
```
