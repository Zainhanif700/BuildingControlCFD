---
license: cc-by-4.0
---

# BEAR-CFD-Dataset

This dataset is released with paper **Data-driven operator learning for energy-efficient and indoor air quality-aware building ventilation control**. 


The data was generated using **ANSYS FLUENT 2023R2** and includes both steady-state and transient flow simulations. It is designed to support research in scientific machine learning, particularly in learning neural operators and data-driven PDE solvers. 
This dataset is released for research purposes only. Please cite the authors if used in published work.

### 📚 BibTeX
```bibtex
@article{bian2025data,
  title={Data-driven operator learning for energy-efficient building control},
  author={Bian, Yuexin and Shi, Yuanyuan},
  journal={arXiv preprint arXiv:2504.21243},
  year={2025}
}
```

## 💻 Code Repository

All code used for data generation, preprocessing, and baseline models is available on GitHub:  
[🔗 GitHub Repository](https://github.com/alwaysbyx/BuildingControlCFD)

## 🧪 Dataset Description

### Steady-State Simulations
- **10 distinct cases** were simulated, each with unique boundary conditions.
- Each simulation converges to a time-independent solution.
- These cases represent equilibrium airflow behavior under varying physical setups.

### Transient Simulations
- **300 cases**, each initialized from a corresponding steady-state solution.
- Each simulation spans **30 minutes** of physical time.
- Simulations use a **time step size of 10 seconds** and output results every 30 seconds.
- This results in **60 time steps per case**, capturing unsteady, time-varying airflow dynamics under changing boundary conditions.

Transient simulation data is used to train neural operators for modeling the evolution of fluid dynamics in time.

## 📂 Dataset Structure

The dataset repository contains the following subfolders:

├── raw_data/ 
    # Transient simulation results in pickle format   
├── processed_data/ # Processed datasets split into train/test sets   
├── models/ # Trained neural operator models   
├── steady_case_data/ # Original ANSYS steady-state result files


- **Pickle Files (`.pkl`)**: Each transient simulation case is stored as a pickle file for convenient loading in Python.
- **ANSYS Data**: The steady-state simulation results are in native ANSYS formats (e.g., `.cas.h5`, `.dat.h5`).

## 📚 Use Cases

- Training neural operators to learn the temporal dynamics of fluid flow.
- Studying how steady-state solutions influence transient behavior.
- Benchmarking data-driven PDE solvers and physics-informed machine learning models.

## 📥 Loading Example

```python
import pickle

with open("raw_data/unsteady_10.pkl", "rb") as f:
    data = pickle.load(f)

import ansys.fluent.core as pyfluent

solver = pyfluent.launch_fluent(mode="solver", precision="double",processor_count=6)
solver.file.read_case_data(file_name='steady_case_data/steady_seed0.cas.h5')
solver.settings.file.read_data(file_name = "steady_case_data/steady_seed0.dat.h5")