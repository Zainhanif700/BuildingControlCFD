"""
Visualize a ground-truth vs. predicted CO2 concentration field for one test
sample, in the style of Figure 5 in the paper: ground truth | ensemble
prediction | relative error, side by side.

Usage (run from the learning/ folder, same as evaluate_ensemble.py):
    python visualize_prediction.py <ckpt1.pt> ... <ckpt5.pt> [--sample-idx N] [--timestep T] [--out prediction.png]

Averages predictions across however many checkpoints you pass (1 for a
single model, 5 for the full ensemble). Saves a 3-panel PNG.

NOTE: written by reading data_utils.py/utils.py carefully (confirmed
UnitTransformer.transform()'s signature and MIODataset.__getitem__'s return
shape), but NOT executed -- no torch/dgl/matplotlib available where this was
written. Please run it and report back the exact error if anything breaks --
most likely culprit: whether coords[:, 0], coords[:, 1] (first two spatial
dims) is actually the most informative 2D projection of this room's mesh,
since I haven't seen the real coordinate data.
"""
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import get_model, MIODataset, MIODataLoader


def main():
    parser = argparse.ArgumentParser(description="Visualize ground-truth vs. predicted CO2 field (paper Figure 5 style).")
    parser.add_argument("checkpoints", nargs="+", help="One or more checkpoint .pt files to average as the prediction.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Which test-set sample (graph) to visualize.")
    parser.add_argument("--timestep", type=int, default=-1, help="Which of the 6 future timesteps to plot (-1 = last).")
    parser.add_argument("--out", type=str, default="prediction.png", help="Output image path.")
    cli = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoints = [torch.load(p, map_location=device) for p in cli.checkpoints]
    train_args = checkpoints[0]["args"]

    train_dataset = MIODataset(
        "dataset/train_data_norm.pkl", name="co2", train=True, train_num=train_args.train_num,
        sort_data=train_args.sort_data, normalize_y=train_args.use_normalizer, normalize_x=train_args.normalize_x,
    )
    test_dataset = MIODataset(
        "dataset/test_data_norm.pkl", name="co2", train=False, test_num=train_args.test_num,
        sort_data=train_args.sort_data, normalize_y=train_args.use_normalizer, normalize_x=train_args.normalize_x,
        y_normalizer=train_dataset.y_normalizer, x_normalizer=train_dataset.x_normalizer,
        up_normalizer=train_dataset.up_normalizer,
    )
    if cli.sample_idx >= len(test_dataset):
        raise IndexError(f"--sample-idx {cli.sample_idx} out of range (test set has {len(test_dataset)} samples)")

    models = []
    for ckpt in checkpoints:
        m = get_model(ckpt["args"]).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)

    single_loader = MIODataLoader([test_dataset[cli.sample_idx]], batch_size=1, shuffle=False, drop_last=False)
    g, u_p, g_u = next(iter(single_loader))
    g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

    with torch.no_grad():
        preds = []
        for m in models:
            out = m(g, u_p, g_u)
            pred = out[0].squeeze() if isinstance(out, tuple) else out.squeeze()
            preds.append(pred)
        ensemble_pred = torch.stack(preds, dim=0).mean(dim=0)

    y_true_norm = g.ndata["y"].squeeze()
    coords_norm = g.ndata["x"].squeeze()

    y_normalizer = test_dataset.y_normalizer.to(device) if test_dataset.y_normalizer is not None else None
    x_normalizer = test_dataset.x_normalizer.to(device) if test_dataset.x_normalizer is not None else None

    y_true = (y_normalizer.transform(y_true_norm, inverse=True) if y_normalizer is not None else y_true_norm).detach().cpu().numpy()
    y_pred = (y_normalizer.transform(ensemble_pred, inverse=True) if y_normalizer is not None else ensemble_pred).detach().cpu().numpy()
    coords = (x_normalizer.transform(coords_norm, inverse=True) if x_normalizer is not None else coords_norm).detach().cpu().numpy()

    t = cli.timestep
    gt_t, pred_t = y_true[:, t], y_pred[:, t]
    rel_err = np.abs(gt_t - pred_t) / (np.abs(gt_t) + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, title, val in zip(axes, ["Ground Truth CO2 (ppm)", "Ensemble Prediction CO2 (ppm)", "Relative Error"], [gt_t, pred_t, rel_err]):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=val, cmap="viridis", s=15)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(sc, ax=ax)
    plt.tight_layout()
    plt.savefig(cli.out, dpi=150)
    print(f"Saved figure to {cli.out}")


if __name__ == "__main__":
    main()
