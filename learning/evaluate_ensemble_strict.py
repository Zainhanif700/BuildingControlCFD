"""
Like evaluate_ensemble.py, but it gives two errors for the same models and samples:
  paper   the relative l2 error of the CO2 in ppm including the 400 ppm of outdoor air (Table 3 of the paper)
  strict  the same error of the CO2 above the outdoor air only (CO2 - 400 ppm), which is harsher
The data of the paper hold the absolute CO2 in ppm (checked: targets 396 to 1109 ppm).

Usage (run from the learning/ folder, with the same environment as for the training):
    python evaluate_ensemble_strict.py data/checkpoints/ensemble_5/*.pt
"""
import sys

import numpy as np
import torch

from data_utils import get_model, MIODataset, MIODataLoader, get_loss_func

OUTDOOR = 400.0     # ppm


def raw_values(metric_func, pred, y):
    """Prediction and target as raw CO2 in ppm (the metric class does the same inverse normalisation)."""
    n = metric_func.normalizer
    if n is None:
        return pred, y
    return n.transform(pred, component=metric_func.component, inverse=True), n.transform(y, inverse=True)


def evaluate(loader, models, metric_func):
    """For every model and for the ensemble: the paper-style error and the strict error of each batch."""
    out = {k: [[] for _ in models] for k in ("paper", "strict")}
    ens = {"paper": [], "strict": []}
    with torch.no_grad():
        for g, u_p, g_u in loader:
            device = next(models[0].parameters()).device
            g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)
            y = g.ndata["y"].squeeze()
            preds = []
            for m in models:
                o = m(g, u_p, g_u)
                preds.append(o[0].squeeze() if isinstance(o, tuple) else o.squeeze())
            preds.append(torch.stack(preds, dim=0).mean(dim=0))          # the last entry is the ensemble
            for i, pred in enumerate(preds):
                pr, yr = raw_values(metric_func, pred, y)
                paper = metric_func._lp_losses(g, pr, yr)[1]
                strict = metric_func._lp_losses(g, pr - OUTDOOR, yr - OUTDOOR)[1]
                if i < len(models):
                    out["paper"][i].append(paper)
                    out["strict"][i].append(strict)
                else:
                    ens["paper"].append(paper)
                    ens["strict"].append(strict)
    return out, ens


def mean_percent(batches):
    v = np.mean(batches, axis=0)
    return 100 * float(np.mean(v))


def print_results(title, per_model, ens):
    print(f"\n=== {title}: relative l2 error in %  (paper = with 400 ppm, strict = CO2 above 400 ppm) ===")
    print(f"  {'':12s}{'paper':>9s}{'strict':>9s}")
    for i in range(len(per_model["paper"])):
        print(f"  {'model ' + str(i + 1):12s}{mean_percent(per_model['paper'][i]):9.2f}{mean_percent(per_model['strict'][i]):9.2f}")
    print(f"  {'ensemble':12s}{mean_percent(ens['paper']):9.2f}{mean_percent(ens['strict']):9.2f}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python evaluate_ensemble_strict.py <ckpt1.pt> ... <ckptN.pt>")
    paths = sys.argv[1:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}; {len(paths)} models")
    checkpoints = [torch.load(p, map_location=device) for p in paths]
    args = checkpoints[0]["args"]

    train_dataset = MIODataset(
        "dataset/train_data_norm.pkl", name="co2", train=True, train_num=args.train_num,
        sort_data=args.sort_data, normalize_y=args.use_normalizer, normalize_x=args.normalize_x,
    )
    test_dataset = MIODataset(
        "dataset/test_data_norm.pkl", name="co2", train=False, test_num=args.test_num,
        sort_data=args.sort_data, normalize_y=args.use_normalizer, normalize_x=args.normalize_x,
        y_normalizer=train_dataset.y_normalizer, x_normalizer=train_dataset.x_normalizer,
        up_normalizer=train_dataset.up_normalizer,
    )
    train_loader = MIODataLoader(train_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)
    test_loader = MIODataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)
    normalizer = args.normalizer.to(device) if getattr(args, "normalizer", None) is not None else None
    metric_func = get_loss_func(name="rel2", args=args, regularizer=False, normalizer=normalizer)

    models = []
    for ckpt in checkpoints:
        m = get_model(ckpt["args"]).to(device)
        m.load_state_dict(ckpt["model"])
        models.append(m.eval())

    print_results("TRAIN set", *evaluate(train_loader, models, metric_func))
    print_results("TEST set", *evaluate(test_loader, models, metric_func))
    print("\nThe 'paper' column should be close to the earlier result (test ensemble 11.46 %, paper 10.90 %).")


if __name__ == "__main__":
    main()
