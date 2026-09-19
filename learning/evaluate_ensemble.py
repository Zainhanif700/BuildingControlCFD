"""
Evaluates your own 5-model ensemble on both the training and test sets, the
same way the paper's Table 3 does: run all 5 independently-trained models on
each sample, average their predictions (Eq. 9a), then compute the
relative-L2 metric on that averaged prediction.

Uses the exact same metric class (WeightedLpRelLoss, via get_loss_func('rel2'))
that train.py's validate_epoch() uses to print "val metric" / "best val" during
training -- so the numbers this script prints are directly comparable to the
per-model numbers you already have, and to the paper's Table 3.

Usage (run from the learning/ folder, same conda env used for training):
    python evaluate_ensemble.py <ckpt1.pt> <ckpt2.pt> <ckpt3.pt> <ckpt4.pt> <ckpt5.pt>
"""
import sys
import torch
import numpy as np

from data_utils import get_model, MIODataset, MIODataLoader, get_loss_func


def evaluate(loader, models, metric_func):
    """Run all models (plus their average) over one data loader, no training.
    This is the exact same loop that already worked for the test set --
    just factored out so it can also run on the training set."""
    per_model_metric = [[] for _ in models]
    ensemble_metric = []
    with torch.no_grad():
        for data in loader:
            g, u_p, g_u = data
            device = next(models[0].parameters()).device
            g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)
            y = g.ndata["y"].squeeze()

            preds = []
            for i, m in enumerate(models):
                out = m(g, u_p, g_u)
                pred = out[0].squeeze() if isinstance(out, tuple) else out.squeeze()
                preds.append(pred)
                _, _, metric = metric_func(g, pred, y)
                per_model_metric[i].append(metric)

            ensemble_pred = torch.stack(preds, dim=0).mean(dim=0)
            _, _, metric = metric_func(g, ensemble_pred, y)
            ensemble_metric.append(metric)
    return per_model_metric, ensemble_metric


def print_results(title, checkpoint_paths, seeds, per_model_metric, ensemble_metric):
    print(f"\n=== {title} relative L2 error (%), same metric as Table 3 ===")
    for i, (p, s) in enumerate(zip(checkpoint_paths, seeds)):
        val = np.mean(per_model_metric[i], axis=0)
        val_scalar = np.mean(val) if hasattr(val, "__len__") else val
        print(f"Model {i+1} (seed {s}): {100*val_scalar:.2f}%")
    ens = np.mean(ensemble_metric, axis=0)
    ens_scalar = np.mean(ens) if hasattr(ens, "__len__") else ens
    print(f"Your 5-model ensemble: {100*ens_scalar:.2f}%")


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_ensemble.py <ckpt1.pt> <ckpt2.pt> ... <ckptN.pt>")
        print("(pass all 5 of your checkpoint paths)")
        sys.exit(1)

    checkpoint_paths = sys.argv[1:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating {len(checkpoint_paths)} models as an ensemble:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    checkpoints = [torch.load(p, map_location=device) for p in checkpoint_paths]
    seeds = [c["args"].seed for c in checkpoints]
    print(f"Seeds found: {seeds}")
    if len(set(seeds)) != len(seeds):
        print(f"WARNING: duplicate seeds in this list ({seeds}) -- "
              f"these aren't all independent models, the ensemble average "
              f"will be biased toward whichever seed repeats.")

    args = checkpoints[0]["args"]

    train_dataset = MIODataset(
        "dataset/train_data_norm.pkl", name="co2", train=True, train_num=args.train_num,
        sort_data=args.sort_data, normalize_y=args.use_normalizer, normalize_x=args.normalize_x,
    )
    test_dataset = MIODataset(
        "dataset/test_data_norm.pkl", name="co2", train=False, test_num=args.test_num,
        sort_data=args.sort_data,
        normalize_y=args.use_normalizer, normalize_x=args.normalize_x,
        y_normalizer=train_dataset.y_normalizer, x_normalizer=train_dataset.x_normalizer,
        up_normalizer=train_dataset.up_normalizer,
    )
    # shuffle=False for both -- order doesn't affect evaluation, and keeps
    # results directly comparable run to run.
    train_loader = MIODataLoader(train_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)
    test_loader = MIODataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)

    normalizer = args.normalizer.to(device) if getattr(args, "normalizer", None) is not None else None
    metric_func = get_loss_func(name="rel2", args=args, regularizer=False, normalizer=normalizer)

    models = []
    for ckpt in checkpoints:
        m = get_model(ckpt["args"]).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)

    train_per_model, train_ensemble = evaluate(train_loader, models, metric_func)
    test_per_model, test_ensemble = evaluate(test_loader, models, metric_func)

    print_results("Training-set", checkpoint_paths, seeds, train_per_model, train_ensemble)
    print_results("Test-set", checkpoint_paths, seeds, test_per_model, test_ensemble)

    print("\nPaper's reported ensemble (Table 3) -- train: 5.9%, test: 10.90%")


if __name__ == "__main__":
    main()
