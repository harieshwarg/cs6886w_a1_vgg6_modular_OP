# FULL pipeline: 25-run sweep → charts → scatter → curves → final retrain/test → save checkpoints
import os, random, copy, datetime, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import plotly.express as px, plotly.graph_objects as go
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataclasses import dataclass
from itertools import product

# W&B
os.environ["WANDB_MODE"] = "online"
import wandb

from model import VGG6
from data_loader import get_loaders
from utils import set_seed, accuracy, eval_loss_acc, train_one_epoch, build_optimizer, build_scheduler

WANDB_PROJECT = "cs6886w-a1-vgg6-sweep_try2"
WANDB_ENTITY  = None
RUN_GROUP     = "25-run-sweep"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| W&B mode:", os.environ.get("WANDB_MODE"))
set_seed(42)

# ---------------- SWEEP SPACE ----------------
@dataclass
class Config:
    activation: str
    optimizer: str
    lr: float
    batch_size: int
    epochs: int
    init: str = "kaiming"
    augment: bool = True
    batch_norm: bool = True

search_space = {
    "activation": ["relu", "gelu", "silu", "tanh", "sigmoid"],
    "optimizer": ["adam", "nadam", "sgd", "nesterov-sgd", "rmsprop", "adagrad"],
    "lr": [1e-3, 5e-4, 1e-4],
    "batch_size": [64, 128],
    "epochs": [20],
}
keys = list(search_space.keys())
full_grid = [Config(**dict(zip(keys, vals))) for vals in product(*[search_space[k] for k in keys])]

def plausible(cfg: Config):
    adam_like = {"adam", "nadam", "rmsprop", "adagrad"}
    key = str(cfg.optimizer).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in adam_like and cfg.lr > 1e-3:  return False
    if key in {"sgd", "nesterovsgd", "nesterov"} and cfg.lr < 5e-4:  return False
    return True

plausible_grid = [c for c in full_grid if plausible(c)]
random.seed(42)
if len(plausible_grid) < 25:
    raise RuntimeError(f"Not enough plausible configs ({len(plausible_grid)}).")
sweep_25 = random.sample(plausible_grid, 25)
print(f"Total plausible configs: {len(plausible_grid)} | Sampling exactly 25 runs.")

# ---------------- SWEEP + LOGGING ----------------
best = {"val_acc": -1.0, "cfg": None}
rows = []; sweep_rows_epoch = []

for idx, cfg in enumerate(sweep_25, 1):
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=RUN_GROUP,
                     job_type="sweep_run",
                     config=dict(
                         activation=cfg.activation, optimizer=cfg.optimizer, lr=cfg.lr,
                         batch_size=cfg.batch_size, epochs=cfg.epochs, init=cfg.init,
                         augment=cfg.augment, batch_norm=cfg.batch_norm, run_idx=idx
                     ),
                     mode="online", reinit=True)

    created_ts = datetime.datetime.now(datetime.timezone.utc)
    wandb.run.summary["created_ts"] = created_ts.isoformat()

    print(f"\n==> [Run {idx:02d}/25] {cfg}")
    train_loader, val_loader, test_loader, train_set, val_set = get_loaders(
        batch_size=cfg.batch_size, augment=cfg.augment, val_ratio=0.1, num_workers=2
    )

    model = VGG6(num_classes=10, activation=cfg.activation, batch_norm=cfg.batch_norm).to(device)
    model.init_weights(mode=cfg.init, nonlinearity=("relu" if cfg.activation=="relu" else "linear"))

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    scheduler = build_scheduler(optimizer, cfg.optimizer, cfg.epochs)

    use_amp = torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_local = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler if use_amp else None)
        tr_acc  = accuracy(model, train_loader, device)
        va_loss, va_acc = eval_loss_acc(model, val_loader, criterion, device)

        if scheduler: scheduler.step()
        if (epoch % 5 == 0) or (epoch == cfg.epochs):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:02d} | train_loss {tr_loss:.4f} | train_acc {tr_acc:.2f} | val_loss {va_loss:.4f} | val_acc {va_acc:.2f} | lr {lr_now:g}")

        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss,   "val/acc":  va_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        label = f"{cfg.activation}|{cfg.optimizer}|bs{cfg.batch_size}|lr{cfg.lr:g}"
        sweep_rows_epoch.append({
            "run_idx": idx, "label": label, "epoch": epoch,
            "train_loss": float(tr_loss), "val_loss": float(va_loss),
            "train_acc": float(tr_acc),  "val_acc": float(va_acc),
        })
        best_val_local = max(best_val_local, va_acc)

    print("  -> best_val_acc:", f"{best_val_local:.2f}")
    wandb.run.summary["best_val_acc"] = float(best_val_local)

    if best_val_local > best["val_acc"]:
        best["val_acc"] = best_val_local
        best["cfg"] = copy.deepcopy(cfg)

    rows.append(dict(
        run_idx=idx, activation=cfg.activation, optimizer=cfg.optimizer,
        lr=float(cfg.lr), batch_size=int(cfg.batch_size),
        epochs=int(cfg.epochs), best_val_acc=float(best_val_local),
        created_ts=created_ts.isoformat()
    ))
    run.finish()

print("\nBEST CONFIG (by validation):", best["cfg"], "| best_val_acc =", f"{best['val_acc']:.2f}")

# ---------------- SWEEP TIME-SERIES CHARTS ----------------
sweep_timeseries_df = pd.DataFrame(sweep_rows_epoch)

def _plot_timeseries(df, y_col, title, yaxis_title):
    fig = px.line(
        df.sort_values(["label","epoch"]),
        x="epoch", y=y_col, color="label",
        labels={"epoch":"Step", y_col:yaxis_title, "label":""}, title=title
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=60, r=20, t=60, b=120)
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridwidth=0.5)
    return fig

train_loss_fig = _plot_timeseries(sweep_timeseries_df, "train_loss", "train_loss", "Loss")
val_loss_fig   = _plot_timeseries(sweep_timeseries_df, "val_loss",   "val_loss",   "Loss")
train_acc_fig  = _plot_timeseries(sweep_timeseries_df, "train_acc",  "train_acc",  "Accuracy (%)")
val_acc_fig    = _plot_timeseries(sweep_timeseries_df, "val_acc",    "val_acc",    "Accuracy (%)")

charts_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=RUN_GROUP,
                        job_type="sweep_charts", config={"charts_from": "sweep_timeseries_table"},
                        mode="online", reinit=True)
charts_run.log({
    "sweep_timeseries_table": wandb.Table(dataframe=sweep_timeseries_df),
    "train_loss_chart": train_loss_fig, "val_loss_chart": val_loss_fig,
    "train_acc_chart":  train_acc_fig,  "val_acc_chart":  val_acc_fig,
})
charts_run.finish()

# ---------------- SCATTER: VAL ACC vs CREATED ----------------
scatter_df = pd.DataFrame(rows)
scatter_df["created_ts"] = pd.to_datetime(scatter_df["created_ts"])
scatter_df = scatter_df.sort_values("created_ts")
scatter_df["cum_best"] = scatter_df["best_val_acc"].cummax()
scatter_df["label"] = (scatter_df["activation"] + "|" + scatter_df["optimizer"] +
                       "|bs" + scatter_df["batch_size"].astype(str) + "|lr" + scatter_df["lr"].map("{:g}".format))

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=scatter_df["created_ts"], y=scatter_df["best_val_acc"],
    mode="markers", marker=dict(size=9), text=scatter_df["label"],
    name="runs", hovertemplate="%{text}<br>%{x|%b %d %H:%M:%S} — %{y:.2f}%<extra></extra>"
))
fig_scatter.add_trace(go.Scatter(
    x=scatter_df["created_ts"], y=scatter_df["cum_best"],
    mode="lines", name="cumulative best"
))
fig_scatter.update_layout(
    title="val_acc v. created",
    xaxis_title="Created", yaxis_title="Validation Accuracy (%)",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    margin=dict(l=60, r=20, t=60, b=80)
)
scatter_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=RUN_GROUP,
                         job_type="valacc_scatter", mode="online", reinit=True)
scatter_run.log({
    "val_acc_scatter_table": wandb.Table(dataframe=scatter_df),
    "val_acc_vs_created": fig_scatter
})
scatter_run.finish()

# ---------------- CURVES RUN (Best Config) ----------------
cfg = best["cfg"]; print("\n[Curves Run] Using best config:", cfg)
train_loader, val_loader, test_loader, train_set, val_set = get_loaders(
    batch_size=cfg.batch_size, augment=cfg.augment, val_ratio=0.1, num_workers=2
)
model = VGG6(num_classes=10, activation=cfg.activation, batch_norm=cfg.batch_norm).to(device)
model.init_weights(mode=cfg.init, nonlinearity=("relu" if cfg.activation=="relu" else "linear"))

criterion = nn.CrossEntropyLoss()
optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
scheduler = build_scheduler(optimizer, cfg.optimizer, epochs=30)
use_amp = torch.cuda.is_available()
try:
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
curve_epochs = 30

curve_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=RUN_GROUP,
                       job_type="curves_run", config=dict(best_config=True, **cfg.__dict__),
                       mode="online", reinit=True)
for epoch in range(1, curve_epochs + 1):
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler if use_amp else None)
    tr_acc  = accuracy(model, train_loader, device)
    va_loss, va_acc = eval_loss_acc(model, val_loader, criterion, device)
    hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
    hist["val_loss"].append(va_loss);   hist["val_acc"].append(va_acc)
    if scheduler: scheduler.step()
    if epoch % 5 == 0 or epoch == curve_epochs:
        print(f"[curves] epoch {epoch:02d} | train_acc {tr_acc:.2f} | val_acc {va_acc:.2f}")
    wandb.log({"epoch": epoch, "curves/train_loss": tr_loss, "curves/train_acc": tr_acc,
               "curves/val_loss": va_loss, "curves/val_acc": va_acc})
curve_run.finish()

# quick matplotlib preview (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5)); plt.plot(hist["train_loss"], label="Train Loss"); plt.plot(hist["val_loss"], label="Val Loss"); plt.legend(); plt.show()
plt.figure(figsize=(8,5)); plt.plot(hist["train_acc"], label="Train Acc (%)"); plt.plot(hist["val_acc"], label="Val Acc (%)"); plt.legend(); plt.show()

# ---------------- FINAL RETRAIN + TEST ----------------
print("\n[Retrain] TRAIN+VAL → TEST:", cfg)
full_train = ConcatDataset([train_set, val_set])
full_train_loader = DataLoader(full_train, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)

final_model = VGG6(num_classes=10, activation=cfg.activation, batch_norm=cfg.batch_norm).to(device)
final_model.init_weights(mode=cfg.init, nonlinearity=("relu" if cfg.activation=="relu" else "linear"))
final_criterion = nn.CrossEntropyLoss()
final_optimizer = build_optimizer(cfg.optimizer, final_model.parameters(), cfg.lr)
final_epochs = 50
final_scheduler = build_scheduler(final_optimizer, cfg.optimizer, final_epochs)
try:
    final_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
except Exception:
    final_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

final_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=RUN_GROUP,
                       job_type="final_train", config=cfg.__dict__, mode="online", reinit=True)
for epoch in range(1, final_epochs + 1):
    loss = train_one_epoch(final_model, full_train_loader, final_criterion, final_optimizer, device, final_scaler)
    if final_scheduler: final_scheduler.step()
    if epoch % 10 == 0 or epoch == final_epochs:
        tr_acc = accuracy(final_model, full_train_loader, device)
        print(f"[final] epoch {epoch:02d} | loss {loss:.4f} | train_acc {tr_acc:.2f}")
    wandb.log({"final/epoch": epoch, "final/train_loss": loss})

test_acc = accuracy(final_model, test_loader, device)
print("\nFINAL TEST top-1 accuracy:", f"{test_acc:.2f}%")
print("Best config used:", cfg)
final_run.summary["final_test_acc"] = float(test_acc)
final_run.finish()

# ---------------- SAVE BEST CHECKPOINTS ----------------
def save_ckpt(path, model, optimizer, epoch, metrics: dict, cfg_obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": {k: float(v) for k,v in metrics.items()},
        "config": {
            "activation": cfg_obj.activation,
            "optimizer": cfg_obj.optimizer,
            "lr": float(cfg_obj.lr),
            "batch_size": int(cfg_obj.batch_size),
            "epochs": int(cfg_obj.epochs),
            "init": cfg_obj.init,
            "augment": bool(cfg_obj.augment),
            "batch_norm": bool(cfg_obj.batch_norm),
        },
        "saved_with": "sweep_full.py",
        "framework": "pytorch"
    }, path)

os.makedirs("checkpoints", exist_ok=True)
_ckpt_metrics = {"best_val_acc": float(best["val_acc"]), "final_test_acc": float(test_acc)}
_ckpt_tag = f"{cfg.activation}-{cfg.optimizer}-lr{cfg.lr}-bs{cfg.batch_size}"
best_tagged_path = f"checkpoints/best-{_ckpt_tag}.pt"
save_ckpt(best_tagged_path, final_model, final_optimizer, final_epochs, _ckpt_metrics, cfg)
save_ckpt("checkpoints/best.pt", final_model, final_optimizer, final_epochs, _ckpt_metrics, cfg)

with open("checkpoints/MANIFEST.json", "w") as f:
    json.dump({
        "best_checkpoint": os.path.basename(best_tagged_path),
        "best_val_acc": round(float(best["val_acc"]), 4),
        "final_test_acc": round(float(test_acc), 4),
        "config": {
            "activation": cfg.activation, "optimizer": cfg.optimizer,
            "lr": cfg.lr, "batch_size": cfg.batch_size, "epochs": cfg.epochs,
            "init": cfg.init, "augment": cfg.augment, "batch_norm": cfg.batch_norm
        }
    }, f, indent=2)

print("\n✅ Saved checkpoints:")
os.system("ls -lh checkpoints")

# ---------------- PARALLEL COORDS ----------------
runs_df = pd.DataFrame(rows)
act_order = ["relu", "gelu", "silu", "tanh", "sigmoid"]
opt_order = ["adam", "nadam", "sgd", "nesterov-sgd", "rmsprop", "adagrad"]
runs_df["activation_cat"] = pd.Categorical(runs_df["activation"], categories=act_order, ordered=True)
runs_df["optimizer_cat"]  = pd.Categorical(runs_df["optimizer"], categories=opt_order, ordered=True)

fig = px.parallel_coordinates(
    runs_df,
    dimensions=["best_val_acc","lr","batch_size","activation_cat","optimizer_cat","epochs"],
    color="best_val_acc",
    labels={"best_val_acc":"Best Val Acc","lr":"LR","batch_size":"Batch",
            "activation_cat":"Activation","optimizer_cat":"Optimizer","epochs":"Epochs"}
)
fig.update_layout(title="Parallel Coordinates: 25-Run Sweep (VGG6)", title_x=0.5)
pc_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=RUN_GROUP,
                    job_type="parallel_plot", mode="online", reinit=True)
pc_run.log({"parallel_coordinates": fig, "sweep_table": wandb.Table(dataframe=runs_df)})
pc_run.finish()

print("\nDone. Logged: multi-run charts (4), val_acc vs created (scatter), curves run, final train, checkpoints, and parallel coordinates.")
