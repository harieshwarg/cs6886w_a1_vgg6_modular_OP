import os, argparse, torch, torch.nn as nn
from model import VGG6
from data_loader import get_loaders
from utils import set_seed, train_one_epoch, eval_loss_acc, accuracy, build_optimizer

def save_ckpt(path, model, optimizer, epoch, metrics: dict, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "config": cfg,
        "framework": "pytorch"
    }, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="gelu")
    ap.add_argument("--optimizer", default="rmsprop")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-norm", action="store_true", default=True)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, *_ = get_loaders(
        batch_size=args.batch_size, augment=args.augment, val_ratio=0.1, num_workers=2, data_root=args.data_root
    )

    model = VGG6(num_classes=10, activation=args.activation, batch_norm=args.batch_norm).to(device)
    model.init_weights(mode="kaiming", nonlinearity=("relu" if args.activation=="relu" else "linear"))
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    # AMP scaler (works for both torch>=2.1 and older fallback)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = eval_loss_acc(model, val_loader, criterion, device)
        print(f"epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_acc {va_acc:.2f}")

        cfg = vars(args).copy()
        save_ckpt(os.path.join(args.save_dir, "latest.pt"), model, optimizer, epoch, {"val_acc": va_acc}, cfg)
        if va_acc > best_val:
            best_val = va_acc
            tag = f"{args.activation}-{args.optimizer}-lr{args.lr}-bs{args.batch_size}"
            save_ckpt(os.path.join(args.save_dir, "best.pt"), model, optimizer, epoch, {"val_acc": best_val}, cfg)
            save_ckpt(os.path.join(args.save_dir, f"best-{tag}.pt"), model, optimizer, epoch, {"val_acc": best_val}, cfg)

    print("Done. Best val_acc:", best_val)

if __name__ == "__main__":
    main()
