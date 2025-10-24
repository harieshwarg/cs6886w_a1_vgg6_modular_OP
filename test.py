import argparse, torch, torch.nn as nn
from model import VGG6
from data_loader import get_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--activation", default="gelu")
    ap.add_argument("--batch-norm", action="store_true", default=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, *_ = get_loaders(args.batch_size, augment=False, val_ratio=0.1, num_workers=2, data_root=args.data_root)

    model = VGG6(activation=args.activation, batch_norm=args.batch_norm).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            test_loss += loss.item() * imgs.size(0)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)

    test_acc = 100.0 * correct / total
    test_loss = test_loss / total
    print(f"TEST — loss {test_loss:.4f} | acc {test_acc:.2f}% | from ckpt {args.ckpt}")
    if isinstance(ckpt, dict) and 'metrics' in ckpt and 'val_acc' in ckpt['metrics']:
        print(f"(saved best val_acc: {ckpt['metrics']['val_acc']:.2f}%)")

if __name__ == "__main__":
    main()
