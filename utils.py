import random, numpy as np, torch, torch.nn as nn, torch.optim as optim

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)

def eval_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        running += loss.item()
    return running / max(1, len(loader))

def build_optimizer(name, params, lr):
    key = str(name).strip().lower().replace(' ', '').replace('-', '').replace('_', '')
    if key == 'sgd':               return optim.SGD(params, lr=lr, momentum=0.9)
    if key in {'nesterovsgd','nesterov'}: return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if key == 'adam':              return optim.Adam(params, lr=lr)
    if key == 'nadam':             return optim.NAdam(params, lr=lr)
    if key == 'adagrad':           return optim.Adagrad(params, lr=lr)
    if key == 'rmsprop':           return optim.RMSprop(params, lr=lr, momentum=0.9)
    raise ValueError(f'Unknown optimizer: {name}')

def build_scheduler(optimizer, opt_name, epochs):
    key = str(opt_name).strip().lower().replace(' ', '').replace('-', '').replace('_', '')
    if key in {'sgd','nesterovsgd','nesterov'} and epochs >= 20:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(0.6*epochs), int(0.85*epochs)], gamma=0.1
        )
    return None
