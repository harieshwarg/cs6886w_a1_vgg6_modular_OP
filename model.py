import torch, torch.nn as nn, torch.nn.functional as F

_ACTS = {
    "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
    "silu": nn.SiLU, "gelu": nn.GELU,
}

class VGG6(nn.Module):
    def __init__(self, num_classes=10, activation="relu", batch_norm=True):
        super().__init__()
        Act = _ACTS[activation]
        layers = []; in_ch = 3
        cfg = [64, 64, "M", 128, 128, "M"]
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                conv = nn.Conv2d(in_ch, v, 3, padding=1, bias=not batch_norm)
                if batch_norm:
                    layers += [conv, nn.BatchNorm2d(v), Act(inplace=True) if activation=="relu" else Act()]
                else:
                    layers += [conv, Act(inplace=True) if activation=="relu" else Act()]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def init_weights(self, mode="kaiming", nonlinearity="relu"):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                (nn.init.kaiming_normal_ if mode=="kaiming" else nn.init.xavier_normal_)(m.weight, **({"nonlinearity":nonlinearity} if mode=="kaiming" else {}))
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                (nn.init.kaiming_normal_ if mode=="kaiming" else nn.init.xavier_normal_)(m.weight, **({"nonlinearity":nonlinearity} if mode=="kaiming" else {}))
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
