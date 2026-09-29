"""
Autoencoder + linear classifier baseline (R2). The encoder has the same architecture as the model kept by NNS
(layers [0, split) of the network), it is trained without labels to reconstruct the input with a small
decoder, and a linear head identical to the NNS_LP one (feature normalization with temperature + linear
layer) is then trained on the frozen encoder. Compares an unsupervised representation with the
supervised features that NNS selects during training.

Example:
    python autoencoder_baseline.py -trainer configs/trainer/resnet18.yaml -tunnel configs/tunnel/linear_probing.yaml \
        -set dataset.name=cifar10 settings.seed=2025 --split 7
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from posthoc import base_parser, setup, loaders, save_json
from dataset import DATASET_NUM_CHANNELS, DATASET_IMG_SIZE
from flops import count_params


def make_decoder(feature_shape, channels, img_size):
    """Decoder from the encoder output (C, H, W) or (d,) to the input image."""
    if len(feature_shape) == 3:
        c, h, _ = feature_shape
        layers = []
        while h < img_size:
            out = max(c // 2, 32)
            layers += [nn.ConvTranspose2d(c, out, 4, stride=2, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            c, h = out, h * 2
        layers += [nn.Conv2d(c, channels, 3, padding=1)]
        return nn.Sequential(*layers)
    d = feature_shape[0]
    return nn.Sequential(nn.Linear(d, 1024), nn.ReLU(inplace=True), nn.Linear(1024, channels * img_size * img_size),
                         nn.Unflatten(1, (channels, img_size, img_size)))


class LinearHead(nn.Module):
    """Same head as NNS_LP: tau * normalize(features) followed by a linear layer."""
    def __init__(self, d, num_classes, tau, normalize, bias):
        super().__init__()
        self.tau, self.normalize = tau, normalize
        self.fc = nn.Linear(d, num_classes, bias=bias)

    def forward(self, h):
        h = h.reshape(h.size(0), -1)
        if self.normalize:
            h = self.tau * F.normalize(h, dim=1)
        return self.fc(h)


def main():
    parser = base_parser("Autoencoder + linear classifier baseline")
    parser.add_argument('--split', type=int, required=True, help='the encoder keeps the layers [0, split) (use the NNS split)')
    parser.add_argument('--ae_epochs', type=int, default=100)
    parser.add_argument('--probe_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    if args.tunnel is None:
        args.tunnel = 'configs/tunnel/linear_probing.yaml'

    config, dataset, net = setup(args)
    device = config["settings"]["device"]
    name = config["dataset"]["name"]
    train_loader, _, test_loader = loaders(config, dataset, eval_transform_train=False)

    encoder = nn.Sequential(*list(net.extractor.net)[:args.split]).to(device)
    X, _ = next(iter(train_loader))
    with torch.no_grad():
        encoder.eval()
        feature_shape = tuple(encoder(X[:2].to(device)).shape[1:])
    decoder = make_decoder(feature_shape, DATASET_NUM_CHANNELS[name], DATASET_IMG_SIZE[name]).to(device)
    print(f"encoder output {feature_shape}, encoder params {count_params(encoder):,}, decoder params {count_params(decoder):,}")

    # 1) unsupervised training of the encoder
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ae_epochs)
    t0 = time.time()
    for epoch in range(args.ae_epochs):
        encoder.train(); decoder.train()
        total = 0.0
        for X, _ in train_loader:
            X = X.to(device)
            loss = F.mse_loss(decoder(encoder(X)), X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"autoencoder epoch {epoch}: reconstruction loss {total / len(train_loader):.4f}")
    ae_time = time.time() - t0

    # 2) linear head on the frozen encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    ta = config["tunnel_args"]
    head = LinearHead(int(torch.tensor(feature_shape).prod()), dataset.num_classes, ta["temperature"], ta["normalize"], ta["bias"]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)
    t0 = time.time()
    for epoch in range(args.probe_epochs):
        head.train()
        total = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                h = encoder(X)
            loss = F.cross_entropy(head(h), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"probe epoch {epoch}: loss {total / len(train_loader):.4f}")
    probe_time = time.time() - t0

    head.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            correct += (head(encoder(X)).argmax(1) == y).sum().item()
            n += y.numel()
    result = {"method": "autoencoder_linear", "model": config["model"]["type"], "dataset": name, "seed": config["settings"]["seed"],
              "num_classes": dataset.num_classes, "split": args.split, "ae_epochs": args.ae_epochs, "probe_epochs": args.probe_epochs,
              "test_accuracy": correct / n, "params": count_params(encoder) + count_params(head),
              "time_autoencoder": ae_time, "time_probe": probe_time}
    print(f"test accuracy {result['test_accuracy']:.4f}")
    save_json(result, args.out or f"results/autoencoder/{name}_{config['model']['type']}_{config['settings']['seed']}_split{args.split}.json")


if __name__ == "__main__":
    main()
