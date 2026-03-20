"""
Variational Autoencoder for time series anomaly detection.

Trains on normal windows; reconstruction error on test windows = anomaly score.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional

from .data import sliding_windows, normalize_windows


class TimeSeriesVAE(nn.Module):
    """
    VAE for 1D time series windows.

    Encoder: FC -> latent mean & logvar
    Decoder: FC -> reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """VAE loss = reconstruction (MSE) + beta * KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_vae(
    windows: np.ndarray,
    latent_dim: int = 8,
    hidden_dims: tuple[int, ...] = (64, 32),
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
    verbose: bool = True,
) -> tuple[TimeSeriesVAE, list[float]]:
    """Train VAE on normalised windows."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Normalise
    X = normalize_windows(windows, method="zscore")
    X = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TimeSeriesVAE(
        input_dim=windows.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    model.train()
    for ep in range(epochs):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if verbose and (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={avg:.6f}")

    return model, losses


def vae_anomaly_scores(
    model: TimeSeriesVAE,
    values: np.ndarray,
    window_size: int = 64,
    stride: int = 1,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute per-timestep anomaly scores from reconstruction error.

    Uses MSE per window, mapped back to timeline (same logic as isolation forest).
    Higher = more anomalous.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    windows = sliding_windows(values, window_size, stride)
    X = normalize_windows(windows, method="zscore")
    X = torch.tensor(X, dtype=torch.float32)

    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = X[i : i + 256].to(device)
            recon, _, _ = model(batch)
            err = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
            recon_errors.append(err)
    recon_errors = np.concatenate(recon_errors)

    n = len(values)
    scores = np.full(n, np.nan)
    counts = np.zeros(n)

    for i in range(len(windows)):
        start = i * stride
        end = start + window_size
        for j in range(start, min(end, n)):
            if np.isnan(scores[j]):
                scores[j] = recon_errors[i]
                counts[j] = 1
            else:
                scores[j] += recon_errors[i]
                counts[j] += 1

    counts = np.where(counts == 0, 1, counts)
    scores = scores / counts
    scores = np.nan_to_num(scores, nan=0.0)
    return scores.astype(np.float32)
