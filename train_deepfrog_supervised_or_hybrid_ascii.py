# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, time, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------------
# FROGNet (SHG)  differentiable forward model (E -> I)
# ----------------------------------------------------------------------------------
class FROGNetSHG(nn.Module):
    """
    Differentiable forward model for SHG-FROG trace generation: E(t) -> I(ω,τ)
    
    Implements the FROG trace equation:
        I(ω,τ) = |∫ E(t)E(t-τ) exp(-iωt) dt|²
    
    Input: x_ri with shape (B, 2N) = [Re(E), Im(E)] concatenated
    Output: I with shape (B, N, N) = FROG trace
    """
    def __init__(
        self,
        N: int = 64,
        normalize_input: bool = True,
        normalize_trace: bool = True,
    ) -> None:
        super().__init__()
        self.N = int(N)
        self.normalize_input = bool(normalize_input)
        self.normalize_trace = bool(normalize_trace)

    @staticmethod
    def _to_complex_from_ri(x_ri: torch.Tensor) -> torch.Tensor:
        assert x_ri.shape[-1] % 2 == 0, "Last dimension must be 2N"
        N = x_ri.shape[-1] // 2
        real = x_ri[..., :N]
        imag = x_ri[..., N:]
        return torch.complex(real, imag)

    def forward(self, x_ri: torch.Tensor) -> torch.Tensor:
        if x_ri.ndim == 1:
            x_ri = x_ri.unsqueeze(0)
        twoN = x_ri.shape[-1]
        assert twoN % 2 == 0, f"Expected last dim 2N, got {twoN}"
        N = twoN // 2
        # Convert from real/imaginary representation to complex
        # The result is automatically complex64 if input is float32
        E = self._to_complex_from_ri(x_ri)  # (B, N)

        if self.normalize_input:
            max_abs = E.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20)
            E = E / max_abs

        # Build circularly shifted replicas E(t-τ_j); (B, N_delays, N_time)
        # This mimics the delay: E(t-τ) for each delay τ_j
        E_shifts = torch.stack([torch.roll(E, shifts=j, dims=-1) for j in range(N)], dim=1)
        
        # Gate signal: E(t) * E(t-τ) for each delay
        gate = E.unsqueeze(1) * E_shifts  # (B, N, N)

        # FFT of gated signal: FT{E(t)E(t-τ)}
        spec = torch.fft.fft(gate, dim=-1)
        
        # Shift both axes to center zero frequency (to match experimental FROG traces)
        # fftshift on frequency axis (last dim) - moves zero frequency to center
        spec = torch.fft.fftshift(spec, dim=-1)
        # fftshift on delay axis (dim=-2) - centers the delay axis
        spec = torch.fft.fftshift(spec, dim=-2)
        
        # FROG trace intensity: I(ω,τ) = |FT{E(t)E(t-τ)}|²
        I = (spec.real**2 + spec.imag**2).real
        
        # Flip frequency axis to match training data convention
        # (Different FFT implementations may have opposite frequency ordering)
        I = torch.flip(I, dims=(-1,))

        if self.normalize_trace:
            I = I / I.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-20)
        
        # Return as float (FROG traces are real-valued intensities)
        return I.to(dtype=x_ri.dtype)

# ----------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_Evec_maxabs(Evec: torch.Tensor) -> torch.Tensor:
    """Normalize (B, 2N) RI vector by per-sample max |E| across time N."""
    N2 = Evec.shape[-1] // 2
    Er = Evec[..., :N2]
    Ei = Evec[..., N2:]
    mag = torch.sqrt(Er**2 + Ei**2)  # (B, N)
    maxabs = mag.amax(dim=-1, keepdim=True).clamp_min(1e-20)
    Ern = Er / maxabs
    Ein = Ei / maxabs
    return torch.cat([Ern, Ein], dim=-1)

# ----------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------
class PairedFROGDataset(Dataset):
    """Supervised paired dataset: (I, Evec), index-aligned."""
    def __init__(self, I: np.ndarray, Evec: np.ndarray, renorm_traces: bool = True):
        assert I.ndim == 3 and I.shape[1:] == (64, 64), f"I must be (N,64,64), got {I.shape}"
        assert Evec.ndim == 2 and Evec.shape[1] == 128, f"Evec must be (N,128), got {Evec.shape}"
        assert I.shape[0] == Evec.shape[0], "I and Evec must have the same length (paired by index)."
        I = I.astype(np.float32)
        if renorm_traces:
            # Ensure each trace has unit max (safe even if already normalized)
            m = I.reshape(I.shape[0], -1).max(axis=1, keepdims=True)
            m = np.clip(m, 1e-20, None)
            I = I / m.reshape(-1, 1, 1)
        self.I = I
        self.Evec = Evec.astype(np.float32)

    def __len__(self): return self.I.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.I[idx]), torch.from_numpy(self.Evec[idx])

class UnsupervisedFROGDataset(Dataset):
    """Unsupervised-only traces for consistency training."""
    def __init__(self, I: np.ndarray, renorm_traces: bool = True):
        assert I.ndim == 3 and I.shape[1:] == (64, 64), f"I must be (N,64,64), got {I.shape}"
        I = I.astype(np.float32)
        if renorm_traces:
            m = I.reshape(I.shape[0], -1).max(axis=1, keepdims=True)
            m = np.clip(m, 1e-20, None)
            I = I / m.reshape(-1, 1, 1)
        self.I = I
    def __len__(self): return self.I.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.I[idx])

# ----------------------------------------------------------------------------------
# Model: 64x64 trace -> 128 (Re[0:64], Im[64:128])
# ----------------------------------------------------------------------------------
def multires_block(c_in: int, c_out: int) -> nn.ModuleList:
    ks = [11, 7, 5, 3]
    per = c_out // len(ks)
    return nn.ModuleList([nn.Conv2d(c_in, per, k, padding=k//2) for k in ks])

class MultiresNet(nn.Module):
    def __init__(self, N: int = 64):
        super().__init__()
        c = 32
        self.blocks = nn.ModuleList([
            # Stage 1: 64x64 -> 32x32, 32ch -> 64ch
            multires_block(1,   c),     nn.Conv2d( c,   2*c, 3, stride=2, padding=1),
            # Stage 2: 32x32 -> 16x16, 64ch -> 128ch
            multires_block(2*c, 2*c),   nn.Conv2d(2*c,  4*c, 3, stride=2, padding=1),
            # Stage 3: 16x16 -> 8x8, 128ch -> 256ch
            multires_block(4*c, 4*c),   nn.Conv2d(4*c,  8*c, 3, stride=2, padding=1),
        ])
        feat_dim = (8*c) * (N//8) * (N//8)  # 64->8
        self.fc1 = nn.Linear(feat_dim, 512)
        self.fc2 = nn.Linear(512, 2*N)

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        x = I.unsqueeze(1)  # (B,1,64,64)
        for layer in self.blocks:
            if isinstance(layer, nn.ModuleList):
                x = torch.cat([F.relu(m(x)) for m in layer], dim=1)
            else:
                x = F.relu(layer(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # (B, 128)

# ----------------------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------------------
def run_epoch_paired(model, frog, loader, device, lam, opt=None, train=True):
    if train: model.train()
    else: model.eval()
    Ls, Lu, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for I, Evec in loader:
            I = I.to(device)
            Evec = Evec.to(device)
            if train and opt is not None: opt.zero_grad(set_to_none=True)

            E_pred = model(I)                  # (B, 128)
            # Supervised loss on normalized profiles
            E_pred_n = normalize_Evec_maxabs(E_pred)
            E_true_n = normalize_Evec_maxabs(Evec)
            loss_sup = F.l1_loss(E_pred_n, E_true_n)

            # Optional unsupervised consistency on the same batch
            loss_uns = torch.tensor(0.0, device=device)
            if lam > 0.0:
                I_pred = frog(E_pred_n)        # normalized inside
                loss_uns = F.l1_loss(I_pred, I)

            loss = loss_sup + lam * loss_uns
            if train and opt is not None:
                loss.backward()
                opt.step()

            bs = I.size(0)
            Ls += float(loss_sup.item()) * bs
            Lu += float(loss_uns.item()) * bs
            n  += bs
    n = max(n, 1)
    return Ls/n, Lu/n

@torch.no_grad()
def run_epoch_uns_only(model, frog, loader, device, opt=None, train=True):
    if train: model.train()
    else: model.eval()
    Lu, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for I in loader:
            I = I.to(device)
            if train and opt is not None: opt.zero_grad(set_to_none=True)
            E_pred = model(I)
            E_pred_n = normalize_Evec_maxabs(E_pred)
            I_pred = frog(E_pred_n)
            loss_uns = F.l1_loss(I_pred, I)
            if train and opt is not None:
                loss_uns.backward()
                opt.step()
            bs = I.size(0)
            Lu += float(loss_uns.item()) * bs
            n  += bs
    n = max(n, 1)
    return Lu/n

def ramp_lambda(epoch: int, warmup: int, lam_max: float) -> float:
    if warmup <= 0: return lam_max
    t = min(1.0, epoch / float(warmup))
    return lam_max * t

def train(model, frog, loaders, device, epochs, lr, lam_max, warmup, log_dir: Path, ckpt_dir: Path):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device); frog.to(device)

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","lambda","train_Lsup","train_Luns","extra_unsup_Luns","val_Lsup","val_Luns","time_s"])

    for e in range(1, epochs+1):
        t0 = time.time()
        lam = ramp_lambda(e, warmup, lam_max)

        tr_sup, tr_uns = run_epoch_paired(model, frog, loaders["train"], device, lam, opt, train=True)
        if "unsup" in loaders:
            extra_uns = run_epoch_uns_only(model, frog, loaders["unsup"], device, opt, train=True)
        else:
            extra_uns = float("nan")
        val_sup, val_uns = run_epoch_paired(model, frog, loaders["val"], device, lam, opt=None, train=False)

        dt = time.time() - t0
        print(f"Epoch {e:03d} | lambda={lam:.3f} | Train L_sup={tr_sup:.4e} L_uns={tr_uns:.4e} | "
              f"ExtraUns L_uns={extra_uns:.4e} | Val L_sup={val_sup:.4e} L_uns={val_uns:.4e} | {dt:.1f}s")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([e, lam, tr_sup, tr_uns, extra_uns, val_sup, val_uns, dt])

        if e % 10 == 0 or e == epochs:
            torch.save({"epoch": e, "model": model.state_dict()}, ckpt_dir / f"epoch_{e:03d}.pt")

# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Train DeepFROG: supervised or hybrid (paired I,E; optional unsupervised consistency).")
    ap.add_argument("--paired_profile", type=str, required=True,
                    help="Path to E_profile_*.npy with shape (N,128): [Re(0:64), Im(64:128)]")
    ap.add_argument("--paired_traces", type=str, required=True,
                    help="Path to FROG_T_*.npy with shape (N,64,64); MUST be index-aligned with --paired_profile")
    ap.add_argument("--unsup_traces", type=str, default="",
                    help="Optional extra unsupervised-only traces (M,64,64)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_unsup", type=float, default=0.0, help="Max weight for unsupervised consistency; 0 = pure supervised")
    ap.add_argument("--warmup", type=int, default=5, help="Epochs to ramp lambda from 0 -> max")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load paired supervised data
    Evec = np.load(args.paired_profile)
    Ipair = np.load(args.paired_traces)
    if Evec.ndim != 2 or Evec.shape[1] != 128:
        raise ValueError(f"paired_profile must be (N,128). Got {Evec.shape}")
    if Ipair.ndim != 3 or Ipair.shape[1:] != (64,64):
        raise ValueError(f"paired_traces must be (N,64,64). Got {Ipair.shape}")
    if Evec.shape[0] != Ipair.shape[0]:
        raise ValueError(f"Mismatched lengths: profiles N={Evec.shape[0]} vs traces N={Ipair.shape[0]}")
    N = Evec.shape[0]

    # Build frozen forward model (N_time = 64)
    # This model converts pulse E(t) to FROG trace I(ω,τ) using the SHG-FROG equation
    frog = FROGNetSHG(N=64, normalize_input=True, normalize_trace=True)

    # Train/val split
    n_val = max(1, int(N * args.val_frac))
    perm = np.random.permutation(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    train_ds = PairedFROGDataset(Ipair[tr_idx], Evec[tr_idx], renorm_traces=True)
    val_ds   = PairedFROGDataset(Ipair[val_idx], Evec[val_idx], renorm_traces=True)

    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False),
    }

    # Optional unsupervised-only traces
    if args.unsup_traces and Path(args.unsup_traces).exists():
        I_uns = np.load(args.unsup_traces).astype(np.float32)
        if I_uns.ndim != 3 or I_uns.shape[1:] != (64,64):
            raise ValueError(f"unsup_traces must be (M,64,64). Got {I_uns.shape}")
        loaders["unsup"] = DataLoader(UnsupervisedFROGDataset(I_uns, renorm_traces=True),
                                      batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model
    model = MultiresNet(N=64)

    # Train
    log_dir  = Path("logs")
    ckpt_dir = Path("checkpoints")
    print(f"Device: {device}. Train/val sizes: {len(train_ds)}/{len(val_ds)}. "
          f"Extra unsup: {len(loaders['unsup'].dataset) if 'unsup' in loaders else 0}")
    train(model, frog, loaders, device,
          epochs=args.epochs, lr=args.lr, lam_max=args.lambda_unsup, warmup=args.warmup,
          log_dir=log_dir, ckpt_dir=ckpt_dir)

if __name__ == "__main__":
    main()
