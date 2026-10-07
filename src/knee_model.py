import os
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["sag", "cor", "axi"]
DEPTH, SIZE, CROP = 32, 256, 224
CENTERS = list(range(1, DEPTH - 1, 2))
SCORE_MAX = 4.5
SILVER_POS = 3.0


def submission_frame(ids, probs):
    df = pd.DataFrame(np.clip(np.asarray(probs, dtype=float), 0.0, 1.0), columns=LABELS)
    df.insert(0, "StudyInstanceUID", list(ids))
    return df


def check_submission(sub, sample, test_ids):
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append("columns differ from sample_submission")
    if set(sub["StudyInstanceUID"]) != set(test_ids):
        problems.append("study ids differ from test.csv")
    if sub["StudyInstanceUID"].duplicated().any():
        problems.append("duplicate study ids")
    if sub[LABELS].isna().any().any():
        problems.append("missing predictions")
    return problems


def to_targets(scores):
    return (scores[LABELS].clip(0, SCORE_MAX) / SCORE_MAX).astype("float32").values


def prepare(vols, train=False, rng=None):
    x = np.zeros((3, DEPTH, CROP, CROP), np.uint8)
    mask = np.zeros(3, np.float32)
    for p, key in enumerate(PLANES):
        v = vols.get(key) if vols else None
        if v is None or v.shape != (DEPTH, SIZE, SIZE):
            continue
        if train:
            oy, ox = rng.integers(0, SIZE - CROP + 1, 2)
        else:
            oy = ox = (SIZE - CROP) // 2
        x[p] = v[:, oy:oy + CROP, ox:ox + CROP]
        mask[p] = 1.0
    return x, mask


class KneeDataset(Dataset):
    def __init__(self, ids, cache_dir, targets=None, train=False):
        self.ids, self.cache_dir = list(ids), cache_dir
        self.targets, self.train = targets, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vols = {}
        try:
            with np.load(os.path.join(self.cache_dir, f"{self.ids[i]}.npz")) as z:
                vols = {k: z[k] for k in z.files}
        except Exception:
            pass
        x, mask = prepare(vols, self.train, np.random.default_rng() if self.train else None)
        out = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask)}
        if self.targets is not None:
            out["y"] = torch.from_numpy(self.targets[i])
        return out


class KneeNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, n_out=len(LABELS), drop=0.3):
        super().__init__()
        import timm
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        f = self.enc.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(len(PLANES) * 2 * f, n_out))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("centers", torch.tensor(CENTERS))

    def forward(self, x, mask):
        b, p = x.shape[0], x.shape[1]
        x = x.float() / 255.0
        c = self.centers
        trip = torch.stack([x[:, :, c - 1], x[:, :, c], x[:, :, c + 1]], dim=3)
        n = trip.shape[2]
        imgs = trip.reshape(b * p * n, 3, x.shape[-2], x.shape[-1])
        feats = self.enc((imgs - self.mean) / self.std).float()
        feats = feats.view(b, p, n, -1)
        pooled = torch.cat([feats.mean(dim=2), feats.amax(dim=2)], dim=-1) * mask[:, :, None]
        return self.head(pooled.reshape(b, -1))


def silver_aucs(scores, preds):
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, k in enumerate(LABELS):
        y = (scores[k].values >= SILVER_POS).astype(int)
        if 0 < y.sum() < len(y):
            out[k] = roc_auc_score(y, preds[:, j])
    return out


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    out = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(b["x"].cuda(non_blocking=True), b["mask"].cuda(non_blocking=True))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(cfg):
    fold, out_dir = cfg["fold"], cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg["seed"] + fold)
    np.random.seed(cfg["seed"] + fold)

    df = pd.read_csv(cfg["folds_csv"]).merge(pd.read_csv(cfg["labels_csv"]), on="StudyInstanceUID")
    tr, va = df[df["fold"] != fold].reset_index(drop=True), df[df["fold"] == fold].reset_index(drop=True)
    kw = dict(num_workers=cfg["workers"], pin_memory=True, persistent_workers=True)
    dl_tr = DataLoader(KneeDataset(tr["StudyInstanceUID"], cfg["cache_dir"], to_targets(tr), train=True),
                       batch_size=cfg["bs"], shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(KneeDataset(va["StudyInstanceUID"], cfg["cache_dir"]),
                       batch_size=cfg["bs"], shuffle=False, **kw)

    model = KneeNet(cfg["backbone"], pretrained=True, drop=cfg["drop"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    accum = cfg.get("accum", 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                                total_steps=cfg["epochs"] * (len(dl_tr) // accum), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    run = None
    key = os.environ.get("WANDB_API_KEY")
    if key:
        try:
            import wandb
            wandb.login(key=key, relogin=True)
            run = wandb.init(project=cfg["wandb_project"], group=cfg["exp"], name=f"{cfg['exp']}-f{fold}",
                             job_type="train", config=cfg)
            print("wandb run:", run.url, flush=True)
        except Exception as err:
            print("wandb off:", repr(err)[:300], flush=True)
    else:
        print("wandb off: no WANDB_API_KEY in this process", flush=True)

    print(f"fold {fold}: train {len(tr)}, val {len(va)}, steps/epoch {len(dl_tr)}", flush=True)
    best, history = -1.0, []
    for ep in range(cfg["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, b in enumerate(dl_tr):
            x, m, y = (b["x"].cuda(non_blocking=True), b["mask"].cuda(non_blocking=True),
                       b["y"].cuda(non_blocking=True))
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
            if ep == 0 and step == 39:
                sec = (time.time() - t0) / 40
                print(f"  first 40 batches: {sec:.2f} s/batch, about {sec * len(dl_tr) / 60:.0f} min/epoch, "
                      f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        preds = predict_loader(model, dl_va)
        val_loss = float(F.binary_cross_entropy(torch.from_numpy(preds).clamp(1e-6, 1 - 1e-6),
                                                torch.from_numpy(to_targets(va))))
        aucs = silver_aucs(va, preds)
        macro = float(np.mean(list(aucs.values())))
        rec = {"epoch": ep, "train_loss": tot / n, "val_loss": val_loss, "val_silver_macro": macro,
               "minutes": (time.time() - t0) / 60}
        history.append(rec)
        print(f"epoch {ep}: train {rec['train_loss']:.4f} val {val_loss:.4f} silver macro {macro:.4f} "
              f"({rec['minutes']:.1f} min)", flush=True)
        if run:
            run.log({**rec, **{f"val_silver/{k}": v for k, v in aucs.items()}})
        if macro > best:
            best = macro
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": ep, "val_silver_macro": macro},
                       os.path.join(out_dir, f"best_f{fold}.pt"))
            oof = pd.DataFrame(preds, columns=LABELS)
            oof.insert(0, "StudyInstanceUID", va["StudyInstanceUID"].values)
            oof.to_csv(os.path.join(out_dir, f"oof_f{fold}.csv"), index=False)
    json.dump(history, open(os.path.join(out_dir, f"history_f{fold}.json"), "w"), indent=1)
    if run:
        run.summary["best_val_silver_macro"] = best
        run.finish()
    print(f"fold {fold} done, best silver macro {best:.4f}", flush=True)


LAT_FRACS = (0.5, 0.65, 0.8)


def lat_input(vols):
    v = vols.get("cor") if vols else None
    if v is None or v.shape != (DEPTH, SIZE, SIZE):
        return None
    return v[[int(round(f * (DEPTH - 1))) for f in LAT_FRACS]]


def make_lat_net(pretrained=False):
    import timm
    return timm.create_model("resnet18", pretrained=pretrained, num_classes=1, in_chans=3)


def lat_norm(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def lat_predict(model, arr, device="cuda"):
    x = torch.from_numpy(np.ascontiguousarray(arr)).to(device).float() / 255.0
    o = (SIZE - CROP) // 2
    x = x[:, :, o:o + CROP, o:o + CROP]
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        p1 = torch.sigmoid(model(lat_norm(x)).float().squeeze(1))
        p2 = torch.sigmoid(model(lat_norm(x.flip(-1))).float().squeeze(1))
    return ((p1 + 1 - p2) / 2).cpu().numpy()


def to_canonical(vols, side):
    if side != "L":
        return vols
    out = {}
    for k, v in vols.items():
        out[k] = v[::-1].copy() if k == "sag" else v[:, :, ::-1].copy()
    return out


def load_models(paths, device="cuda"):
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        m = KneeNet(ck["cfg"]["backbone"], pretrained=False, drop=ck["cfg"]["drop"])
        m.load_state_dict(ck["state_dict"])
        models.append(m.to(device).eval())
    return models


@torch.no_grad()
def predict_study(models, vols, device="cuda"):
    x, mask = prepare(vols, train=False)
    x = torch.from_numpy(x)[None].to(device)
    mask = torch.from_numpy(mask)[None].to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        p = torch.stack([torch.sigmoid(m(x, mask).float()) for m in models]).mean(0)
    return p[0].cpu().numpy()
