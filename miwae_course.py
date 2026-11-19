"""Train one observed-data MIWAE per supplied course mask."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import time

from filelock import FileLock
import fastexcel
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

DATA = Path(__file__).resolve().parent / "Course Project Datasets"
OUTPUT = DATA.parent / "outputs" / "miwae"
CATEGORICAL = {"TTTTEG", "HOV", "MUSH", "Splice", "C4"}
MIXED = {
    "CREDIT": [0, 3, 4, 5, 6, 8, 9, 11, 12],
    "Abalone": [0],
    "Adult": [1, 3, 5, 6, 7, 8, 9, 13],
    "Aheart": [4],
    "KDD": [1, 2, 3],
}


@dataclass(frozen=True)
class Settings:
    seed: int = 1234
    hidden: int = 32
    latent: int = 4
    train_samples: int = 4
    impute_samples: int = 8
    batch: int = 256
    min_updates: int = 64
    min_epochs: int = 1
    learning_rate: float = 0.002


def datasets(group="all"):
    names = sorted(
        p.stem for p in (DATA / "Original Datasets Without Labels").glob("*.xlsx")
    )
    groups = {
        "all": names,
        "numerical": [n for n in names if n not in MIXED and n not in CATEGORICAL],
        "categorical": [n for n in names if n in CATEGORICAL or n in MIXED],
    }
    return groups[group]


def read_frame(path):
    batch = fastexcel.read_excel(path).load_sheet(
        0, header_row=None, skip_rows=0, schema_sample_rows=None, eager=True
    )
    frame = batch.to_pandas()
    frame.columns = [f"x{i+1}" for i in range(frame.shape[1])]
    return frame


class MIWAE(nn.Module):
    def __init__(self, numeric, levels, cfg):
        super().__init__()
        self.numeric = numeric
        self.levels = levels
        width = numeric + sum(levels)

        def network(inputs, outputs):
            return nn.Sequential(
                nn.Linear(inputs, cfg.hidden),
                nn.ReLU(),
                nn.Linear(cfg.hidden, cfg.hidden),
                nn.ReLU(),
                nn.Linear(cfg.hidden, outputs),
            )

        self.encoder = network(width + numeric + len(levels), 2 * cfg.latent)
        self.decoder = network(cfg.latent, 3 * numeric + sum(levels))

    def forward(self, x, mask, samples):
        n = self.numeric
        cats = [
            F.one_hot(x[:, n + i].long(), size).float() * mask[:, n + i, None]
            for i, size in enumerate(self.levels)
        ]
        loc, raw = self.encoder(torch.cat([x[:, :n], *cats, mask], dim=-1)).chunk(2, -1)
        scale = F.softplus(raw) + 1e-4
        noise = torch.randn(samples, *loc.shape, device=x.device)
        z = loc + scale * noise
        # Standard-normal prior minus diagonal-Gaussian proposal log density.
        logw = (scale.log() + 0.5 * (noise.square() - z.square())).sum(-1)
        out = self.decoder(z)
        means = out[..., :n]
        if n:
            dist = torch.distributions.StudentT(
                3 + F.softplus(out[..., 2 * n : 3 * n]),
                means,
                0.001 + F.softplus(out[..., n : 2 * n]),
                validate_args=False,
            )
            logw = logw + (dist.log_prob(x[:, :n]) * mask[:, :n]).sum(-1)
        log_probs = []
        offset = 3 * n
        for i, size in enumerate(self.levels):
            probs = out[..., offset : offset + size].log_softmax(-1)
            idx = x[:, n + i].long()[None, :, None].expand(samples, -1, 1)
            logw = logw + probs.gather(-1, idx).squeeze(-1) * mask[:, n + i]
            log_probs.append(probs)
            offset += size
        return logw, means, log_probs


def impute(frame, categorical, cfg, loss_trace=None):
    torch.manual_seed(cfg.seed)
    numeric = [j for j in range(frame.shape[1]) if j not in categorical]
    order = numeric + categorical
    mask = frame.iloc[:, order].notna().to_numpy()
    if not mask.any(axis=0).all():
        raise ValueError("A feature has no observations from which to learn")
    values = frame.iloc[:, numeric].to_numpy(dtype=float)
    center, scale = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    scale[scale == 0] = 1
    encoded = np.zeros(mask.shape, dtype=np.float32)
    n = len(numeric)
    encoded[:, :n] = np.nan_to_num((values - center) / scale)
    labels = []
    for i, j in enumerate(categorical):
        codes, vocabulary = pd.factorize(frame.iloc[:, j])
        encoded[:, n + i] = np.maximum(codes, 0)
        labels.append(vocabulary.to_numpy())
    x = torch.from_numpy(encoded)
    m = torch.from_numpy(mask.astype(np.float32))
    model = MIWAE(n, [len(v) for v in labels], cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    epochs = max(
        cfg.min_epochs, math.ceil(cfg.min_updates / math.ceil(len(x) / cfg.batch))
    )
    total_steps = epochs * math.ceil(len(x) / cfg.batch)
    trace_interval = max(1, math.ceil(total_steps / 200))
    step = 0
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(cfg.batch):
            optimizer.zero_grad()
            logw, _, _ = model(x[idx], m[idx], cfg.train_samples)
            loss = -logw.logsumexp(0).mean() + math.log(cfg.train_samples)
            step += 1
            if loss_trace is not None and (
                step == 1 or step % trace_interval == 0 or step == total_steps
            ):
                loss_trace.append([step, float(loss.detach())])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
    pred = np.empty_like(encoded)
    with torch.no_grad():
        for start in range(0, len(x), cfg.batch):
            sl = slice(start, start + cfg.batch)
            logw, means, log_probs = model(x[sl], m[sl], cfg.impute_samples)
            weights = logw.softmax(0).unsqueeze(-1)
            pred[sl, :n] = (weights * means).sum(0).numpy()
            for i, probs in enumerate(log_probs):
                pred[sl, n + i] = (weights * probs.exp()).sum(0).argmax(-1).numpy()
    result = frame.copy()
    for i, j in enumerate(order):
        missing = ~mask[:, i]
        fill = pred[missing, i]
        result.iloc[missing, j] = (
            fill * scale[i] + center[i] if i < n else labels[i - n][fill.astype(int)]
        )
    return result, epochs


def score(frame, filled, truth, categorical):
    if (
        frame.shape != truth.shape
        or filled.isna().any().any()
        or truth.isna().any().any()
    ):
        raise ValueError("Check workbook alignment and missing values")
    residual = norm = 0.0
    hits = count = 0
    for j in range(frame.shape[1]):
        obs = frame.iloc[:, j].notna().to_numpy()
        actual = truth.iloc[:, j].to_numpy()
        estimate = filled.iloc[:, j].to_numpy()
        if j in categorical:
            aligned = np.array_equal(actual[obs], estimate[obs])
            hits += np.count_nonzero(actual[~obs] == estimate[~obs])
            count += (~obs).sum()
        else:
            aligned = np.allclose(actual[obs], estimate[obs], rtol=1e-12, atol=1e-12)
            residual += np.square(actual - estimate).sum()
            norm += np.square(actual).sum()
        if not aligned:
            raise ValueError(
                f"Observed values disagree with reference in feature {j+1}"
            )
    return dict(
        nrms=float(np.sqrt(residual / norm)) if norm else None,
        ae=float(hits / count) if count else None,
        categorical_missing=int(count),
    )


def run(
    dataset=None,
    group="all",
    variant=None,
    cfg=Settings(),
    output=OUTPUT,
    visualizations=True,
):
    if visualizations:
        from miwae_figures import (
            comparison_galleries,
            diagnostics,
            version_current,
            version_figures,
        )
    changed = set()
    names = datasets(group)
    if dataset:
        alias = {"ttteg": "tttteg"}.get(dataset.lower(), dataset.lower())
        names = [n for n in names if n.lower() == alias]
        if not names:
            raise ValueError(f"Choose a dataset from {datasets(group)}")
    torch.set_num_threads(1)
    for name in names:
        truth = None
        folder = DATA / "Incomplete Datasets Without Labels" / name
        paths = sorted(folder.glob(f"{name}_*.xlsx"))
        if variant:
            selection = [variant] if isinstance(variant, str) else variant
            chosen = {f"{name}_{v}".lower() for v in selection}
            paths = [p for p in paths if p.stem.lower() in chosen]
            if len(paths) != len(chosen):
                raise ValueError(
                    f"Unknown variant for {name}: {chosen - {p.stem.lower() for p in paths}}"
                )
        if not paths:
            raise ValueError(f"No matching input in {folder}")
        dest = Path(output) / "imputed" / name
        dest.mkdir(parents=True, exist_ok=True)
        with FileLock(str(dest / ".train.lock")):
            for path in paths:
                record = dest / (path.stem + ".json")
                parquet = record.with_suffix(".parquet")
                cached = record.exists() and parquet.exists()
                if cached:
                    info = json.loads(record.read_text())
                    if info["settings"] != asdict(cfg):
                        raise ValueError(
                            "Use a different output folder for changed settings"
                        )
                    if not visualizations or version_current(output, record):
                        continue
                start = time.monotonic()
                frame = read_frame(path)
                if truth is None:
                    truth = read_frame(
                        DATA / "Original Datasets Without Labels" / f"{name}.xlsx"
                    )
                    if name == "Splice":
                        # Reference DNA symbols have leading spaces absent in the mask files.
                        truth = truth.apply(lambda col: col.str.strip())
                # Excel may omit fully blank boundary rows/columns from its used range.
                if frame.shape[0] > truth.shape[0] or frame.shape[1] > truth.shape[1]:
                    raise ValueError(f"Input exceeds reference dimensions: {path}")
                frame = frame.reindex(index=truth.index, columns=truth.columns)
                cat = (
                    list(range(frame.shape[1]))
                    if name in CATEGORICAL
                    else MIXED.get(name, [])
                )
                if cached:
                    filled = pd.read_parquet(parquet)
                else:
                    loss_trace = []
                    filled, epochs = impute(frame, cat, cfg, loss_trace=loss_trace)
                    metrics = score(frame, filled, truth, cat)
                    filled.to_parquet(parquet.with_suffix(".part"), index=False)
                    parquet.with_suffix(".part").replace(parquet)
                    info = dict(
                        variant=path.stem,
                        settings=asdict(cfg),
                        epochs=epochs,
                        loss_trace=loss_trace,
                        seconds=round(time.monotonic() - start, 3),
                        **metrics,
                    )
                    record.with_suffix(".tmp").write_text(
                        json.dumps(info, allow_nan=False)
                    )
                    record.with_suffix(".tmp").replace(record)
                    print(path.stem, metrics, flush=True)
                if visualizations:
                    version_figures(
                        output, info, diagnostics(frame, filled, truth, cat)
                    )
                    changed.add(name)
    if visualizations:
        return comparison_galleries(output, changed)


def report_rows(output=OUTPUT):
    """Collect completed metrics in the order used by the course template."""
    from openpyxl import load_workbook

    book = load_workbook(DATA / "Table-NRMS-AE.xlsx", read_only=True)
    try:
        identifiers = [
            row[0]
            for row in book.active.values
            if isinstance(row[0], str) and len(row[0].rsplit("_", 2)) == 3
        ]
    finally:
        book.close()

    def key(identifier):
        return identifier.casefold().replace("tttteg_", "ttteg_")

    records = {
        key(record.stem): json.loads(record.read_text())
        for record in (Path(output) / "imputed").glob("*/*.json")
        if record.with_suffix(".parquet").exists()
    }
    if records.keys() - {key(identifier) for identifier in identifiers}:
        raise ValueError("Found results without a corresponding template row")
    rows = {}
    for identifier in identifiers:
        info = records.get(key(identifier))
        if info is None:
            continue
        cfg = info["settings"]
        details = (
            f"h={cfg['hidden']}, d={cfg['latent']}, K={cfg['train_samples']}, "
            f"L={cfg['impute_samples']}, batch={cfg['batch']}, epochs={info['epochs']}, "
            f"Adam lr={cfg['learning_rate']}, seed={cfg['seed']}"
        )
        ae = info["ae"]
        if ae is None and info["variant"].rsplit("_", 2)[0] in MIXED:
            ae = "N/A"
            details += ", 0 hidden categorical cells"
        rows[identifier] = [info["nrms"], ae, details]
    return rows


COMPLEXITY = (
    "For a fixed network, training O(E × N × K) and imputation O(N × L). "
    "Batch activations scale with batch size and sample count. Input/output storage scales with N × features."
)


def export_table(output=OUTPUT):
    """Refresh the Excel report from every finished run in this output folder."""
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "Table-NRMS-AE.xlsx"
    with FileLock(str(output / ".report.lock")):
        rows = report_rows(output)
        book = load_workbook(DATA / target.name)
        sheet = book.active
        for cell in sheet["A"]:
            name = cell.value
            if name in rows:
                for col, value in enumerate(rows[name], 2):
                    sheet.cell(cell.row, col, value)
            if name == "Average NRMS/AE":
                for col in ["B", "C"]:
                    sheet[f"{col}{cell.row}"] = (
                        f'=IFERROR(AVERAGE({col}2:{col}{cell.row-1}),"")'
                    )
            if name == "Complexity Analysis":
                sheet.cell(cell.row, 4, COMPLEXITY).alignment = Alignment(
                    wrap_text=True
                )
                sheet.row_dimensions[cell.row].height = 45
        temp = target.with_suffix(".tmp.xlsx")
        book.save(temp)
        temp.replace(target)
    print(f"{len(rows)} completed variant(s) included in {target}")
    return target
