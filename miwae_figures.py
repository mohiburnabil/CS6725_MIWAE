"""Static report figures for MIWAE imputations and benchmark comparisons."""

from pathlib import Path
import json
from html import escape
from urllib.parse import quote

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from filelock import FileLock

BLUE = "#236aa0"
TEAL = "#087e78"
ORANGE = "#bd6531"
INK = "#203548"
PALE = "#e5edf1"
METRICS = {"nrms": "NRMS (lower is better)", "ae": "AE accuracy (higher is better)"}


def diagnostics(frame, filled, truth, categorical):
    """Compute feature profiles on every cell and sample paired points for legibility."""
    missing = frame.isna().to_numpy()
    counts = missing.sum(axis=0)
    n, p = missing.shape
    all_cols = np.arange(p)
    numeric = [j for j in all_cols if j not in categorical]
    rng = np.random.default_rng(1234)
    sample = np.sort(rng.choice(n, min(n, 250), replace=False))
    data = dict(
        mask=missing[sample],
        counts=counts,
        rows=n,
        numeric=numeric,
        categorical=categorical,
        row_counts=np.bincount(missing.sum(axis=1), minlength=p + 1),
    )
    for kind, columns in (("num", numeric), ("cat", categorical)):
        if not columns or not counts[columns].sum():
            continue
        focus = max(columns, key=lambda j: counts[j])
        data[kind + "_focus"] = focus
        scores = []
        for j in columns:
            mask = missing[:, j]
            actual = truth.iloc[:, j].to_numpy()
            observed = actual[mask]
            predicted = filled.iloc[:, j].to_numpy()[mask]
            if kind == "num":
                norm = np.sum(np.square(actual, dtype=float))
                scores.append(
                    np.sqrt(np.square(observed - predicted).sum() / norm)
                    if norm
                    else np.nan
                )
            else:
                scores.append(np.mean(observed == predicted) if mask.any() else np.nan)
            if j != focus:
                continue
            if kind == "num":
                take = rng.choice(
                    len(observed), min(5000, len(observed)), replace=False
                )
                data["actual"] = observed[take]
                data["predicted"] = predicted[take]
            else:
                # Include the most common labels across both sides, pooling the remaining tail.
                classes = (
                    pd.Series(np.concatenate([observed, predicted]))
                    .value_counts()
                    .index
                )
                labels = list(classes[:9])
                if len(classes) > 9:
                    labels.append("Other classes")
                lookup = {value: i for i, value in enumerate(classes[:9])}
                a = (
                    pd.Series(observed)
                    .map(lookup)
                    .fillna(len(labels) - 1)
                    .to_numpy(dtype=int)
                )
                b = (
                    pd.Series(predicted)
                    .map(lookup)
                    .fillna(len(labels) - 1)
                    .to_numpy(dtype=int)
                )
                k = len(labels)
                data["confusion"] = np.bincount(k * a + b, minlength=k * k).reshape(
                    k, k
                )
                data["labels"] = np.asarray(labels, dtype=str)
        data[kind + "_scores"] = scores
    return data


def chart(title, subtitle, *, width=7.5, height=4.4):
    fig = Figure(figsize=(width, height), facecolor="white")
    FigureCanvasAgg(fig)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.76)
    fig.text(0.055, 0.935, title, fontsize=14, fontweight="bold", color=INK)
    fig.text(0.055, 0.865, subtitle, fontsize=9, color="#4a6072")
    ax = fig.add_subplot()
    ax.tick_params(labelsize=8, colors=INK)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#a8b8c3")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=PALE, linewidth=0.6)
    return fig, ax


def save(fig, folder, name, figures, caption):
    target = folder / (name + ".png")
    tmp = target.with_suffix(".tmp")
    fig.savefig(tmp, format="png", dpi=155, pil_kwargs={"compress_level": 3})
    tmp.replace(target)
    fig.clear()
    figures.append((target.name, caption))


def gallery(folder, title, description, figures, links=()):
    """A self-contained local index, with full-resolution images one click away."""
    folder.mkdir(parents=True, exist_ok=True)
    tiles = "".join(
        f'<figure><a href="{quote(file)}"><img loading="lazy" src="{quote(file)}" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'
        for file, caption in figures
    )
    nav = " ".join(
        f'<a href="{quote(str(url))}">{escape(text)}</a>' for text, url in links
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title>
<style>body{{font:16px system-ui;color:#203548;background:#f6f8fa;max-width:1440px;margin:36px auto;padding:0 24px}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}}nav a{{background:white;padding:6px 10px;border:1px solid #cedce3;border-radius:4px}}a{{color:#185b88}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:16px}}figure{{margin:0;background:white;border:1px solid #d7e1e7}}img{{width:100%;display:block}}figcaption{{padding:8px 14px;color:#354f63}}</style><h1>{escape(title)}</h1><p>{escape(description)}</p><nav>{nav}</nav><main>{tiles}</main></html>"""
    temp = folder / "index.tmp"
    temp.write_text(page, encoding="utf-8")
    temp.replace(folder / "index.html")


def feature_axis(ax, positions):
    ticks = positions[:: max(1, int(np.ceil(len(positions) / 15)))]
    ax.set_xticks(ticks, [f"x{int(j)+1}" for j in ticks], rotation=45, ha="right")
    ax.set_xlabel("Source feature")


def agreement_plot(actual, predicted, subtitle, folder, figures):
    fig, ax = chart("Imputed versus true values", subtitle)
    if len(actual) < 200:
        ax.scatter(actual, predicted, s=25, alpha=0.72, color=BLUE, edgecolors="none")
    else:
        bins = ax.hexbin(
            actual, predicted, gridsize=32, mincnt=1, cmap="viridis", bins="log"
        )
        fig.colorbar(bins, ax=ax, label="Cells (log scale)")
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], color=ORANGE, lw=1.3, ls="--", label="Exact recovery")
    ax.set(xlabel="True value", ylabel="Imputed value")
    ax.legend(fontsize=8)
    save(fig, folder, "numeric-agreement", figures, "True versus imputed agreement")


def version_figures(output, info, data):
    stem = info["variant"]
    dataset, pattern, rate = stem.rsplit("_", 2)
    folder = Path(output) / "visualizations" / dataset / stem
    folder.mkdir(parents=True, exist_ok=True)
    subtitle = f"{dataset}  ·  {pattern}  ·  {rate}% nominal missingness"
    counts, n = np.asarray(data["counts"]), int(data["rows"])
    features = np.arange(len(counts))
    figures = []
    fig, ax = chart(
        "Where values are missing",
        subtitle + f"  ·  {len(data['mask']):,} sampled rows in source order",
    )
    ax.grid(False)
    ax.imshow(
        data["mask"],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([PALE, ORANGE]),
        vmin=0,
        vmax=1,
    )
    ax.set_ylabel("Sampled row position")
    feature_axis(ax, features)
    fig.text(0.66, 0.04, "Light = observed     Rust = missing", size=8, color=INK)
    save(fig, folder, "missingness-map", figures, "Missingness map")
    fig, ax = chart("Missingness by feature", subtitle + f"  ·  all {n:,} rows")
    ax.bar(features, 100 * counts / n, color=BLUE)
    feature_axis(ax, features)
    ax.set_ylabel("Missing cells (%)")
    save(fig, folder, "missingness-by-feature", figures, "Feature missingness profile")
    bins = np.flatnonzero(data["row_counts"])
    fig, ax = chart("How many features are missing per row?", subtitle)
    ax.bar(bins, 100 * np.asarray(data["row_counts"])[bins] / n, color=TEAL)
    ax.set(xlabel="Number of missing features", ylabel="Share of rows (%)")
    ax.set_xticks(bins[:: max(1, int(np.ceil(len(bins) / 12)))])
    save(
        fig,
        folder,
        "missing-features-per-row",
        figures,
        "Distribution of incomplete rows",
    )

    for kind, columns, metric in (
        ("num", "numeric", "NRMS"),
        ("cat", "categorical", "AE accuracy"),
    ):
        if kind + "_focus" not in data:
            continue
        ids = np.asarray(data[columns], dtype=int)
        scores = np.asarray(data[kind + "_scores"])
        fig, ax = chart(
            f"{metric} by feature", subtitle + "  ·  one score per eligible feature"
        )
        valid = np.isfinite(scores)
        if kind == "num":
            ax.vlines(ids[valid], 0, scores[valid], color=BLUE, alpha=0.65)
        ax.scatter(ids[valid], scores[valid], s=30, color=TEAL)
        if kind == "cat":
            ax.set_ylim(-0.04, 1.04)
        feature_axis(ax, ids)
        ax.set_ylabel(metric)
        save(
            fig, folder, kind + "-feature-scores", figures, f"{metric} feature profile"
        )
    if "num_focus" in data:
        a, b = data["actual"], data["predicted"]
        j = int(data["num_focus"])
        sub = (
            subtitle + f"  ·  x{j+1}  ·  {len(a):,} of {int(counts[j]):,} hidden cells"
        )
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        agreement_plot(a, b, sub, folder, figures)
        fig, ax = chart("Distributions of true and imputed values", sub)
        bins = np.histogram_bin_edges([lo, hi], bins=32)
        ax.hist(a, bins=bins, alpha=0.65, color=BLUE, label="True")
        ax.hist(b, bins=bins, histtype="step", color=ORANGE, lw=2, label="Imputed")
        ax.set(xlabel=f"x{j+1} in original units", ylabel="Hidden cells")
        ax.legend()
        save(fig, folder, "numeric-distributions", figures, "Overlaid histograms")
        fig, ax = chart("Signed residuals across the true range", sub)
        ax.scatter(a, b - a, s=8, alpha=0.35, edgecolors="none", color=TEAL)
        ax.axhline(0, color=ORANGE, ls="--", lw=1)
        ax.set(xlabel="True value", ylabel="Imputed − true")
        save(fig, folder, "numeric-residuals", figures, "Residual scatterplot")
        fig, ax = chart("Cumulative distribution of absolute error", sub)
        err = np.sort(np.abs(b - a))
        ax.step(
            err, np.arange(1, len(err) + 1) / len(err), where="post", color=BLUE, lw=2
        )
        ax.set(
            xlabel="Absolute error in original units",
            ylabel="Fraction at or below error",
            ylim=(0, 1.03),
        )
        save(fig, folder, "absolute-error-ecdf", figures, "Absolute-error ECDF")
    if "cat_focus" in data:
        j = int(data["cat_focus"])
        sub = subtitle + f"  ·  x{j+1}  ·  all {int(counts[j]):,} hidden cells"
        cm = data["confusion"]
        labels = list(data["labels"])
        fig, ax = chart("Which categories were confused?", sub, height=5.2)
        fig.subplots_adjust(bottom=0.30, left=0.23)
        ax.grid(False)
        freq = np.divide(
            cm,
            cm.sum(1, keepdims=True),
            out=np.full(cm.shape, np.nan),
            where=cm.sum(1, keepdims=True) > 0,
        )
        im = ax.imshow(freq, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax, label="Fraction of true class")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlabel="Imputed category", ylabel="True category")
        save(
            fig,
            folder,
            "categorical-confusion",
            figures,
            "Row-normalized confusion matrix",
        )
        fig, ax = chart("True and imputed category frequencies", sub, height=4.8)
        fig.subplots_adjust(bottom=0.29)
        x = np.arange(len(labels))
        ax.bar(x - 0.19, cm.sum(1) / cm.sum(), width=0.38, color=BLUE, label="True")
        ax.bar(
            x + 0.19, cm.sum(0) / cm.sum(), width=0.38, color=ORANGE, label="Imputed"
        )
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("Fraction of hidden cells")
        ax.legend(fontsize=8)
        save(
            fig,
            folder,
            "categorical-frequencies",
            figures,
            "Grouped category-frequency bars",
        )
    if info.get("loss_trace"):
        steps, losses = np.array(info["loss_trace"]).T
        fig, ax = chart(
            "Training objective across optimizer updates",
            subtitle
            + (
                "  ·  replayed minibatch loss"
                if info.get("loss_trace_replayed")
                else "  ·  sampled minibatch loss"
            ),
        )
        ax.plot(steps, losses, lw=1.3, color=TEAL)
        ax.set(xlabel="Optimizer update", ylabel="Negative MIWAE bound")
        save(fig, folder, "training-objective", figures, "Training loss trace")
    scores = " · ".join(
        f"{k.upper()} {info[k]:.4f}" for k in METRICS if info[k] is not None
    )
    gallery(
        folder,
        stem,
        f"{scores}. Detailed recovery plots use the feature with the most hidden cells of each type. Select a figure to open its PNG.",
        figures,
        [("Compare versions", "../index.html"), ("All datasets", "../../index.html")],
    )
    return folder


def _heatmap(folder, figures, key, title, sub, matrix, metric, xlabel, ylabel):
    vals = matrix.to_numpy(float)
    fig, ax = chart(title, sub, width=8, height=max(4.2, 2.3 + 0.23 * len(matrix)))
    fig.subplots_adjust(left=0.20, bottom=0.19, top=0.82)
    ax.grid(False)
    im = ax.imshow(
        np.ma.masked_invalid(vals),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=1 if metric == "ae" else max(np.nanmax(vals), 1e-9),
    )
    ax.set_facecolor("#e6e7e9")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=ax, label=METRICS[metric])
    save(fig, folder, key, figures, title)


def _box(ax, groups, labels, horizontal=False):
    # bxp also handles groups with only one observation in a partial benchmark.
    boxes = ax.boxplot(
        groups,
        vert=not horizontal,
        patch_artist=True,
        medianprops={"color": ORANGE, "linewidth": 1.6},
        flierprops={"markersize": 2, "markeredgecolor": BLUE},
    )
    for patch in boxes["boxes"]:
        patch.set(facecolor="#c7e1ee", edgecolor=BLUE)
    if horizontal:
        ax.set_yticks(np.arange(1, len(labels) + 1), labels)
    else:
        ax.set_xticks(np.arange(1, len(labels) + 1), labels)


def dataset_gallery(root, name, rows):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    figures = []
    tag = f"{name}  ·  {len(rows)} completed version{'s' if len(rows)!=1 else ''}"
    for metric, axis in METRICS.items():
        valid = rows.dropna(subset=[metric])
        if valid.empty:
            continue
        matrix = valid.pivot(index="pattern", columns="rate", values=metric).reindex(
            index=sorted(rows.pattern.unique()), columns=sorted(rows.rate.unique())
        )
        _heatmap(
            folder,
            figures,
            metric + "-pattern-rate",
            f"{metric.upper()} by mask and rate",
            tag,
            matrix,
            metric,
            "Nominal missingness (%)",
            "Mask code",
        )
        fig, ax = chart(f"How {metric.upper()} changes with missingness", tag)
        for _, series in matrix.iterrows():
            ax.plot(
                matrix.columns, series, color="#8eb3c6", lw=1, marker=".", alpha=0.8
            )
        ax.plot(
            matrix.columns,
            matrix.median(),
            color=ORANGE,
            marker="o",
            lw=2,
            label="Median across mask codes",
        )
        ax.set(xlabel="Nominal missingness (%)", ylabel=axis)
        ax.set_xticks(matrix.columns)
        ax.legend(fontsize=8)
        save(
            fig,
            folder,
            metric + "-rate-traces",
            figures,
            f"{metric.upper()} rate trajectories",
        )
        fig, ax = chart(
            f"Spread of {metric.upper()} at each rate",
            tag + "  ·  each dot is a scored version",
        )
        groups = list(valid.groupby("rate", sort=True)[metric])
        _box(ax, [s.to_numpy() for _, s in groups], [str(k) + "%" for k, _ in groups])
        for i, (_, s) in enumerate(groups, 1):
            ax.scatter(
                i + np.linspace(-0.12, 0.12, len(s)),
                s,
                s=14,
                color=TEAL,
                alpha=0.8,
                zorder=3,
            )
        ax.set(xlabel="Nominal missingness", ylabel=axis)
        save(
            fig,
            folder,
            metric + "-rate-boxplot",
            figures,
            f"{metric.upper()} box-and-dot plot",
        )
        means = valid.groupby("pattern")[metric].mean().sort_values()
        fig, ax = chart(
            f"Mean {metric.upper()} by mask code",
            tag + "  ·  equal weight per scored version",
        )
        ax.barh(means.index, means, color=BLUE)
        ax.set(xlabel=axis, ylabel="Mask code")
        save(
            fig,
            folder,
            metric + "-mask-means",
            figures,
            f"Mask-code comparison for {metric.upper()}",
        )
    fig, ax = chart(
        "Elapsed time by missingness rate",
        tag + "  ·  recorded run time, excluding figure rendering",
    )
    groups = list(rows.groupby("rate")["seconds"])
    _box(ax, [s for _, s in groups], [str(k) + "%" for k, _ in groups])
    ax.set(xlabel="Nominal missingness", ylabel="Elapsed seconds")
    save(fig, folder, "runtime-by-rate", figures, "Runtime distribution")
    links = [("All datasets", "../index.html")]
    for row in rows.sort_values(["pattern", "rate"]).itertuples():
        if (folder / row.variant / "index.html").is_file():
            links.append((f"{row.pattern} · {row.rate}%", f"{row.variant}/index.html"))
    gallery(
        folder,
        f"{name} version comparisons",
        "Choose a version for its missingness and imputation diagnostics. Gray heatmap cells have no score.",
        figures,
        links,
    )


def comparison_galleries(output, changed=None):
    output = Path(output)
    root = output / "visualizations"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".figure-comparisons.lock")):
        records = []
        for path in sorted((output / "imputed").glob("*/*.json")):
            if path.with_suffix(".parquet").exists():
                r = json.loads(path.read_text())
                name, pattern, rate = r["variant"].rsplit("_", 2)
                records.append(
                    dict(
                        dataset=name,
                        pattern=pattern,
                        rate=int(rate),
                        variant=r["variant"],
                        nrms=r["nrms"],
                        ae=r["ae"],
                        seconds=r["seconds"],
                        epochs=r["epochs"],
                    )
                )
        if not records:
            return
        rows = pd.DataFrame(records)
        for name, group in rows.groupby("dataset"):
            if (
                changed is None
                or name in changed
                or not (root / name / "index.html").exists()
            ):
                dataset_gallery(root, name, group)
        folder = root / "common"
        folder.mkdir(exist_ok=True)
        figures = []
        subtitle = (
            f"{rows.dataset.nunique()} datasets  ·  {len(rows):,} completed versions"
        )
        for metric, axis in METRICS.items():
            valid = rows.dropna(subset=[metric])
            if valid.empty:
                continue
            metric_summary = f"{valid.dataset.nunique()} eligible datasets  ·  {len(valid):,} scored versions"
            groups = sorted(
                valid.groupby("dataset")[metric], key=lambda item: item[1].median()
            )
            fig, ax = chart(
                f"{metric.upper()} across datasets",
                metric_summary + "  ·  sorted by median",
                height=max(4.4, 2.2 + 0.29 * len(groups)),
            )
            fig.subplots_adjust(left=0.18, top=0.83)
            _box(
                ax,
                [s for _, s in groups],
                [name for name, _ in groups],
                horizontal=True,
            )
            ax.set_xlabel(axis)
            save(
                fig,
                folder,
                metric + "-dataset-boxplot",
                figures,
                f"Distribution of {metric.upper()} by dataset",
            )
            for dim, label in (
                ("rate", "nominal missingness (%)"),
                ("pattern", "mask code"),
            ):
                matrix = valid.pivot_table(
                    index="dataset", columns=dim, values=metric, aggfunc="mean"
                )
                _heatmap(
                    folder,
                    figures,
                    f"{metric}-dataset-{dim}",
                    f"Mean {metric.upper()} by dataset and {dim}",
                    metric_summary + "  ·  equal weight per scored version",
                    matrix,
                    metric,
                    label.capitalize(),
                    "Dataset",
                )
            rates = valid.pivot_table(
                index="dataset", columns="rate", values=metric, aggfunc="mean"
            )
            if rates.shape[1] > 1:
                fig, ax = chart(
                    f"{metric.upper()} as missingness increases",
                    metric_summary + "  ·  one mean per dataset at each rate",
                )
                ax.fill_between(
                    rates.columns,
                    rates.quantile(0.25),
                    rates.quantile(0.75),
                    color="#c5e0eb",
                    label="Middle 50% of dataset means",
                )
                ax.plot(
                    rates.columns,
                    rates.median(),
                    marker="o",
                    color=BLUE,
                    linewidth=2,
                    label="Median dataset mean",
                )
                ax.set_xticks(rates.columns)
                ax.set(xlabel="Nominal missingness (%)", ylabel=axis)
                ax.legend(fontsize=8)
                save(
                    fig,
                    folder,
                    metric + "-rate-ribbon",
                    figures,
                    f"{metric.upper()} rate trend and interquartile band",
                )
                lo, hi = rates.columns.min(), rates.columns.max()
                pairs = rates[[lo, hi]].dropna().sort_values(hi)
                if not pairs.empty:
                    fig, ax = chart(
                        f"{metric.upper()} at {lo}% and {hi}% missingness",
                        metric_summary + "  ·  mean across available mask codes",
                        height=max(4.4, 2.2 + 0.29 * len(pairs)),
                    )
                    fig.subplots_adjust(left=0.18, top=0.81)
                    y = np.arange(len(pairs))
                    ax.hlines(y, pairs[lo], pairs[hi], color="#acbdc9", lw=2)
                    ax.scatter(pairs[lo], y, color=BLUE, label=f"{lo}%", s=24)
                    ax.scatter(pairs[hi], y, color=ORANGE, label=f"{hi}%", s=24)
                    ax.set_yticks(y, pairs.index)
                    ax.set_xlabel(axis)
                    ax.legend(fontsize=8)
                    save(
                        fig,
                        folder,
                        metric + "-rate-endpoints",
                        figures,
                        f"Paired low- and high-missingness means for {metric.upper()}",
                    )
            fig, ax = chart(
                f"Distribution of {metric.upper()} across versions", metric_summary
            )
            values = np.sort(valid[metric])
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                lw=2,
                color=BLUE,
            )
            ax.set(xlabel=axis, ylabel="Fraction of scored versions", ylim=(0, 1.03))
            save(
                fig,
                folder,
                metric + "-ecdf",
                figures,
                f"Benchmark-wide {metric.upper()} ECDF",
            )
        epochs = (
            rows.groupby("dataset")
            .epochs.agg(["min", "median", "max"])
            .sort_values("median")
        )
        fig, ax = chart(
            "Training epochs per dataset",
            subtitle + "  ·  median and range across completed versions",
            height=max(4.4, 2.2 + 0.27 * len(epochs)),
        )
        fig.subplots_adjust(left=0.18, top=0.83)
        y = np.arange(len(epochs))
        ax.hlines(y, epochs["min"], epochs["max"], color=BLUE, lw=2)
        ax.scatter(epochs["median"], y, color=TEAL, s=28)
        ax.set_yticks(y, epochs.index)
        ax.set_xlabel("Epochs per version")
        save(fig, folder, "epochs-per-dataset", figures, "Training budget by dataset")
        times = rows.groupby("dataset").seconds.median().sort_values()
        fig, ax = chart(
            "Typical elapsed time per version",
            subtitle + "  ·  median recorded seconds, excluding plots",
            height=max(4.4, 2.2 + 0.27 * len(times)),
        )
        fig.subplots_adjust(left=0.18, top=0.83)
        ax.barh(times.index, times, color=TEAL)
        ax.set_xlabel("Median elapsed seconds")
        save(fig, folder, "runtime-per-dataset", figures, "Median runtime per dataset")
        counts = rows.groupby("dataset")[["nrms", "ae"]].count()
        fig, ax = chart(
            "Scored versions by dataset",
            subtitle,
            height=max(4.4, 2 + 0.28 * len(counts)),
        )
        fig.subplots_adjust(left=0.18, top=0.83)
        y = np.arange(len(counts))
        ax.barh(y - 0.19, counts.nrms, height=0.36, color=BLUE, label="NRMS")
        ax.barh(y + 0.19, counts.ae, height=0.36, color=ORANGE, label="AE")
        ax.set_yticks(y, counts.index)
        ax.set_xlabel("Versions with a defined score")
        ax.legend(fontsize=8)
        save(fig, folder, "metric-coverage", figures, "Metric coverage by dataset")
        both = rows.dropna(subset=["nrms", "ae"])
        if not both.empty:
            fig, ax = chart(
                "Numerical and categorical recovery on mixed data",
                subtitle + "  ·  each dot is one mixed version",
            )
            for name, sub in both.groupby("dataset"):
                ax.scatter(sub.nrms, sub.ae, s=17, alpha=0.65, label=name)
            ax.set(xlabel=METRICS["nrms"], ylabel=METRICS["ae"])
            ax.legend(fontsize=8)
            save(
                fig,
                folder,
                "mixed-metric-scatter",
                figures,
                "NRMS versus AE for mixed versions",
            )
        links = [
            (name, f"../{name}/index.html") for name in sorted(rows.dataset.unique())
        ]
        gallery(
            folder,
            "Cross-dataset comparisons",
            subtitle
            + ". Metrics exclude undefined scores. Runtime depends on the execution environment.",
            figures,
            links,
        )
        gallery(
            root,
            "MIWAE figure gallery",
            subtitle
            + ". Open a dataset to compare its masks and inspect the original-unit diagnostics for each version.",
            [("common/" + f, t) for f, t in figures],
            [("Common comparisons", "common/index.html")]
            + [(n, f"{n}/index.html") for n in sorted(rows.dataset.unique())],
        )
    return root / "index.html"


def version_current(output, record):
    """Repair missing images on resume, even when the training record is current."""
    from html.parser import HTMLParser

    class Images(HTMLParser):
        def __init__(self):
            super().__init__()
            self.files = []

        def handle_starttag(self, tag, attrs):
            if tag == "img":
                self.files.append(dict(attrs)["src"])

    folder = Path(output) / "visualizations" / record.parent.name / record.stem
    index = folder / "index.html"
    if not index.exists() or index.stat().st_mtime_ns < max(
        record.stat().st_mtime_ns, record.with_suffix(".parquet").stat().st_mtime_ns
    ):
        return False
    images = Images()
    images.feed(index.read_text(encoding="utf-8"))
    return bool(images.files) and all((folder / f).is_file() for f in images.files)
