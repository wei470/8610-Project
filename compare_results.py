"""Combine and plot the outputs of Energy_KAN.ipynb and Energy_MLP.ipynb."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent / "results"
MODELS = ("kan", "mlp")
SPLITS = ("row", "shape")
LABELS = {"kan": "RBF-KAN", "mlp": "MLP"}
MAX_EPOCHS = 250


def read_combined(filename, order):
    frames = [pd.read_csv(OUT / model / filename) for model in MODELS]
    return pd.concat(frames, ignore_index=True).sort_values(order, kind="stable")


def combine_results():
    kan_env, mlp_env = [json.loads((OUT / model / "environment.json").read_text(encoding="utf-8"))
                        for model in MODELS]
    matched_settings = ("split_seed", "evaluation_seeds", "tuning_seed", "max_epochs",
                        "patience", "batch", "data_sha256")
    assert all(kan_env[key] == mlp_env[key] for key in matched_settings)
    assert kan_env["kan_basis"] == "gaussian_rbf_only"
    kan_split, mlp_split = [pd.read_csv(OUT / model / "split_manifest.csv") for model in MODELS]
    pd.testing.assert_frame_equal(kan_split, mlp_split)

    runs = read_combined("runs.csv", ["protocol", "model", "seed"])
    tuning = read_combined("tuning.csv", ["protocol", "model", "width", "lr"])
    pred = read_combined("predictions.csv", ["protocol", "model", "seed"])
    curves = read_combined("learning_curves.csv", ["protocol", "model", "seed", "epoch"])
    summary = read_combined("summary.csv", ["protocol", "model"])
    assert len(runs) == 12 and len(tuning) == 24 and len(summary) == 4
    expected_board = kan_split.set_index("row_id")
    for (protocol, model, seed), group in pred.groupby(["protocol", "model", "seed"]):
        actual_rows = group["row_id"].sort_values().tolist()
        test_rows = expected_board.index[expected_board[protocol] == "test"].tolist()
        assert actual_rows == test_rows, (protocol, model, seed)
    for filename, data in (("runs", runs), ("tuning", tuning), ("predictions", pred),
                           ("learning_curves", curves), ("summary", summary),
                           ("split_manifest", kan_split)):
        data.to_csv(OUT / f"{filename}.csv", index=False)
    (OUT / "environment.json").write_text(json.dumps(kan_env, indent=2) + "\n", encoding="utf-8")
    return pred, curves


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)

PALETTE = {"mlp": "#185C94", "kan": "#A54B33"}
INK = "#15283A"

def scatter_plot(protocol, pred):
    image = Image.new("RGB", (1020, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 16), f"{protocol.title()} split  |  observed and predicted loads  |  seed 101",
              fill=INK, font=font(24))
    data = pred.query("protocol == @protocol and seed == 101")
    for j, target in enumerate(("heating", "cooling")):
        x0, y0, x1, y1 = 76+j*500, 91, 462+j*500, 405
        obs, estimates = f"{target}_actual", f"{target}_pred"
        lo = np.floor(data[[obs, estimates]].min().min()/10)*10
        hi = np.ceil(data[[obs, estimates]].max().max()/10)*10
        def xy(x, y):
            return (x0+(x-lo)/(hi-lo)*(x1-x0), y1-(y-lo)/(hi-lo)*(y1-y0))
        for tick in np.linspace(lo, hi, 5):
            tx, ty = xy(tick, tick)
            draw.line((tx,y0,tx,y1), fill="#E1E7ED")
            draw.line((x0,ty,x1,ty), fill="#E1E7ED")
            draw.text((tx, y1+10), f"{tick:g}", font=font(14), fill=INK, anchor="mt")
            draw.text((x0-8, ty), f"{tick:g}", font=font(14), fill=INK, anchor="rm")
        draw.line((*xy(lo,lo), *xy(hi,hi)), fill="#788898", width=2)
        draw.rectangle((x0,y0,x1,y1), outline="#92A0AE", width=1)
        for model, group in data.groupby("model", sort=False):
            for a, b in zip(group[obs], group[estimates], strict=True):
                px, py = xy(a,b)
                draw.ellipse((px-3,py-3,px+3,py+3), outline=PALETTE[model], width=2)
        draw.text((x0, 63), target.title() + " load  |  predicted (vertical)",
                  font=font(17), fill=INK)
        draw.text(((x0+x1)/2, 445), "Observed load", font=font(16), fill=INK, anchor="mt")
    for j, (model, color) in enumerate(PALETTE.items()):
        x = 380+145*j
        draw.ellipse((x,471,x+10,481), outline=color, width=2)
        draw.text((x+17,468), LABELS[model], font=font(16), fill=INK)
    path = OUT / f"{protocol}_predictions.png"
    image.save(path)

def validation_plot(curves):
    image = Image.new("RGB", (1020, 465), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 16), "Validation objective by epoch  |  seed 101", font=font(24), fill=INK)
    for j, protocol in enumerate(SPLITS):
        x0,y0,x1,y1 = 78+j*500, 95, 465+j*500, 374
        data = curves.query("protocol == @protocol and seed == 101")
        lo, hi = -4.0, 0.5
        def xy(epoch, loss):
            return x0+(epoch-1)/(MAX_EPOCHS-1)*(x1-x0), y1-(np.log10(loss)-lo)/(hi-lo)*(y1-y0)
        for exponent in range(-4,1):
            _,ty = xy(1, 10.0**exponent)
            draw.line((x0,ty,x1,ty), fill="#E1E7ED")
            draw.text((x0-7,ty), f"1e{exponent}", fill=INK, font=font(14), anchor="rm")
        draw.rectangle((x0,y0,x1,y1), outline="#92A0AE")
        draw.text((x0,65), protocol.title()+" split  |  validation MSE", font=font(17), fill=INK)
        for ep in (1, 50, 100, 150, 200, 250):
            tx,_=xy(ep,1)
            draw.text((tx,y1+9), str(ep), font=font(14), fill=INK, anchor="mt")
        for model, group in data.groupby("model"):
            draw.line([xy(e,m) for e,m in zip(group.epoch, group.val_mse, strict=True)],
                      width=3, fill=PALETTE[model])
    draw.text((502, 414), "Epoch", font=font(16), fill=INK, anchor="mt")
    for j, (model, color) in enumerate(PALETTE.items()):
        x = 380+145*j
        draw.line((x,443,x+12,443), fill=color, width=3)
        draw.text((x+17,435), LABELS[model], font=font(16), fill=INK)
    path = OUT / "validation_curves.png"
    image.save(path)


if __name__ == "__main__":
    predictions, learning_curves = combine_results()
    for split in SPLITS:
        scatter_plot(split, predictions)
    validation_plot(learning_curves)
