"""Quick checks (run: python -m pytest -q  OR  python tests/test_pipeline.py)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_data, make_splits, feature_sets, file_md5, TARGET, SPLIT_PATH
from src.models import make_pipeline, build_model

cfg = json.loads((ROOT / "configs" / "baseline.json").read_text())

def test_dataset_version():
    assert file_md5() == cfg["dataset_md5"]

def test_shape_and_classes():
    df = load_data()
    assert df.shape == (4424, 37) and df.isna().sum().sum() == 0
    assert set(df[TARGET]) == {"Dropout", "Enrolled", "Graduate"}

def test_split_is_disjoint_complete_and_stratified():
    df = load_data(); Xtr, Xva, Xte, ytr, yva, yte = make_splits(df)
    a, b, c = set(Xtr.index), set(Xva.index), set(Xte.index)
    assert not (a & b) and not (a & c) and not (b & c) and len(a | b | c) == len(df)
    assert (len(Xtr), len(Xva), len(Xte)) == (3096, 664, 664)
    for y in (ytr, yva, yte):
        assert abs((y == "Graduate").mean() - 0.499) < 0.01

def test_stored_split_matches():
    if SPLIT_PATH.exists():
        df = load_data(); Xtr, Xva, Xte, *_ = make_splits(df)
        s = json.loads(SPLIT_PATH.read_text())
        assert s["train"] == Xtr.index.tolist() and s["test"] == Xte.index.tolist()

def test_feature_sets():
    fs = feature_sets(load_data())
    assert len(fs["early"]) == 30 and len(fs["full"]) == 36
    assert not any("2nd sem" in c for c in fs["early"])

def test_pipeline_fits_and_predicts():
    df = load_data(); Xtr, Xva, _, ytr, yva, _ = make_splits(df); cols = feature_sets(df)["early"]
    p = make_pipeline(cols, build_model("logreg")).fit(Xtr[cols], ytr)
    assert (p.predict(Xva[cols]) == yva).mean() > 0.70

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
