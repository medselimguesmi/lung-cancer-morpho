from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LUNA = ROOT / "data" / "luna16"
N_DEV, N_TEST = 15, 10
SEED = 42

ann = pd.read_csv(LUNA / "annotations.csv")
nodules_par_scan = ann.groupby("seriesuid").size()

rows = []
for subset, split, n in [("subset0", "dev", N_DEV), ("subset1", "test", N_TEST)]:
    uids = sorted(p.stem for p in (LUNA / subset).rglob("*.mhd"))
    avec_nodule = [u for u in uids if u in nodules_par_scan.index]
    print(f"{subset}: {len(uids)} scanners, {len(avec_nodule)} avec au moins un nodule")
    choisis = pd.Series(avec_nodule).sample(n=min(n, len(avec_nodule)), random_state=SEED)
    for uid in sorted(choisis):
        rows.append({"seriesuid": uid, "subset": subset, "split": split,
                     "n_nodules": int(nodules_par_scan[uid])})

out = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
out.to_csv(ROOT / "results" / "patients.csv", index=False)
print(out)
print(f"\n{len(out)} patients enregistrés dans results/patients.csv")