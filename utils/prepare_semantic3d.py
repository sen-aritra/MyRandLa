import numpy as np
from pathlib import Path

ROOT_PATH = (Path(__file__) / ".." / "..").resolve()
DATASET_PATH = ROOT_PATH / "urban_scenes_velodyne"
RAW_PATH = DATASET_PATH / "urban_scenes_velodyne"
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"
VAL_PATH = DATASET_PATH / "val"

md = {'ground':1, 'house':2, 'fence':3, 'person':4, 'street_sign':5, 'tree':6, 'car':7, 'other':0}

for folder in [TRAIN_PATH, TEST_PATH, VAL_PATH]:
    folder.mkdir(exist_ok=True)

for pc_path in RAW_PATH.iterdir():
    if pc_path.suffixes != []:
        continue
    pc_name = pc_path.stem + ".npy"

    if list(ROOT_PATH.rglob(pc_name)) != []:
        continue

    points = np.loadtxt(pc_path, usecols=np.arange(1, 4), dtype=np.float32)
    if "1" not in pc_name or "0" in pc_name:
        label_s = np.loadtxt(pc_path, usecols=np.array([4]), dtype=object)
        labels = np.vectorize(md.get)(label_s)
    else:
        np.save(TEST_PATH / pc_name, points)
        continue

    dir = VAL_PATH if "9" in pc_name else TRAIN_PATH
    np.save(dir / pc_name, np.vstack((points.T, labels)).T)
print("Done.")
