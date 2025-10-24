# pyright: reportAttributeAccessIssue=false

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from temporaldata import Data, ArrayDict


def arraydict_to_df(ad: ArrayDict) -> pd.DataFrame:
    data = {}
    for k in ad.keys():
        val = getattr(ad, k)

        # Convert byte strings to regular strings if present
        if isinstance(val, np.ndarray) and val.dtype.kind == "O":
            val = np.array(
                [s.decode("utf-8") if isinstance(s, bytes) else s for s in val]
            )

        data[k] = val

    return pd.DataFrame(data)


root = Path("../data/processed/steinmetz_2019")
fname = next(iter(root.iterdir()))

df_list = []
for fname in tqdm(root.iterdir()):
    if not fname.name.endswith(".h5"):
        continue
    with h5py.File(fname, "r") as f:
        data = Data.from_hdf5(f)
        subject_id = str(data.subject.id)
        units = data.units.materialize()

    sess_id = fname.name.removesuffix(".h5")
    df = arraydict_to_df(units)
    df["session_id"] = sess_id
    df["subject_id"] = subject_id

    df_list.append(df)

print(f"{len(df_list)=}")
df = pd.concat(df_list)
output_file = "./neuron_metadata/steinmetz.csv"
df.to_csv(output_file, index=False)
print(f"Saved dataframe with {len(df)} rows to {output_file}")
