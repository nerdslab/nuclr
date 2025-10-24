# pyright: reportAttributeAccessIssue=false

from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from temporaldata import Data, ArrayDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=Path, help="Directory containing processed data files"
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to save the output CSV file"
    )
    args = parser.parse_args()

    df_list = []
    for fname in tqdm(args.data_dir.iterdir()):
        if not fname.name.endswith(".h5"):
            continue
        with h5py.File(fname, "r") as f:
            data = Data.from_hdf5(f)
            df = arraydict_to_df(data.units)
            df["session_id"] = data.session.id
            df["subject_id"] = data.subject.id

        df_list.append(df)

    print(f"{len(df_list)=}")

    df = pd.concat(df_list)
    assert not df["id"].duplicated().any()

    df.to_csv(args.output_file, index=False)
    print(f"Saved dataframe with {len(df)} rows to {args.output_file}")


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


if __name__ == "__main__":
    main()
