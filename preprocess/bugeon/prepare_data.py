# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import argparse
import datetime
import logging
from pathlib import Path

import h5py
import numpy as np

from requests import session
from temporaldata import (
    Data,
    ArrayDict,
    IrregularTimeSeries,
    Interval,
)

from brainsets.taxonomy import Species
from brainsets.descriptions import (
    SessionDescription,
    SubjectDescription,
    BrainsetDescription,
)
from brainsets.core import serialize_fn_map

logging.basicConfig(level=logging.INFO)


STIMULUS_SHORTNAME_MAP = {
    "Blank": "bl",
    "Drifting Gratings": "dg",
    "Natural Scenes": "ns",
}


def read_txt_list(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()


def extract_neural_activity(experiment_dir, time_offset: float):
    df_over_f = np.load(f"{experiment_dir}/frame.neuralActivity.npy")
    timestamps = np.load(f"{experiment_dir}/frame.times.npy").squeeze()
    # states = np.load(f"{experiment_dir}/frame.states.npy").squeeze()

    # df_over_f_norm1 = (df_over_f - df_over_f.mean()) / df_over_f.std()
    # df_over_f_norm2 = (
    #     df_over_f - df_over_f.mean(axis=0, keepdims=True)
    # ) / df_over_f.std(axis=0, keepdims=True)
    # df_over_f_norm = df_over_f / np.percentile(df_over_f.flatten(), 99.0)
    return IrregularTimeSeries(
        timestamps=timestamps + time_offset,
        df_over_f=df_over_f,
        # df_over_f_norm1=df_over_f_norm1,
        # df_over_f_norm2=df_over_f_norm2,
        # df_over_f_norm=df_over_f_norm,
        # states=states,
        domain="auto",
    )


def extract_neurons(sess_dir: Path, sess_id: str) -> ArrayDict:
    ttypes = np.array(read_txt_list(f"{sess_dir}/neuron.ttype.txt"))
    num_neurons = len(ttypes)

    id = np.array([f"{sess_id}_u{x}" for x in range(num_neurons)])

    stackpos = np.load(f"{sess_dir}/neuron.stackPosCorrected.npy")

    data = {
        "id": id,
        "ttype": ttypes,
        # "gene_count": np.load(f"{sess_dir}/neuron.gene_count.npy"),
        # "prob_ttypes": np.load(f"{sess_dir}/neuron.prob_ttypes.npy"),
        "depth": np.load(f"{sess_dir}/neuron.depth.npy").squeeze(),
        "stackPos_x": stackpos[:, 0],
        "stackPos_y": stackpos[:, 1],
        "rf_center_xvis": np.load(f"{sess_dir}/neuron.rfCenter_Xvis.npy").squeeze(),
        "rf_center_yvis": np.load(f"{sess_dir}/neuron.rfCenter_Yvis.npy").squeeze(),
        # "rf_map": np.load(f"{sess_dir}/neuron.rfMap.npy"),
        "plane": np.load(f"{sess_dir}/neuron.plane.npy").squeeze(),
        # "unique_id": np.load(f"{sess_dir}/neuron.UniqueID.npy").squeeze(),
    }

    data["ei_class"] = np.array(
        ["IN" if x not in ["IN", "EC"] else x for x in data["ttype"]]
    )
    data["subclass"] = np.array(
        [x.split("-")[0] if x not in ["IN", "EC"] else "unknown" for x in data["ttype"]]
    )

    return ArrayDict(**data)


def extract_running(exp_dir):
    return IrregularTimeSeries(
        timestamps=np.load(f"{exp_dir}/running.times.npy").squeeze(),
        speed=np.load(f"{exp_dir}/running.speed.npy").squeeze(),
        domain="auto",
    )


def extract_eye(exp_dir):
    return IrregularTimeSeries(
        timestamps=np.load(f"{exp_dir}/eye.times.npy").squeeze(),
        size=np.load(f"{exp_dir}/eye.size.npy").squeeze(),
        x_pos=np.load(f"{exp_dir}/eye.xPos.npy").squeeze(),
        y_pos=np.load(f"{exp_dir}/eye.yPos.npy").squeeze(),
        domain="auto",
    )


def extract_dg_stimulus(exp_dir):
    startend = np.load(f"{exp_dir}/ori.intervals.npy")
    return Interval(
        start=startend[:, 0],
        end=startend[:, 1],
        size=np.load(f"{exp_dir}/ori.stimSize.npy").squeeze(),
        direction=np.load(f"{exp_dir}/ori.stimDirection.npy").squeeze(),
        position_xvis=np.load(f"{exp_dir}/ori.stimPosition_Xvis.npy").squeeze(),
        position_yvis=np.load(f"{exp_dir}/ori.stimPosition_Yvis.npy").squeeze(),
        contrast=np.load(f"{exp_dir}/ori.contrast.npy").squeeze(),
    )


def extract_ns_stimulus(exp_dir):
    startend = np.load(f"{exp_dir}/nat.intervals.npy")
    return Interval(
        start=startend[:, 0],
        end=startend[:, 1],
        img_index=np.load(f"{exp_dir}/nat.img.npy").squeeze(),
        blank_indicator=np.load(f"{exp_dir}/nat.blank.npy").squeeze(),
    )


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", type=str)
    parser.add_argument("--raw_dir", type=Path)
    parser.add_argument("--processed_dir", type=Path)
    args = parser.parse_args()

    subject_id, date = args.session_dir.split("/")
    session_id = "_".join((subject_id.lower(), "".join(date.split("-"))))

    raw_dir = Path(args.raw_dir)
    stimulus_dirs = [x for x in (raw_dir / args.session_dir).iterdir() if x.is_dir()]
    experiment_dirs = sum([list(x.iterdir()) for x in stimulus_dirs], [])

    # subject_id, date, stimulus_name, num = args.session.split("/")
    # experiment_id = "_".join((session_id, STIMULUS_SHORTNAME_MAP[stimulus_name], num))

    brainset = BrainsetDescription(
        id="bugeon_transcriptomic_2022",
        origin_version="0.0.0",
        derived_version="0.0.0",
        source="https://www.nature.com/articles/s41586-022-04915-7",
        description="A_transcriptomic_axis_predicts_state_modulation_of_cortical_interneurons",
    )

    subject = SubjectDescription(
        id=subject_id,
        species=Species.UNKNOWN,
    )

    session = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(date, "%Y-%m-%d"),
    )

    # we're stringing together different experiments in the same data file since they all have
    # the same neurons;
    # adding a 1 hour time difference between them so that the two views dont see them together
    calcium_traces_list = [
        extract_neural_activity(x, i * 3600) for i, x in enumerate(experiment_dirs)
    ]
    timestamps = np.concatenate([x.timestamps for x in calcium_traces_list], axis=0)
    df_over_f = np.concatenate([x.df_over_f for x in calcium_traces_list], axis=0)
    df_over_f_norm = df_over_f / np.percentile(df_over_f.flatten(), 99.0)
    domain = calcium_traces_list[0].domain
    for i in range(1, len(calcium_traces_list)):
        domain = domain | calcium_traces_list[i].domain  # type: ignore

    calcium_traces = IrregularTimeSeries(
        timestamps=timestamps,
        df_over_f=df_over_f,
        df_over_f_norm=df_over_f_norm,
        domain=domain,
    )
    assert calcium_traces.is_sorted()

    data = Data(
        brainset=brainset,  # type: ignore
        subject=subject,  # type: ignore
        session=session,  # type: ignore
        units=extract_neurons(sess_dir=args.raw_dir / subject_id / date, sess_id=session_id),  # type: ignore
        calcium_traces=calcium_traces,  # type: ignore
        domain=calcium_traces.domain,
    )

    args.processed_dir.mkdir(exist_ok=True, parents=True)
    out_fname = args.processed_dir / f"{session_id}.h5"
    with h5py.File(out_fname, "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)
    print(f"Output written to: {out_fname}")


if __name__ == "__main__":
    main()
