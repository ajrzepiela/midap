import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pytest import fixture, mark

from midap.tracking.delta_lineage import DeltaTypeLineages
from midap.tracking.tracking_analysis import FluoChangeAnalysis


# Fixtures
##########


@fixture()
@mark.usefixtures("tracking_instance")
def fake_data_output(tracking_instance):
    """
    Creates a Lineages instance and generates the lineages with the tracking output of the fake data used in the
    deltav2 tests
    :param tracking_instance: A pytest fixture of a DeltaV2Tracking instance from the deltav2 test file
    :return: The path to the CSV file
    """

    # A temp directory to save everything
    tmp_dir = tempfile.TemporaryDirectory()

    # run the tracking
    inputs, results_all = tracking_instance.run_model_crop()

    # create the instance
    lin = DeltaTypeLineages(np.array(inputs), results_all, connectivity=2)

    # save track output to file
    out_file = os.path.join(tmp_dir.name, "track_output_delta.csv")
    lin.track_output.to_csv(out_file)

    yield out_file

    # cleanup
    tmp_dir.cleanup()


@fixture()
def example_data_output():
    """
    Creates a Lineages instance and generates the lineages with the tracking output of the example data of the repo
    saved in the test_data directory and saves it into a temporary directory as CSV.
    :return: The path to the CSV file
    """

    # load the data
    path = Path(__file__).absolute().parent.joinpath("test_data")
    # results
    data_res = np.load(path.joinpath("results_all_red.npz"))
    results_all_red = data_res["results_all_red"]
    # inputs
    data_inp = np.load(path.joinpath("inputs_all_red.npz"))
    inputs_all = data_inp["inputs_all"]

    # create the instance
    lin = DeltaTypeLineages(inputs_all, results_all_red, connectivity=2)

    # creat lineages
    lin.generate_lineages()

    # A temp directory to save everything
    tmp_dir = tempfile.TemporaryDirectory()
    out_file = os.path.join(tmp_dir.name, "track_output_delta.csv")

    # save to file
    lin.track_output.to_csv(out_file)

    # save lineage to file
    raw_inputs = lin.inputs[:, :, :, 0]
    data_file = os.path.join(tmp_dir.name, "tracking_delta.h5")
    with h5py.File(data_file, "w") as hf:
        hf.create_dataset("images", data=raw_inputs.astype(float), dtype=float)
        hf.create_dataset("labels", data=lin.label_stack.astype(int), dtype=int)

    yield out_file

    # cleanup
    tmp_dir.cleanup()


# Tests
#######


def test_ID_assigmment(example_data_output):
    """
    Test assignment of mother and daughter IDs.
    :param example_data_output: The fixture creating an output of lineages in a temporary directory
    """

    track_output = pd.read_csv(example_data_output, index_col="Unnamed: 0")

    ix_split = track_output.index[track_output.split == 1][0]
    trackID = track_output.loc[ix_split].trackID
    frame = track_output.loc[ix_split].frame
    trackID_d1 = track_output.loc[ix_split].trackID_d1
    trackID_d2 = track_output.loc[ix_split].trackID_d2

    filter_d1 = track_output.index[
        (track_output.frame == frame + 1) & (track_output.trackID == trackID_d1)
    ][0]
    trackID_mother_d1 = track_output.loc[filter_d1].trackID_mother

    assert trackID == trackID_mother_d1

    filter_d2 = track_output.index[
        (track_output.frame == frame + 1) & (track_output.trackID == trackID_d2)
    ][0]
    trackID_mother_d2 = track_output.loc[filter_d2].trackID_mother

    assert trackID == trackID_mother_d2


def test_fake_lineage(fake_data_output):
    """
    Tests the lineage output of the fake data
    :param fake_data_output: The path to the output CSV generated with fake data from the deltav2 tests
    """

    df = pd.read_csv(fake_data_output)

    # check if everything checks out
    assert np.all(df["frame"] == np.array([0, 1, 2, 2]))
    assert np.all(df["trackID"] == np.array([1, 1, 2, 3]))
    assert np.all(df["lineageID"] == np.array([1, 1, 1, 1]))
    assert np.all(df["trackID_d1"][:2] == np.array([2, 2]))
    assert np.all(df["trackID_d2"][:2] == np.array([3, 3]))
    assert np.all(df["trackID_mother"][2:] == np.array([1, 1]))


def test_fluo_change_analysis(example_data_output):
    path = Path(example_data_output).parent
    channels = ["ph", "gfp", "mcherry"]
    tracking_class = "delta"
    path_ref_h5 = path.joinpath("tracking_delta.h5")
    fca = FluoChangeAnalysis(path, channels, tracking_class)
    fca.gen_column_names()

    _, fca.labels_ref = fca.open_h5(path_ref_h5)
    fca.images_fluo = np.array([fca.open_h5(pf)[0] for pf in [path_ref_h5]])
    fca.images_fluo_raw = np.array([fca.open_h5(pf)[0] for pf in [path_ref_h5]])

    fca_output = fca.create_output_df(example_data_output)
    fca_output = fca.add_fluo_intensity(fca_output)

    # check if new columns were cerated
    assert all([c in fca_output.columns for c in fca.new_columns]) == True
    assert all([c in fca_output.columns for c in fca.new_columns_raw]) == True

    # check if columns contain right values
    val_df = fca_output[
        (fca_output.frame == 0) & (fca_output.labelID == 1)
    ].mean_norm_intensity_gfp.values[0]
    val_img = fca.images_fluo[0][0][fca.labels_ref[0] == 1].mean()

    assert val_df == val_img
