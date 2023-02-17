import numpy as np
import pandas as pd
import tempfile
import os

from midap.tracking.delta_lineage import DeltaTypeLineages
from pytest import fixture, mark
from pathlib import Path

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
    #results_all_red = tracking_instance.reduce_data(output_folder=tmp_dir.name, inputs=inputs, results=results_all)

    # create the instance
    lin = DeltaTypeLineages(np.array(inputs), results_all)

    # save to file
    out_file = os.path.join(tmp_dir.name, 'track_output_delta.csv')
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
    inputs_all = data_inp['inputs_all']

    # create the instance
    lin = DeltaTypeLineages(inputs_all, results_all_red)

    # creat lineages
    lin.generate_lineages()

    # A temp directory to save everything
    tmp_dir = tempfile.TemporaryDirectory()
    out_file = os.path.join(tmp_dir.name, 'track_output_delta.csv')

    # save to file
    lin.track_output.to_csv(out_file)

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

    track_output = pd.read_csv(example_data_output, index_col='Unnamed: 0')

    ix_split = track_output.index[track_output.split == 1][0]
    trackID = track_output.loc[ix_split].trackID
    frame = track_output.loc[ix_split].frame
    trackID_d1 = track_output.loc[ix_split].trackID_d1
    trackID_d2 = track_output.loc[ix_split].trackID_d2

    filter_d1 = track_output.index[(track_output.frame == frame + 1) & (track_output.trackID == trackID_d1)][0]
    trackID_mother_d1 = track_output.loc[filter_d1].trackID_mother

    assert trackID == trackID_mother_d1

    filter_d2 = track_output.index[(track_output.frame == frame + 1) & (track_output.trackID == trackID_d2)][0]
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
