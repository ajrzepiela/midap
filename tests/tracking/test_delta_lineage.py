import numpy as np
import pandas as pd
import tempfile
import os

from midap.tracking.lineage import Lineages
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

    # run the tracking
    tracking_instance.track_all_frames_crop()

    # prep results
    res_shape = (len(tracking_instance.results_all), ) + tracking_instance.results_all[0].shape[1:3] + (2, )
    results_all_red = np.zeros(res_shape)

    for t in range(len(tracking_instance.results_all)):
        for ix, cell_id in enumerate(tracking_instance.results_all[t]):
            if cell_id[:, :, 0].sum() > 0:
                results_all_red[t, cell_id[:, :, 0] > 0, 0] = ix + 1
            if cell_id[:, :, 1].sum() > 0:
                results_all_red[t, cell_id[:, :, 1] > 0, 1] = ix + 1

    # create the instance
    lin = Lineages(np.array(tracking_instance.inputs_all), results_all_red)

    # creat lineages
    lin.generate_lineages()

    # A temp directory to save everything
    tmp_dir = tempfile.TemporaryDirectory()
    out_file = os.path.join(tmp_dir.name, 'track_output.csv')

    # save to file
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
    lin = Lineages(inputs_all, results_all_red)

    # creat lineages
    lin.generate_lineages()

    # A temp directory to save everything
    tmp_dir = tempfile.TemporaryDirectory()
    out_file = os.path.join(tmp_dir.name, 'track_output.csv')

    # save to file
    lin.track_output.to_csv(out_file)

    yield out_file

    # cleanup
    tmp_dir.cleanup()

# Tests
#######


def test_global_labels(example_data_output):
    """
    Test if sum of local IDs is equal to max of global IDs.
    :param example_data_output: The fixture creating an output of lineages in a temporary directory
    """

    track_output = pd.read_csv(example_data_output, index_col='Unnamed: 0')
    sum_local_ID = sum([track_output[track_output['frame'] == f]
                        ['labelID'].max() for f in pd.unique(track_output['frame'])])
    max_global_ID = track_output.index.max()

    assert sum_local_ID == max_global_ID

def test_ID_assigmment(example_data_output):
    """
    Test assignment of mother and daughter IDs.
    :param example_data_output: The fixture creating an output of lineages in a temporary directory
    """

    track_output = pd.read_csv(example_data_output, index_col='Unnamed: 0')

    ix_split = np.where(track_output.split == 1)[0][0] + 1
    trackID = track_output.loc[ix_split].trackID
    frame = track_output.loc[ix_split].frame
    trackID_d1 = track_output.loc[ix_split].trackID_d1
    
    trackID_mother = track_output.loc[np.where((track_output.frame == frame + 1) & \
                        (track_output.trackID == trackID_d1))[0][0] + 1].trackID_mother

    assert trackID == trackID_mother

def test_fake_lineage(fake_data_output):
    """
    Tests the lineage output of the fake data
    :param fake_data_output: The path to the output CSV generated with fake data from the deltav2 tests
    """

    df = pd.read_csv(fake_data_output)

    # TODO: Implement this test, currently there is no splitting event because the last frame is ignored!
