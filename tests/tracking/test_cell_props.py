import tempfile
from pathlib import Path

import pytest

from midap.tracking.cell_props import CellProps


# Fixtures
##########

@pytest.fixture()
def tmpdir():
    """
    Sets up a tmpdir and returns the path as pathlib Path
    :return: The name of the tmpdir
    """

    # setup
    directory = tempfile.TemporaryDirectory()

    # return
    yield Path(directory.name)

    # cleanup
    directory.cleanup()


# Tests
#######

def test_cell_props(tmpdir):
    """
    Tests the CellProps init, the functionality is tested in the app tests
    :param tmpdir: A fixture providing a temporary directory
    """

    # the inputs
    csv_file = tmpdir.joinpath("csv_file.csv")
    data_file = tmpdir.joinpath("data_file.h5")

    # all possibilities
    with pytest.raises(FileNotFoundError):
        _ = CellProps(csv_file=csv_file, data_file=data_file)
    csv_file.touch()
    with pytest.raises(FileNotFoundError):
        _ = CellProps(csv_file=csv_file, data_file=data_file)
    data_file.touch()
    _ = CellProps(csv_file=csv_file, data_file=data_file)
