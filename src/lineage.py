import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial import distance

class Lineages:
    """
    A class to generate lineages based on trackinng outputs.

    ...

    Attributes
    ----------
    inputs : array
        input array for tracking network
    results : array
        output array of tracking network

    Methods
    -------
    generate_lineages()
        Generate lineages based on input and output arrays of tracking network.
    generate_global_IDs()
        Label cells with unique ID across all time frames.
    get_new_daughter_ID()
        Get local ID of daughter cell in following time frame.
    remove_global_ID()
        Remove global ID from list with all global IDs.
    """

    def __init__(self, inputs, results):
        """
        Parameters
        ----------
        inputs : array
            input array for tracking network
        results : array
            output array of tracking network
        """

        self.inputs = inputs
        self.results = results

    def generate_lineages(self):
        """Generates lineages based on output of tracking (U-Net) network.

        """

        # Generates global IDs for each frame and time step.
        self.generate_global_IDs()

        # Init dataframe and generate empty label stack
        self.init_dataframe()
        self.label_stack = np.empty((self.results.shape[:-1]))

        # Loop over all global IDs in list and connects cells between time frames
        # with help of tracking results
        num_time_steps = len(self.results)  # -1
        unique_ID = 0
        while len(self.global_IDs) > 0:

            # Picks first entry from global IDs list
            start_ID = self.global_IDs[0]

            # Init list to store frame numbers where cell is present
            frames = []

            # Gets first time frame where cell is present
            first_frame_cell = np.where(self.global_label == start_ID)[0][0]

            # Loop over all time frames
            for frame_cell in range(first_frame_cell, num_time_steps):

                # Check if track ID for current global ID was already assigned
                if self.track_output.isna().loc[start_ID,'trackID']:
                    unique_ID += 1
                else:
                    unique_ID = self.track_output.loc[start_ID,'trackID']

                # Store frame numbers where cell is present
                frames.append(frame_cell)

                # Get local ID from input
                local_ID = self.inputs[frame_cell, :, :, 1][self.global_label[frame_cell].astype(
                    int) == start_ID][0].astype(int)
                
                # Generate binary masks for mother and daughter cells
                mother = self.inputs[frame_cell, :, :, 1] == local_ID
                daughter_1 = self.results[frame_cell][:, :, 0] == local_ID
                daughter_2 = self.results[frame_cell][:, :, 1] == local_ID

                # Add mother with new (unique) ID to label stack
                self.label_stack[frame_cell][self.inputs[frame_cell,
                                                         :, :, 1] == local_ID] = unique_ID
                
                # Compute features for cell from binary mask
                cell = (self.global_label == start_ID)[frame_cell]
                area, edges, minor_axis_length, major_axis_length = self.get_features(cell)

                # Add data/features to DataFrame
                self.track_output.loc[start_ID, 'frame'] = frame_cell
                self.track_output.loc[start_ID, 'labelID'] = local_ID
                self.track_output.loc[start_ID, 'trackID'] = unique_ID
                self.track_output.loc[start_ID, 'area'] = area
                self.track_output.loc[start_ID, 'edges'] = edges
                self.track_output.loc[start_ID, 'minor_axis_length'] = minor_axis_length
                self.track_output.loc[start_ID, 'major_axis_length'] = major_axis_length
                self.track_output.loc[start_ID, 'frames'] = frames

                # For all cells which are not in the last time frame
                if frame_cell <= num_time_steps-2:

                    # Find global ID in next time frame to follow cell through time frames.
                    # Only in case cell still exists in next frame.

                    # Case 1: only daughter 1 is present
                    if daughter_1.sum() > 0 and daughter_2.sum() == 0:
                        new_global_ID = self.get_new_daughter_ID(
                            daughter_1, frame_cell)

                        # no cell split, daughter cell has same ID as mother cell
                        self.track_output.loc[start_ID, 'trackID_d1'] = unique_ID
                        self.track_output.loc[new_global_ID, 'trackID'] = unique_ID
                        self.track_output.loc[new_global_ID, 'trackID_mother'] = unique_ID

                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID

                    # Case 2: only daughter 2 is present
                    elif daughter_1.sum() == 0 and daughter_2.sum() > 0:
                        new_global_ID = self.get_new_daughter_ID(
                            daughter_2, frame_cell)

                        # no cell split, daughter cell has same ID as mother cell
                        self.track_output.loc[start_ID, 'trackID_d1'] = unique_ID
                        self.track_output.loc[new_global_ID, 'trackID'] = unique_ID
                        self.track_output.loc[new_global_ID, 'trackID_mother'] = unique_ID

                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID

                    # Case 3: cell split: both daughters are present
                    elif daughter_1.sum() > 0 and daughter_2.sum() > 0:
                        # get new ID for daughter one to continue tracking
                        new_global_ID_d1 = self.get_new_daughter_ID(
                            daughter_1, frame_cell)
                        new_global_ID_d2 = self.get_new_daughter_ID(
                            daughter_2, frame_cell)

                        # cell split, new IDs for both daughter cells
                        self.track_output.loc[start_ID, 'trackID_d1'] = unique_ID + 1
                        self.track_output.loc[start_ID, 'trackID_d2'] = unique_ID + 2

                        self.track_output.loc[new_global_ID_d1, 'trackID'] = unique_ID + 1
                        self.track_output.loc[new_global_ID_d2, 'trackID'] = unique_ID + 2

                        self.track_output.loc[new_global_ID_d1, 'trackID_mother'] = unique_ID
                        self.track_output.loc[new_global_ID_d2, 'trackID_mother'] = unique_ID

                        # new_global_ID_d2 will stay in self.global_IDs and lineage generation is
                        # continued at later time point.
                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID_d1

                    # case 4: cell disappears
                    elif daughter_1.sum() == 0 and daughter_2.sum() == 0:
                        self.remove_global_ID(start_ID)
                        break

                # Lineage generation is stopped in last time frame
                if frame_cell > num_time_steps-2:
                    self.remove_global_ID(start_ID)

    def generate_global_IDs(self):
        """Generates global IDs with one ID per cell and time frame and
        list with all global IDs.

        """

        self.global_label = np.empty(self.inputs[:, :, :, 1].shape)

        # Loops through all time frames and adding the maximal cell ID of the previous time frame to the
        # IDs/labels of the current time frame. By that every cell in every time frame gets one global ID.
        max_val = 0
        for i, inp in enumerate(self.inputs[:, :, :, 1]):
            new_label = inp + max_val
            new_label[inp == 0] = 0
            self.global_label[i] = new_label
            max_val = np.max(self.global_label[i])

        self.global_IDs = list(np.unique(self.global_label)[1:].astype(int))

    def init_dataframe(self):
        """Initialize dataframe for tracking output.

        """

        columns = ['frame', 'labelID', 'trackID', 'lineageID', 'trackID_d1', 'trackID_d2', 
                    'trackID_mother', 'area', 'edges', 'minor_axis_length', 'major_axis_length',
                    'frames']
        self.track_output = pd.DataFrame(columns=columns, index=self.global_IDs)

    def get_new_daughter_ID(self, daughter, frame_cell):
        """Extracts global ID of daughter cell in next time frame.

        Parameters
        ----------
        daughter: array
            Segmentation image of daughter cell generated by the U-Net.

        frame_cell: int
            Number of the current time frame. 
        """

        # get center of daughter in current frame
        daughter_cent = regionprops(daughter.astype(int))[0].centroid

        # get center of single cells in next frame
        seg_next_frame = self.inputs[frame_cell+1, :, :, 1]
        segs_next_frame_cents = [
            r.centroid for r in regionprops(seg_next_frame.astype(int))]
        min_dist = np.argmin([distance.euclidean(daughter_cent, c)
                             for c in segs_next_frame_cents])

        # get local ID in next frame from index => + 1
        new_local_ID = min_dist + 1

        # get global ID in next frame
        new_global_ID = self.global_label[frame_cell+1][self.inputs[(
            frame_cell+1), :, :, 1] == new_local_ID][0].astype(int)

        return new_global_ID

    def get_features(self, cell):
        """Extracts features for each cell.

        Parameters
        ----------
        daughter: array
            Segmentation image of daughter cell generated by the U-Net.
        """

        # get center of daughter in current frame
        area = regionprops(cell.astype(int))[0].area
        edges = regionprops(cell.astype(int))[0].bbox
        minor_axis_length = regionprops(cell.astype(int))[0].minor_axis_length
        major_axis_length = regionprops(cell.astype(int))[0].major_axis_length

        return area, edges, minor_axis_length, major_axis_length

    def remove_global_ID(self, current_ID):
        """Removes global ID from list of all global IDs during the lineage generation.

        Parameters
        ----------
        current_ID: int
            ID which is currently processed during tracking.
        """

        # remove current ID from global IDs
        try:
            self.global_IDs.remove(current_ID)
        except ValueError:
            None
