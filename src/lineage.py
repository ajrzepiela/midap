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
        self.__generate_global_IDs()

        # Init dataframe and generate empty label stack
        self.__init_dataframe()
        self.label_stack = np.zeros((self.results.shape[:-1]))

        # Loop over all global IDs in list and connects cells between time frames
        # with help of tracking results
        num_time_steps = len(self.results)
        self.trackID_list = np.arange(1, np.max(self.global_IDs)).tolist()

        while len(self.global_IDs) > 0:
            # Gets first time frame where cell is present
            start_ID = self.global_IDs[0] # globalID of currently tracked cell
            first_frame_cell = np.where(self.global_label == start_ID)[0][0]

            # Init list to store frame numbers where cell is present
            frames = []
            for frame_cell in range(first_frame_cell, num_time_steps):
                # Check if track ID for current global ID was already assigned
                if self.track_output.isna().loc[start_ID,'trackID']:
                    trackID = self.__set_new_trackID()
                else:
                    trackID = self.track_output.loc[start_ID,'trackID']

                # Get localID from input for current start ID/global ID
                local_ID = self.__get_localID(frame_cell, start_ID)

                # Add mother with new (unique) ID to label stack
                self.label_stack[frame_cell][self.inputs[frame_cell,
                                                         :, :, 1] == local_ID] = trackID
                
                # Compute features/IDs for cell
                frames.append(frame_cell)
                if len(frames) == 1:
                    lineage_ID = trackID
                cell_props = \
                    self.__get_features(frame_cell, start_ID)

                # Add data/features to DataFrame
                self.track_output.loc[start_ID, 'frame'] = frame_cell
                self.track_output.loc[start_ID, 'labelID'] = local_ID
                self.track_output.loc[start_ID, 'trackID'] = trackID
                self.track_output.loc[start_ID, 'lineageID'] = lineage_ID
                self.track_output.loc[start_ID, 'area'] = cell_props.area
                self.track_output.loc[start_ID, 'edges_min_row'] = cell_props.bbox[0]
                self.track_output.loc[start_ID, 'edges_min_col'] = cell_props.bbox[1]
                self.track_output.loc[start_ID, 'edges_max_row'] = cell_props.bbox[2]
                self.track_output.loc[start_ID, 'edges_max_col'] = cell_props.bbox[3]
                self.track_output.loc[start_ID, 'intensity_max'] = cell_props.intensity_max
                self.track_output.loc[start_ID, 'intensity_mean'] = cell_props.intensity_mean
                self.track_output.loc[start_ID, 'intensity_min'] = cell_props.intensity_min
                self.track_output.loc[start_ID, 'minor_axis_length'] = cell_props.minor_axis_length
                self.track_output.loc[start_ID, 'major_axis_length'] = cell_props.major_axis_length
                self.track_output.loc[start_ID, 'frames'] = frames

                # Generate binary masks for mother and daughter cells
                daughter_1 = self.results[frame_cell][:, :, 0] == local_ID
                daughter_2 = self.results[frame_cell][:, :, 1] == local_ID

                # For all cells which are not in the last time frame
                if frame_cell <= num_time_steps - 2:

                    # Find global ID in next time frame to follow cell through time frames.
                    # Only in case cell still exists in next frame.

                    # Case 1: only daughter 1 is present
                    if daughter_1.sum() > 0 and daughter_2.sum() == 0:
                        new_global_ID = self.__get_new_daughter_ID(
                            daughter_1, frame_cell)

                        # update of df and set new globalID for next time-frame
                        self.__update_df(new_global_ID, start_ID, trackID)
                        self.__remove_global_ID(start_ID)
                        start_ID = new_global_ID

                    # Case 2: only daughter 2 is present
                    elif daughter_1.sum() == 0 and daughter_2.sum() > 0:
                        new_global_ID = self.__get_new_daughter_ID(
                            daughter_2, frame_cell)

                        # update of df and set new globalID for next time-frame
                        self.__update_df(new_global_ID, start_ID, trackID)
                        self.__remove_global_ID(start_ID)
                        start_ID = new_global_ID

                    # Case 3: cell split: both daughters are present
                    elif daughter_1.sum() > 0 and daughter_2.sum() > 0:
                        # get new ID for daughter one to continue tracking
                        new_global_ID_d1 = self.__get_new_daughter_ID(
                            daughter_1, frame_cell)
                        new_global_ID_d2 = self.__get_new_daughter_ID(
                            daughter_2, frame_cell)

                        # cell split, new IDs for both daughter cells
                        trackID_d1 = self.__set_new_trackID()
                        trackID_d2 = self.__set_new_trackID()

                        self.__update_df(new_global_ID_d1, start_ID, \
                            trackID, trackID_d=trackID_d1)
                        self.__update_df(new_global_ID_d2, start_ID, \
                            trackID, trackID_d=trackID_d2)

                        # Add trackID_mother, trackID_d1 and trackID_d2 to 
                        # preceding cells with same trackID
                        for f in frames:
                            ix_prev_cell = np.where((self.track_output.frame == f) & \
                                (self.track_output.trackID == trackID))[0][0] + 1
                            self.track_output.loc[ix_prev_cell, 'trackID_d1'] = trackID_d1
                            self.track_output.loc[ix_prev_cell, 'trackID_d2'] = trackID_d2
                            if frames[0] > 0:
                                self.track_output.loc[ix_prev_cell, 'trackID_mother'] = trackID

                        # new_global_ID_d2 will stay in self.global_IDs and lineage generation is
                        # continued at later time point.
                        self.__remove_global_ID(start_ID)
                        start_ID = new_global_ID_d1
                        break

                    # case 4: cell disappears
                    elif daughter_1.sum() == 0 and daughter_2.sum() == 0:
                        self.track_output.loc[start_ID, 'split'] = 0
                        self.__remove_global_ID(start_ID)
                        break

                # Empty list to stop lineage generation
                if frame_cell > num_time_steps-2:
                    self.__remove_global_ID(start_ID)
        
        # Replace list of frames with first and last frame
        for i in self.track_output.index:
            self.track_output.loc[i, 'first_frame'] = self.track_output.loc[i, 'frames'][0]
            self.track_output.loc[i, 'last_frame'] = self.track_output.loc[i, 'frames'][-1]
        self.track_output.drop(['frames'],axis=1, inplace=True)


    def __generate_global_IDs(self):
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

    def __init_dataframe(self):
        """Initialize dataframe for tracking output.

        """

        columns = ['frame', 'labelID', 'trackID', 'lineageID', 'trackID_d1', 'trackID_d2', 'split',
                    'trackID_mother', 'area', 'edges_min_row', 'edges_min_col', 'edges_max_row', 
                    'edges_max_col', 'intensity_max', 'intensity_mean', 'intensity_min', 
                    'minor_axis_length', 'major_axis_length', 'frames',
                    'first_frame', 'last_frame']
        self.track_output = pd.DataFrame(columns=columns, index=self.global_IDs)

    def __get_localID(self, frame_cell, start_ID):
        """Get localID in current time-frame given the startID.

        Parameters
        ----------
        frame_cell: int
            Number of the current time frame.

        start_ID: int
            GlobalID which is currently processed. 
        """

        return self.inputs[frame_cell, :, :, 1][self.global_label[frame_cell].astype(
                    int) == start_ID][0].astype(int)

    def __get_new_daughter_ID(self, daughter, frame_cell):
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

    def __get_features(self, frame_cell, start_ID):
        """Extracts features for each cell.

        Parameters
        ----------
        frame_cell: int
            Number of the current time frame.

        start_ID: int
            GlobalID which is currently processed. 
        """

        # get center of daughter in current frame
        cell = (self.global_label == start_ID)[frame_cell]
        cell_props = regionprops(cell.astype(int), intensity_image = self.inputs[frame_cell,:, :, 0])[0]

        return cell_props

    def __update_df(self, new_global_ID, start_ID, trackID, trackID_d=None):
        """Update track output.

        Parameters
        ----------
        new_global_ID: int
            global_ID in next tim-fram.

        start_ID: int
            globalID which is currently processed. 
        
        trackID: int
            Unique trackID of current cell.

        trackID_d: int
            Unique trackID of daughter cell.
        """
        if not trackID_d:
            self.track_output.loc[new_global_ID, 'trackID'] = trackID
            self.track_output.loc[start_ID, 'split'] = 0

        elif trackID_d:
            self.track_output.loc[start_ID, 'trackID_d1'] = trackID_d
            self.track_output.loc[new_global_ID, 'trackID'] = trackID_d
            self.track_output.loc[new_global_ID, 'trackID_mother'] = trackID
            self.track_output.loc[start_ID, 'split'] = 1

    def __remove_global_ID(self, current_ID):
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

    def __set_new_trackID(self):
        """Removes current trackID from list of all trackIDs and sets 
        trackID to next item from list.
        """

        new_trackID = self.trackID_list[0]
        self.trackID_list.remove(new_trackID)
        return new_trackID
