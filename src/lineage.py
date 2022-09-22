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

        # Generate empty label stack and dictionary to map global to unique IDs
        self.init_dataframe()
        self.label_stack = np.empty((self.results.shape[:-1]))
        self.global2uniqueID = {}
        self.label_dict = []
        self.tracks_data = []
        self.graph = {}
        

        # Loops over all global IDs in list and connects cells between time frames
        # with help of tracking results
        num_time_steps = len(self.results)  # -1
        unique_ID = 0
        while len(self.global_IDs) > 0:
            print(len(self.global_IDs))

            # Picks first entry from global IDs list
            start_ID = self.global_IDs[0]

            # Per cell all relevant information are stored in dictionary and collected in label_dict for all cells
            unique_ID += 1
            cell_dict = {}
            cell_dict['ID'] = unique_ID
            cell_dict['frames'] = []
            cell_dict['mother'] = []
            cell_dict['daughters'] = []
            cell_dict['split'] = []

            # Gets first time frame where cell is present
            first_frame_cell = np.where(self.global_label == start_ID)[0][0]

            # Loops over all time frames
            for frame_cell in range(first_frame_cell, num_time_steps):

                # # Get centroid of current cell / needed?
                # cell = (self.global_label == start_ID)[frame_cell] #binary mask for current cell
                # res = regionprops(label(cell))
                # coord = res[0].centroid
                
                # # Add data to tracks data
                # # needed?
                # self.tracks_data.append(
                #     [unique_ID, frame_cell, int(coord[1]), int(coord[0])])
                # # needed?
                cell_dict['frames'].append(frame_cell)

                # Dict to map unique_ID to global_ID
                self.global2uniqueID[start_ID] = unique_ID

                # Get local ID within inputs time frame 
                local_ID = self.inputs[frame_cell, :, :, 1][self.global_label[frame_cell].astype(
                    int) == start_ID][0].astype(int)
                
                # generate binary masks for mother and daughter cells
                mother = self.inputs[frame_cell, :, :, 1] == local_ID
                daughter_1 = self.results[frame_cell][:, :, 0] == local_ID
                daughter_2 = self.results[frame_cell][:, :, 1] == local_ID

                # Add mother with new (unique) ID to label stack / neeeded?
                self.label_stack[frame_cell][self.inputs[frame_cell,
                                                         :, :, 1] == local_ID] = unique_ID
                
                # Add data to DataFrame
                self.track_output.loc[start_ID, 'frame'] = frame_cell
                self.track_output.loc[start_ID, 'labelID'] = local_ID
                self.track_output.loc[start_ID, 'trackID'] = unique_ID

                # For all cells which are not in the last time frame
                if frame_cell <= num_time_steps-2:

                    # Find global ID in next time frame to follow cell through time frames.
                    # Only in case cell still exists in next frame.

                    # Case 1: only daughter 1 is present
                    if daughter_1.sum() > 0 and daughter_2.sum() == 0:
                        new_global_ID = self.get_new_daughter_ID(
                            daughter_1, frame_cell)

                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID

                        if new_global_ID not in self.graph.keys():
                            self.graph[new_global_ID] = [unique_ID]

                        cell_dict['daughters'].append([new_global_ID])

                    # Case 2: only daughter 2 is present
                    elif daughter_1.sum() == 0 and daughter_2.sum() > 0:
                        new_global_ID = self.get_new_daughter_ID(
                            daughter_2, frame_cell)

                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID

                        if new_global_ID not in self.graph.keys():
                            self.graph[new_global_ID] = [unique_ID]

                        cell_dict['daughters'].append([new_global_ID])

                    # Case 3: cell split: both daughters are present
                    elif daughter_1.sum() > 0 and daughter_2.sum() > 0:
                        # get new ID for daughter one to continue tracking
                        new_global_ID_d1 = self.get_new_daughter_ID(
                            daughter_1, frame_cell)
                        new_global_ID_d2 = self.get_new_daughter_ID(
                            daughter_2, frame_cell)

                        if new_global_ID_d1 not in self.graph.keys():
                            self.graph[new_global_ID_d1] = [unique_ID]
                        if new_global_ID_d2 not in self.graph.keys():
                            self.graph[new_global_ID_d2] = [unique_ID]

                        cell_dict['split'].append(frame_cell)
                        cell_dict['daughters'].append(
                            [new_global_ID_d1, new_global_ID_d2])

                        # new_global_ID_d2 will stay in self.global_IDs and lineage generation is
                        # continued at later time point.
                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID_d1
                        unique_ID += 1 #after cell split both daughter cell get their own ID

                    # case 4: cell disappears
                    elif daughter_1.sum() == 0 and daughter_2.sum() == 0:
                        self.remove_global_ID(start_ID)

                        cell_dict['daughters'].append([])
                        break

                # Lineage generation is stopped in last time frame
                if frame_cell > num_time_steps-2:
                    self.remove_global_ID(start_ID)

            self.label_dict.append(cell_dict)

        # convert cell IDs in label dict from global to unique IDs
        for ix, cell in enumerate(self.label_dict):
            ids = []
            for d in cell['daughters']:
                ids.append([self.global2uniqueID[i] for i in d])
            self.label_dict[ix]['daughters'] = ids

        self.graph_unique = {}
        for k in self.graph.keys():
            new_key = self.global2uniqueID[k]
            self.graph_unique[new_key] = self.graph[k]

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

        columns = ['frame', 'labelID', 'trackID']
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

    def get_features(self, daughter):
        """Extracts features for each cell.

        Parameters
        ----------
        daughter: array
            Segmentation image of daughter cell generated by the U-Net.
        """

        # get center of daughter in current frame
        area = regionprops(daughter.astype(int))[0].area
        edges = regionprops(daughter.astype(int))[0].bbox

        return area, edges

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
