import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial import distance


class Lineages:
    """
    A class to generate lienages based on trackinng outputs.

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
        self.generate_global_IDs()

        self.label_stack = np.empty((self.results.shape[:-1]))
        num_time_steps = len(self.results)#-1

        unique_ID = 0
        #daughter_IDs = []
        while len(self.global_IDs) > 0:
            print(len(self.global_IDs))
            start_ID = self.global_IDs[0]
            unique_ID += 1
            
            # get first time frame where cell is present
            first_frame_cell = np.where(self.global_label == start_ID)[0][0]

            for frame_cell in range(first_frame_cell,num_time_steps):
                
                # get local ID within time frame
                local_ID = self.inputs[frame_cell,:,:,1][self.global_label[frame_cell].astype(int) == start_ID][0].astype(int)

                mother = self.inputs[frame_cell,:,:,1] == local_ID
                daughter_1 = self.results[frame_cell][:,:,0] == local_ID
                daughter_2 = self.results[frame_cell][:,:,1] == local_ID

                 # add new ID to label stack
                self.label_stack[frame_cell][self.inputs[frame_cell,:,:,1] == local_ID] = unique_ID

                # for all cells which are not in the last time frame
                if frame_cell <= num_time_steps-2:
            
                    # find local IDs in next time frame if cell still exists in next frame

                    # case 1: olny daughter 1 is present
                    if daughter_1.sum() > 0 and daughter_2.sum() == 0:
                        new_global_ID = self.get_new_daughter_ID(daughter_1, frame_cell)

                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID

                    # case 2: only daughter 2 is present
                    elif daughter_1.sum() == 0 and daughter_2.sum() > 0:
                        new_global_ID = self.get_new_daughter_ID(daughter_2, frame_cell)

                        self.remove_global_ID(start_ID)
                        start_ID = new_global_ID

                    # case 3: cell split: both daughters are present
                    elif daughter_1.sum() > 0 and daughter_2.sum() > 0:
                        # get new ID for daughter one to continue tracking
                        new_global_ID_d1 = self.get_new_daughter_ID(daughter_1, frame_cell)
                        #new_global_ID_d2 = get_new_daughter_ID(daughter_2, inputs_all, global_label, frame_cell)

                        self.remove_global_ID(start_ID)

                        start_ID = new_global_ID_d1

        #                break

                    # case 4: cell disappears
                    elif daughter_1.sum() == 0 and daughter_2.sum() == 0:
                        self.remove_global_ID(start_ID)
                        break
        
                if frame_cell > num_time_steps-2:
                    self.remove_global_ID(start_ID)
        #            break


    def generate_global_IDs(self):
        self.global_label = np.empty(self.inputs[:,:,:,1].shape)
        
        max_val = 0
        for i, inp in enumerate(self.inputs[:,:,:,1]):
            new_label = inp + max_val
            new_label[inp == 0] = 0
            self.global_label[i] = new_label
            max_val = np.max(self.global_label[i])

        self.global_IDs = list(np.unique(self.global_label)[1:].astype(int))


    def get_new_daughter_ID(self, daughter, frame_cell):
        # get center of daughter in current frame
        daughter_cent = regionprops(daughter.astype(int))[0].centroid

        #get center of single cells in next frame
        seg_next_frame = self.inputs[frame_cell+1,:,:,1]
        segs_next_frame_cents = [r.centroid for r in regionprops(seg_next_frame.astype(int))]
        min_dist = np.argmin([distance.euclidean(daughter_cent,c) for c in segs_next_frame_cents])

        # get local ID in next frame from index => + 1
        new_local_ID = min_dist + 1

        # get global ID in next frame
        new_global_ID = self.global_label[frame_cell+1][self.inputs[(frame_cell+1),:,:,1] == new_local_ID][0].astype(int)

        return new_global_ID


    def remove_global_ID(self, start_ID):
        # remove start ID from global IDs
        try:
            self.global_IDs.remove(start_ID)
        except ValueError:
            None
