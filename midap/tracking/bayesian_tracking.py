import btrack
from btrack import datasets
from btrack.constants import BayesianUpdates

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import distance

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

from .base_tracking import Tracking

class BayesianCellTracking(Tracking):
    """
    A class for cell tracking using Bayesian tracking
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DeltaV2Tracking using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)


    def set_params(self):
        """
        Sets the parameters needed for the Bayesian tracking.
        """

        self.features = ["area", 
                            "major_axis_length", 
                            "minor_axis_length", 
                            "orientation",
                            "intensity_mean",
                            "intensity_min",
                            "intensity_max",
                            "minor_axis_length",
                            "major_axis_length",
                            "coords",]

        self.objects = btrack.utils.segmentation_to_objects(
            segmentation = self.seg_imgs, 
            intensity_image = self.raw_imgs,
            properties=tuple(self.features), 
        )

        self.config_file = datasets.cell_config()


    def track_all_frames(self, *args, **kwargs):
        """
        Tracks all frames and converts output to standard format.
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """
        self.run_model()
        self.convert_data()
        self.generate_label_stack()
        self.correct_label_stack()
        self.store_lineages(*args, **kwargs)


    def extract_data(self):
        """
        Extracts input data needed for Bayesian tracking.
        """
        self.seg_imgs = np.array([label(self.load_data(cur_frame)[2]) for cur_frame in range(1, self.num_time_steps)])
        self.raw_imgs = np.array([self.load_data(cur_frame)[0] for cur_frame in range(1, self.num_time_steps)])

    
    def run_model(self):
        """
        Run Bayesian model.
        """
        self.extract_data()
        self.set_params()

        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as self.tracker:

            self.tracker.update_method = BayesianUpdates.EXACT

            # configure the tracker using a config file
            self.tracker.configure_from_file(self.config_file)

            # set params
            self.tracker.max_search_radius = 100
            self.tracker.features = self.features
            self.tracker.tracking_updates = ["VISUAL","MOTION"]

            # append the objects to be tracked
            self.tracker.append(self.objects)

            # track them (in interactive mode)
            self.tracker.track(step_size=100)

            # generate hypotheses and run the global optimizer
            self.tracker.optimize()

            # get the tracks as a python list
            self.tracks = self.tracker.tracks


    def generate_label_stack(self):
        """
        Generate label stack based on tracking output.
        """

        self.label_stack = np.zeros(self.seg_imgs.shape)
        for tr in self.tracks:
            for i, t in enumerate(tr['t']):
                try:
                    self.label_stack[t][tr['coords'][i][:,0],tr['coords'][i][:,1]] = tr['ID']
                    
                except IndexError:
                    None


    def __find_nearest_neighbour(self, point: tuple, seg: np.ndarray):
        """
        Find nearest neighboring cell in segmentation image.
        :point: Center point of tracked cell
        :seg: Segmentation image
        """

        centroids = [r.centroid for r in regionprops(seg)]
        labels = [r.label for r in regionprops(seg)]
        ix_min = np.argsort([distance.euclidean(c, point) for c in centroids])[1]
        return ix_min


    def correct_label_stack(self):
        """
        Correct label_stack and track_output to fit to community standard:
        - new label for mother after cell split
        - add IDs of daughter cells
        """

        self.label_stack_correct = self.label_stack 
        self.track_output_correct = self.track_output.copy()

        self.track_output_correct['trackID_d1'] = self.track_output_correct['trackID']
        self.track_output_correct['trackID_d2'] = self.track_output_correct['trackID']
        self.track_output_correct['trackID_mother'] = self.track_output_correct['trackID']

        for t in range(1,len(self.label_stack)):

            labels_prev_frame = np.unique(self.label_stack[t-1])
            labels_cur_frame = np.unique(self.label_stack[t])

            # find new IDs
            diff_ix = np.array([l2 not in labels_prev_frame for l2 in labels_cur_frame])

            # find labels and centroids of new cells
            reg = regionprops((self.label_stack[t]).astype(int))
            centroids = [r.centroid for r in reg]
            labels = [r.label for r in reg]
            new_cells = labels_cur_frame[diff_ix]

            # find closest cell and do correction of label stack
            for c in new_cells:
                ix = np.where(labels == c)[0][0]
                ix_closest_cell = self.__find_nearest_neighbour(centroids[ix], self.label_stack[t].astype(int))
                new_ID_d1 = int(labels[ix])
                new_ID_d2 = int(self.__new_ID())
                mother = labels[ix_closest_cell]

                self.label_stack_correct[t:][self.label_stack[t:] == mother] = new_ID_d2

                # correct df
                ix_col_ID = np.where(self.track_output_correct.columns == 'trackID')[0][0]
                ix_col_mother = np.where(self.track_output_correct.columns == 'trackID_mother')[0][0]
                ix_col_ID_d1 = np.where(self.track_output_correct.columns == 'trackID_d1')[0][0]
                ix_col_ID_d2 = np.where(self.track_output_correct.columns == 'trackID_d2')[0][0]

                for t_tmp_1 in range(0,t):

                    filter_t = self.track_output_correct['frame']==t_tmp_1
                    filter_ID = self.track_output_correct['trackID']==mother

                    try:
                        ix_cell = np.where(filter_t&filter_ID)[0][0]
                        self.track_output_correct.iloc[ix_cell,ix_col_ID_d1] = new_ID_d1
                        self.track_output_correct.iloc[ix_cell,ix_col_ID_d2] = new_ID_d2

                    except IndexError: #if cell skips frame
                        None

                max_t = self.track_output_correct[self.track_output_correct['trackID'] == new_ID_d1]['frame'].max()
                for t_tmp_2 in range(t,max_t):
                    
                    filter_t = self.track_output_correct['frame']==t_tmp_2
                    filter_ID = self.track_output_correct['trackID']==mother
                    filter_ID_d1 = self.track_output_correct['trackID']==new_ID_d1

                    try:
                        ix_d1 = np.where(filter_t&filter_ID_d1)[0][0]
                        self.track_output_correct.iat[ix_d1,ix_col_mother] = mother

                        ix_cell = np.where(filter_t&filter_ID)[0][0]
                        self.track_output_correct.iat[ix_cell,ix_col_ID] = new_ID_d2
                    except IndexError: #if cell skips frame
                        None


    def __new_ID(self):
        return np.max(self.label_stack_correct)+1


    def convert_data(self):
        """
        Convert tracking output into dataframe in standard format.
        """

        # generate subset of dict
        keys_to_extract = ['t', 'ID', 'x', 'y', 'area', 
                            'intensity_mean', 'intensity_min', 'intensity_max',
                            'minor_axis_length', 'major_axis_length']
        cells = []
        for cell in self.tracks:
            tmp_dict = cell.to_dict()
            tmp_dict['first_frame'] = tmp_dict['t'][0]
            tmp_dict['last_frame'] = tmp_dict['t'][-1]
            cell_dict = {key: tmp_dict[key] for key in keys_to_extract}
            
            cells.append(cell_dict)

        # transform subset into df
        self.track_output = pd.DataFrame(cells[0])
        for c in cells[1:]:
            df_cells_old = self.track_output
            df_cells_new = pd.DataFrame(c)
            self.track_output = pd.concat([df_cells_old, df_cells_new])
            
        self.track_output.sort_values(by='t', inplace=True)
        self.track_output.rename(columns = {'t':'frame', 'ID':'trackID'}, inplace = True)


    def store_lineages(self, logger, output_folder: str):
        """
        Store tracking output files: labeled stack, tracking output, input files.
        :logger: 
        :output_folder: Folder where to store the data
        """

        self.track_output_correct.to_csv(output_folder + '/track_output_bayesian.csv', index=True)

        hf = h5py.File(output_folder + '/raw_inputs_bayesian.h5', 'w')
        raw_inputs = self.raw_imgs
        hf.create_dataset('raw_inputs', data=raw_inputs)
        hf.close()

        hf = h5py.File(output_folder + '/segmentations_bayesian.h5', 'w')
        segs = self.seg_imgs
        hf.create_dataset('segmentations', data=segs)
        hf.close()

        hf = h5py.File(output_folder + '/label_stack_bayesian.h5', 'w')
        hf.create_dataset('label_stack', data=self.label_stack_correct)
        hf.close()