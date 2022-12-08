import btrack
from btrack import datasets
from btrack.constants import BayesianUpdates

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import distance

from skimage.measure import label, regionprops

#from .model_tracking_bayesian import BayesianCellTracking
from .base_tracking import Tracking

class BayesianCellTracking(Tracking):
    """
    A class for cell tracking using Bayesian tracking
    """

    def __init__(self, *args, **kwargs): #input_type, output_type, 
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
                            "coords"]

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
        # inputs = np.array([self.load_data(cur_frame) for cur_frame in range(1, self.num_time_steps)])
        # #self.seg_imgs = np.array([label(self.load_data(cur_frame)[2]) for cur_frame in range(1, self.num_time_steps)])
        self.seg_imgs = np.array([self.load_data(cur_frame)[2] for cur_frame in range(1, self.num_time_steps)])
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
            self.tracker.verbose = True
            self.tracker.max_search_radius = 200
            self.tracker.features = self.features

            self.tracker.tracking_updates = ["VISUAL"]

            # append the objects to be tracked
            self.tracker.append(self.objects)

            # set the tracking volume
            #self.tracker.volume=((0, 512), (0, 512))

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
                    # point = (int(tr['x'][i]), int(tr['y'][i]))
                    # ix_cell = self.__find_nearest_neighbour(point, self.seg_imgs[t])
                    # self.label_stack[t][ix_cell] = tr['ID']
                    # print(tr['ID'], i)


    def __find_nearest_neighbour(self, point: tuple, seg: np.ndarray):
        """
        Find nearest neighboring cell in segmentation image.
        :point: Center point of tracked cell
        :seg: Segmentation image
        """

        centroids = [r.centroid for r in regionprops(seg)]
        labels = [r.label for r in regionprops(seg)]
        ix_min = np.argsort([distance.euclidean(c, point) for c in centroids])[1]
        ix_cell = np.where(seg == labels[ix_min])
        return ix_min


    def correct_label_stack(self):

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
            self.label_stack_correct = self.label_stack 
            self.track_output_correct = self.track_output.copy()
            for c in new_cells:
                ix = np.where(labels == c)[0][0]
                ix_closest_cell = self.__find_nearest_neighbour(centroids[ix], self.label_stack[t].astype(int))
                new_ID_d1 = labels[ix]
                new_ID_d2 = self.__new_ID()
                old_ID_d1 = labels[ix_closest_cell]

                self.label_stack_correct[t:][self.label_stack[t:] == old_ID_d1] = new_ID_d2

                # import pdb
                # pdb.set_trace()

                # update df
                for t_tmp_1 in range(1,t):
                    self.track_output.at[t_tmp_1,'trackID_d1'][self.track_output.at[t_tmp_1,'trackID'] == old_ID_d1] = new_ID_d1 #set new trackID
                    self.track_output.at[t_tmp_1,'trackID_d2'][self.track_output.at[t_tmp_1,'trackID'] == old_ID_d1] = new_ID_d2 #set new trackID
                    self.track_output.at[t_tmp_1,'split_frame'][self.track_output.at[t_tmp_1,'trackID'] == old_ID_d1] = t #set new trackID
                for t_tmp_2 in range(t,len(self.label_stack)):
                    # import pdb
                    # pdb.set_trace()
                    self.track_output.loc[self.track_output.trackID == old_ID_d1].loc[t_tmp_2] = 111
                    #self.track_output.loc[t_tmp_2].at['trackID'== old_ID_d1] = 111
                    #self.track_output_correct.at[t_tmp_2,'trackID'][self.track_output.at[t_tmp_2,'trackID'] == old_ID_d1] = 111#new_ID_d2 #set new trackID
                    # self.track_output_correct.at[t_tmp_2,'trackID_mother'][self.track_output.at[t_tmp_2,'trackID'] == old_ID_d1] = old_ID_d1 #set new trackID
                    # self.track_output_correct.at[t_tmp_2,'trackID_d1'][self.track_output.at[t_tmp_2,'trackID'] == old_ID_d1] = new_ID_d1 #set new trackID
                    # self.track_output_correct.at[t_tmp_2,'trackID_d2'][self.track_output.at[t_tmp_2,'trackID'] == old_ID_d1] = new_ID_d1 #set new trackID
                    
                    # self.track_output_correct.at[t_tmp_2,'trackID'][self.track_output.at[t_tmp_2,'trackID'] == old_ID_d1] = new_ID_d2 #set new trackID
                    # self.track_output_correct.at[t_tmp_2,'trackID_mother'][self.track_output.at[t_tmp_2,'trackID'] == new_ID_d1] = old_ID_d1 #set new trackID
                    # self.track_output_correct.at[t_tmp_2,'trackID_d1'][self.track_output.at[t_tmp_2,'trackID'] == new_ID_d1] = new_ID_d1 #set new trackID
                    # self.track_output_correct.at[t_tmp_2,'trackID_d2'][self.track_output.at[t_tmp_2,'trackID'] == new_ID_d1] = new_ID_d1 #set new trackID



    def __new_ID(self):
        return np.max(self.label_stack_correct)+1


    def convert_data(self):
        """
        Convert tracking output into dataframe in standard format.
        """

        time = []
        area = []
        trackID = []
        trackID_d1 = []
        trackID_d2 = []
        trackID_mother = []
        x_coor = []
        y_coor = []
        intensity_mean = []
        intensity_min = []
        intensity_max = []
        minor_axis_length = []
        major_axis_length = []
        first_frame = []
        last_frame = []

        for t in self.tracks:
            time.append(t.t)
            area.append(t['area'])
            trackID_d1.append([t.ID]*len(t.t))
            trackID_d2.append([t.ID]*len(t.t))
            trackID_mother.append([t.parent]*len(t.t))
            x_coor.append(t.x)
            y_coor.append(t.y)
            intensity_mean.append(t['intensity_mean'])
            intensity_min.append(t['intensity_min'])
            intensity_max.append(t['intensity_max'])
            minor_axis_length.append(t['minor_axis_length'])
            major_axis_length.append(t['major_axis_length'])
            first_frame.append([t.t[0]]*len(t.t))
            last_frame.append([t.t[-1]]*len(t.t))
                
            if t.ID == t.parent:
                trackID.append([t.ID]*len(t.t))
                # trackID_mother.append([t.parent]*len(t.t))
            
            elif t.ID != t.parent:
                trackID.append([t.parent]*len(t.t))

        time = np.concatenate(time)
        area = np.concatenate(area)
        trackID = np.concatenate(trackID)
        trackID_d1 = np.concatenate(trackID_d1)
        trackID_d2 = np.concatenate(trackID_d2)
        trackID_mother = np.concatenate(trackID_mother)
        x_coor = np.concatenate(x_coor)
        y_coor = np.concatenate(y_coor)
        intensity_mean = np.concatenate(intensity_mean)
        intensity_min = np.concatenate(intensity_min)
        intensity_max = np.concatenate(intensity_max)
        minor_axis_length = np.concatenate(minor_axis_length)
        major_axis_length = np.concatenate(major_axis_length)
        first_frame = np.concatenate(first_frame)
        last_frame = np.concatenate(last_frame)

        df_conv = pd.DataFrame({'frame' : time, 
                                'trackID':trackID, 
                                'trackID_d1':trackID_d1, 
                                'trackID_d2':trackID_d2, 
                                'trackID_mother':trackID_mother, 
                                'split_frame':np.zeros(len(time)),
                                'area':area, 
                                'x':x_coor, 
                                'y':y_coor,
                                'intensity_mean':intensity_mean,
                                'intensity_min':intensity_min,
                                'intensity_max':intensity_max,
                                'minor_axis_length':minor_axis_length,
                                'major_axis_length':major_axis_length,
                                'first_frame':first_frame, 
                                'last_frame':last_frame})

        self.track_output = df_conv.groupby(['frame', 'trackID']).aggregate({'frame':'first',
                                                                'trackID':'first',
                                                                'trackID_d1':'first',
                                                                'trackID_d2':'last',
                                                                'trackID_mother':'first', 
                                                                'area':'first',
                                                                'x':'first',
                                                                'y':'first',
                                                                'intensity_mean':'first',
                                                                'intensity_min':'first',
                                                                'intensity_max':'first',
                                                                'minor_axis_length':'first',
                                                                'major_axis_length':'first',
                                                                'first_frame':'first',
                                                                'last_frame':'first'}).reindex(columns=df_conv.columns)

        self.track_output.set_index('frame', inplace=True)


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