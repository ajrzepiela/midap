import os
import glob
import numpy as np

import argparse
import bioformats as bf
import imageio
import javabridge as jb
import tqdm

def create_dir_name(d):
	data_dir = '../data/'
	vsi_dir = data_dir + d #data_dir + d
	tiff_dir = data_dir + d + '_tiff/' #data_dir + d + '_tiff/'
	return vsi_dir, tiff_dir

def create_dir(tiff_dir):
	if not os.path.exists(tiff_dir):
    		os.mkdir(tiff_dir)
    		print("Directory " , tiff_dir ,  " Created ")
	else:    
    		print("Directory " , tiff_dir ,  " already exists")

def svi_to_tiff(file_name, tiff_dir):
    with bf.ImageReader(file_name) as reader:
        # shape of the data
        c_total = reader.rdr.getSizeC()
        z_total = reader.rdr.getSizeZ()
        t_total = reader.rdr.getSizeT()
        #pbar_c = tqdm.tqdm(range(c_total))

        for channel in range(c_total):
            images = []

            for time in range(t_total):
                for z in range(z_total):
                    image = reader.read(c=channel,
                                        z=z,
                                        t=time,
                                        rescale=False)
                    images.append(image)
            new_file_name = tiff_dir + os.path.splitext(os.path.basename(file_name))[0] + '.tiff'
            imageio.mimwrite(new_file_name,np.array(images))

def do_convert(file_name, tiff_dir):
    jb.start_vm(class_path=bf.JARS, max_heap_size="2G")
    #pbar_files = tqdm.tqdm(vsi_files)
    svi_to_tiff(file_name, tiff_dir)
    #for path in pbar_files:
       # print(path)
       # svi_to_tiff(path, tiff_dir)
	
    jb.kill_vm()


