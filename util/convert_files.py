import os
import glob
import numpy as np

import argparse
import bioformats as bf
import javabridge as jb
import tifffile
import tqdm

def create_dir_name(d):
	data_dir = 'data/'
	vsi_dir = data_dir + d
	tiff_dir = data_dir + d + '_tiff'
	return vsi_dir, tiff_dir

def create_dir(tiff_dir):
	if not os.path.exists(tiff_dir):
    		os.mkdir(tiff_dir)
    		print("Directory " , tiff_dir ,  " Created ")
	else:    
    		print("Directory " , tiff_dir ,  " already exists")

def svi_to_tiff(path, tiff_dir):
    with bf.ImageReader(path) as reader:
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
            filename = tiff_dir + os.path.splitext(os.path.basename(path))[0] + '.tiff'
            tifffile.imsave(filename,np.array(images))

def do_convert(vsi_files, tiff_dir):
	jb.start_vm(class_path=bf.JARS, max_heap_size="2G")
	pbar_files = tqdm.tqdm(vsi_files)

	for path in pbar_files:
    		svi_to_tiff(path, tiff_dir)
	
	jb.kill_vm()

def main():
	# pass argument
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--directory", type=str,
	help="Name of directory containing .vsi files.")
	args = vars(ap.parse_args())

	directory = args['directory'] #name of directory with vsi-files:
	vsi_dir, tiff_dir = create_dir_name(directory)
	create_dir(tiff_dir)

	vsi_files = glob.glob(vsi_dir + '/*.vsi')
	do_convert(vsi_files, tiff_dir)
	

if __name__ == "__main__":
    main()
