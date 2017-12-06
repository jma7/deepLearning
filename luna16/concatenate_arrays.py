import numpy as np
import glob
npfilespath="/work/05268/junma7/maverick/luna16/"
for file in ["testImages.npy","trainImages.npy","testMasks.npy","trainMasks.npy"]:
	npfiles=glob.glob("/work/05268/junma7/maverick/luna16/subset*/subset*_nodule/"+file)
	all_arrays = []
	for npfile in npfiles:
    		print(npfile)
    		all_arrays.extend(np.load(npfile))
	all_arrays = np.array(all_arrays)
	np.save(npfilespath+file, all_arrays)
