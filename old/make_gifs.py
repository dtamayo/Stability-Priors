from PIL import Image
import numpy as np
import glob, os

image_type = ".png"

# for k in ["matern52_kernel", "rq_kernel", "se_kernel", "quasi_periodic_kernel"]:

	# directory_with_images = "C:/Users/Christian/Dropbox/GP_research/julia/figs/gp/" + k + "/training/"
	# directory_with_images = "C:/Users/Christian/Dropbox/GP_research/julia/figs/gp/full/" + k + "/training/"

	# print("looking for " + image_type + " images in " + directory_with_images)
print("looking for " + image_type + " images in " + os.path.dirname(os.path.realpath(__file__)))

# os.chdir(directory_with_images)

for j in ["afterMLE", "afterpriors", "beforeMLE", "corner", "e0", "e1", "e2", "K0", "K1", "K2", "totalrvs"]:

    filenames = "figs/" + j + "*"

    # open all of the images
    images=[]
    for file in glob.glob(filenames + image_type):
        images.append(Image.open(file))
        print(file)

    # combine the images into a gif
    images[0].save("figs/" + j + '.gif',
        save_all=True,
        append_images=images[1:],
        duration=1500,
        loop=0)
