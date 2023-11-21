[![Pytest](https://github.com/wimpomp/ndbioimage/actions/workflows/pytest.yml/badge.svg)](https://github.com/wimpomp/ndbioimage/actions/workflows/pytest.yml)

# ndbioimage - Work in progress

Exposes (bio) images as a numpy ndarray-like object, but without loading the whole
image into memory, reading from the file only when needed. Some metadata is read
and stored in an [ome](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2005-6-5-r47) structure.
Additionally, it can automatically calculate an affine transform that corrects for chromatic aberrations etc. and apply
it on the fly to the image.

Currently, it supports imagej tif files, czi files, micromanager tif sequences and anything
[bioformats](https://www.openmicroscopy.org/bio-formats/) can handle. 

## Installation

```
pip install ndbioimage
```

## Usage

- Reading an image file and plotting the frame at channel=2, time=1

```
import matplotlib.pyplot as plt
from ndbioimage import Imread
with Imread('image_file.tif', axes='ctxy', dtype=int) as im:
    plt.imshow(im[2, 1])
```        
        
- Showing some image metadata

```
from ndbioimage import Imread
from pprint import pprint
with Imread('image_file.tif') as im:
    pprint(im)
```

- Slicing the image without loading the image into memory

```
from ndbioimage import Imread
with Imread('image_file.tif', axes='cztxy') as im:
    sliced_im = im[1, :, :, 100:200, 100:200]
```

sliced_im is an instance of Imread which will load any image data from file only when needed


- Converting (part) of the image to a numpy ndarray

```
from ndbioimage import Imread
import numpy as np
with Imread('image_file.tif', axes='cztxy') as im:
    array = np.asarray(im[0, 0])
```

## Adding more formats
Readers for image formats subclass AbstractReader. When an image reader is imported, Imread will
automatically recognize it and use it to open the appropriate file format. Image readers
are required to implement the following methods:

- staticmethod _can_open(path): return True if path can be opened by this reader
- property ome: reads metadata from file and adds them to an OME object imported
from the ome-types library 
- \_\_frame__(self, c, z, t): return the frame at channel=c, z-slice=z, time=t from the file

Optional methods:
- open(self): maybe open some file handle
- close(self): close any file handles

Optional fields:
- priority (int): Imread will try readers with a lower number first, default: 99
- do_not_pickle (strings): any attributes that should not be included when the object is pickled,
for example: any file handles

# TODO
- more image formats
