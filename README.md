# patterns-detection-icpr2020

Release code of _"Unsupervised semantic discovery through visual patterns detection"_ paper.

> Since we are using `opencv` with `xfeatures2d` module for the extraction of DAISY descriptors, we had troubles using `virtualenv` or `conda`...we then opted for `docker`.

![semantical_levels](figure1.png)


# Run 

Download and run the image:

`sudo docker run --rm -it -p 8889:8889 -v /path/to/patterns-detection-icpr2020:/descriptor fpelosin/patterns-detection-icpr:v2 bash`

After inside the container just run:

`jupyter-lab --allow-root --port=8889 --ip=0.0.0.0 --no-browser`

You can now access the notebook at the prompted link. You can also change `8889` to whatever port you like.


# Dataset

In the `datasets` folder you find the released dataset. The labeling has been made through the [labelme](https://github.com/wkentaro/labelme) annotation tool. The format of the annotation is the Pascal VOC format. 


