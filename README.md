# mtcnn-kcf

This is an inference implementation of MTCNN (Multi-task Cascaded Convolutional Network) to perform Face Detection and Alignment using OpenCV's DNN module. 

## MTCNN

[ZHANG2016] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.

https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf

## OpenCV's DNN module

Since OpenCV 3.1 there is a module called DNN that provides the inference support. The module is capable of taking models & weights from various popular frameworks such as Caffe, tensorflow, darknet etc.

You can read more about it here - https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV

Note that at present there is no support to perform training in OpenCV's DNN module and if I understood correctly there is no intention either.

### Requirements

* OpenCV 3.4+
* Boost FileSystem (1.58+)  [only required for the sample application]
* CMake 3.2+

### KCF tracker was added to improve runing time speed
### Build with flowing command
    cd mtcnn-kcf
    mkdir build
    cd build
    cmake ..
    cmake --build .

### Sample command
    ./facetracker/facetracker ../data/models/


## Acknowledgments

The MTCNN implementation take from https://github.com/golunovas/mtcnn-cpp

The model files are taken from https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code
The image file "Aaron_Peirsol_0003.jpg" is from the LFW database (http://vis-www.cs.umass.edu/lfw/)
The image files "dog.jpg" & "2007_007763.jpg" are from dlib's github repository (https://github.com/davisking/dlib/blob/master/examples/faces)