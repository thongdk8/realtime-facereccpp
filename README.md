# Real-time face recognition with mtcnn, kcf tracker and arcface (insightface)

This is an inference implementation of real time face recogtion, wholely written in C++ with TVM inference runtine. Using mtcnn for facedetection, kcf tracker and insightface model.

## MTCNN

[ZHANG2016] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.

https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf


## Insightface
Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019. 

[ArXiv tecnical report](https://arxiv.org/abs/1801.07698)

https://github.com/deepinsight/insightface

### Requirements

* OpenCV 3.4+
* Boost FileSystem (1.58+) 
* CMake 3.2+
* Clang 6.0+
* TVM runtime

#### You can folowing [this doc file](install_requirements.md) or [official doc](https://docs.tvm.ai/install/from_source.html#) to install TVM runtime


### Build with flowing command
Change two line cmake config in [facerecogtion model](facerecognition/CMakeLists.txt) to your TVM source that you downloaded:
- set (DMLC_INCLUDE "/home/thongpb/works/tvm/3rdparty/dmlc-core/include")
- set (DLPACK_INC "/home/thongpb/works/tvm/3rdparty/dlpack/include")

Compile:

    cd realtime-facereccpp
    mkdir build
    cd build
    cmake ..
    make -j4

### Sample command
You can modify parameters in [params.xml](data/params.xml)

    ./facerecognition/facerecognition ../data/params.xml


## Acknowledgments
The MTCNN implementation take from https://github.com/golunovas/mtcnn-cpp

The model used for facial feature extraction came from [insightface MODEL_ZOO](https://github.com/deepinsight/insightface/wiki/Model-Zoo) 

The mtcnn model files are taken from https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code 
