#include <DbManager.h>
#include "tinyxml2.h"
#include <stdio.h>
int main2(int argc, char **argv) {
    cv::Mat A = cv::imread(
        "/home/thongpb/works/face_recognition/opencv-mtcnn/build/thong.jpg");
    cv::Mat B = cv::imread("/media/thongpb/EDisk/datasets/couple_faces/"
                           "aligned_sync/vs/thong_VS/26.png");
    FR_MFN_Deploy deploy(
        "/home/thongpb/works/face_recognition/opencv-mtcnn/scripts/mobilenet");

    int64 t0 = cv::getTickCount();
    cv::Mat v2 = deploy.forward(B);
    cv::Mat v1 = deploy.forward(A);
    int64 t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();

    std::cout << "Similar : " << L2Distance(v1, v2) << std::endl;

    // std::cout << CosineDistance(v1, v2) << std::endl;
    std::cout << "Excution time: " << secs << " ~FPS: " << 1 / secs
              << std::endl;

    fs::path modelDir = fs::path(argv[1]);
    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = (modelDir / "det1.caffemodel").string();
    pConfig.protoText = (modelDir / "det1.prototxt").string();
    pConfig.threshold = 0.6f;

    RefineNetwork::Config rConfig;
    rConfig.caffeModel = (modelDir / "det2.caffemodel").string();
    rConfig.protoText = (modelDir / "det2.prototxt").string();
    rConfig.threshold = 0.7f;

    OutputNetwork::Config oConfig;
    oConfig.caffeModel = (modelDir / "det3.caffemodel").string();
    oConfig.protoText = (modelDir / "det3.prototxt").string();
    oConfig.threshold = 0.7f;

    MTCNNDetector detector(pConfig, rConfig, oConfig);
    DbManager db(
        detector, deploy,
        "/home/thongpb/works/face_recognition/opencv-mtcnn/data/sample_db");

    for (auto r : db.recommend(v2, 2)) {
        std::cout << r.first << ": " << r.second << '\t';
    }

    return 0;
}

using namespace tinyxml2;
using namespace std;

int main3(int argc, char **argv) {

    XMLDocument doc;
    doc.LoadFile("params.xml");
    cout << "aaaaaaaaaa" <<endl;
    // Structure of the XML file:
    // - Element "PLAY"      the root Element, which is the
    //                       FirstChildElement of the Document
    // - - Element "TITLE"   child of the root PLAY Element
    // - - - Text            child of the TITLE Element

    // Navigate to the title, using the convenience function,
    // with a dangerous lack of error checking.
   std::string title =
        doc.FirstChildElement("CONFIG")->FirstChildElement("DATABASE_DIR")->GetText();
    cout << title << endl;

    title =
        doc.FirstChildElement("CONFIG")->FirstChildElement("CAM")->GetText();
    printf("Name of play (2): %s\n", title);
    return 0;
}