#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/timer/timer.hpp>

#include <mutex>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>

#include <string>
#include <thread>

#include "BoxManager.h"
#include <mtcnn/detector.h>
// #include "recognizer.h"
#include "DbManager.h"
#include "tinyxml2.h"

using namespace cv;
using namespace tinyxml2;
namespace fs = boost::filesystem;
using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

//Threading management
cv::Mat frame;
std::mutex fr_mtx;
bool exitDetection = false;
bool exitReader = false;
bool initDbCompleted = false;

//Pramameters read from xml config file
float recognition_threshold;
std::string mtcnn_dir;
std::string facerec_model_dir;
std::string camid;
std::string database_dir;


void frameReader(cv::VideoCapture &cap) {
    while (!exitReader) {
        cap.read(frame);
    }
}

void detectionThread(fs::path modelDir, std::vector<Ptr<Tracker>> &trackers,
                     BoxManager &boxManager) {
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

    FR_MFN_Deploy deploy(
        facerec_model_dir);

    DbManager db(
        detector, deploy,
        database_dir);
    initDbCompleted = true;

    // cv::Mat fea_thong = deploy.forward(cv::imread(
    //     "/home/thongpb/works/face_recognition/opencv-mtcnn/build/27.jpg"));

    // sleep(2);
    while (!exitDetection) {
        fr_mtx.lock();
        cv::Mat img = frame.clone();
        // cv::resize(img, img, img.size(), 1/2, 1/2);
        fr_mtx.unlock();

        if (!img.empty()) {
            std::vector<Face> faces;
            // std::vector<cv::Rect> boxes;
            {
                boost::timer::auto_cpu_timer t(3, "Detection: %w seconds\n");
                faces = detector.detect(img, 40.f, 0.709f);
            }
            // std::cout << "Number of faces found in the supplied image - "
            //           << faces.size() << std::endl;

            std::vector<rectPoints> data;

            // show the image with faces in it
            for (size_t i = 0; i < faces.size(); ++i) {
                boost::timer::auto_cpu_timer t(3, "Recognition: %w seconds\n");
                std::vector<cv::Point> pts;
                for (int p = 0; p < NUM_PTS; ++p) {
                    pts.push_back(cv::Point(faces[i].ptsCoords[2 * p],
                                            faces[i].ptsCoords[2 * p + 1]));
                }

                auto rect = faces[i].bbox.getRect();
                auto d = std::make_pair(rect, pts);
                data.push_back(d);

                cv::Rect r(
                    std::max(rect.x-20, 0), std::max(rect.y-20, 0),
                    std::min(rect.width+40, img.cols - std::max(rect.x, 0)),
                    std::min(rect.height+40, img.rows - std::max(rect.y, 0)));

                cv::Mat croped = img(r);
                // cv::imwrite(std::to_string(rand()%1000) + ".jpg", croped);
                cv::Mat fea =  deploy.forward(croped.clone());

                std::string resid("Similar: ");
                auto ress = db.recommend(fea, 1);
                for (auto r : ress) {
                    std::cout << "Similar : " << r.first << ": " << r.second
                              << '\t';
                    resid += r.first + ' ' + std::to_string(r.second);
                }
                std::cout << std::endl;

                int idBox = boxManager.getBoxIdx(r);
                if (idBox == -1) {
                    // boxManager.addBox(rect, "id" + std::to_string( boxManager.size() + 1 ));
                    if(ress[0].second < recognition_threshold){
                        boxManager.addBox( r, resid, ress[0].second);
                    }
                    else {
                        boxManager.addBox(r, "Unkown", 10);
                    }
                    Ptr<Tracker> tracker;
                    cv::TrackerKCF::Params param;
                    param.desc_pca = cv::TrackerKCF::GRAY;
                    param.desc_npca = cv::TrackerKCF::GRAY;
                    param.max_patch_size = 20 * 20;
                    param.compress_feature = true;
                    param.compressed_size = 1;
                    // param.wrap_kernel = true;

                    tracker = TrackerKCF::create(param);
                    tracker->init(img, cv::Rect2d(r));
                    trackers.push_back(tracker);
                } else if (idBox >= 0){
                    // std::cout << "Box is existed" << std::endl;
                    if (ress[0].second < recognition_threshold &&
                        ress[0].second < boxManager.accuracies[idBox]) {
                        boxManager.ids[idBox] = resid;
                        boxManager.accuracies[idBox] = ress[0].second;
                    }
                }
            }
            // frame = drawRectsAndPoints(img, data);
        }
    }
}



int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "Usage " << argv[0] << ": "
                  << "<param-file> ";
        return -1;
    }
    //Read parameters
    XMLDocument doc;
    doc.LoadFile(argv[1]);
    database_dir = doc.FirstChildElement("CONFIG")->FirstChildElement("DATABASE_DIR")->GetText();
    camid = doc.FirstChildElement("CONFIG")->FirstChildElement("CAM")->GetText();
    mtcnn_dir = doc.FirstChildElement("CONFIG")->FirstChildElement("MTCNN_MODEL")->GetText();
    facerec_model_dir = doc.FirstChildElement("CONFIG")->FirstChildElement("FACE_MODEL")->GetText();
    recognition_threshold = std::stof(doc.FirstChildElement("CONFIG")->FirstChildElement("THRESHOLD")->GetText());

    //Init detector
    BoxManager boxManager(0.2, 10000);
    fs::path modelDir = fs::path(mtcnn_dir);
    std::vector<Ptr<Tracker>> trackers;
    //Start detection and recognition thread
    std::thread detector(detectionThread, modelDir, std::ref(trackers), std::ref(boxManager));

    //Init camera
    cv::VideoCapture cap;
    if (camid.length() < 2){
        cap.open(std::stoi(camid));
    } else {
        cap.open(camid);
    }
    //Start frame reader thread
    std::thread reader(frameReader, std::ref(cap));

    sleep(2);
    while (true) {
        int64 t0 = cv::getTickCount();

        // cap.read(frame);
        cv::Mat frame_dis = frame.clone();
        if (!frame_dis.empty()) {
            std::vector<Rect> newBoxes;
            // Draw tracked objects
            for (unsigned i = 0; i < trackers.size(); i++) {
                cv::Rect2d bbox;
                bool exist = trackers[i]->update(frame_dis, bbox);
                if (exist) {
                    rectangle(frame_dis, bbox, cv::Scalar(255, 0, 0), 2, 1);
                    putText(frame_dis, boxManager.getName(i), Point(bbox.x, bbox.y),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 2);
                    newBoxes.push_back((cv::Rect)bbox);
                } else {
                    trackers.erase(trackers.begin() + i);
                    boxManager.remove(i);
                }
            }
            boxManager.updateBoxes(newBoxes);

            if(!initDbCompleted){
                putText(frame_dis, "INITIALIZING DATABASE .....", Point(0, 50),
                        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 250), 2);
            }

            cv::imshow("test", frame_dis);
            int k = cv::waitKey(1);
            if (k == 27) {
                exitReader = true;
                exitDetection = true;
                break;
            }
        }

        int64 t1 = cv::getTickCount();
        double secs = (t1 - t0) / cv::getTickFrequency();
        std::cout << "Excution time: " << secs << " ~FPS: " << 1 / secs
                  << std::endl;
    }
    detector.join();
    reader.join();
    return 0;
}
