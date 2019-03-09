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

using namespace cv;

namespace fs = boost::filesystem;

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;
cv::Mat frame;
std::mutex fr_mtx;
BoxManager boxManager(0.2, 10000);
bool exitDetection = false;
bool exitReader = false;
std::string trackerType = "KCF";

static cv::Mat drawRectsAndPoints(const cv::Mat &img,
                                  const std::vector<rectPoints> data) {
    cv::Mat outImg;
    img.convertTo(outImg, CV_8UC3);

    for (auto &d : data) {
        cv::rectangle(outImg, d.first, cv::Scalar(0, 0, 255));
        auto pts = d.second;
        for (size_t i = 0; i < pts.size(); ++i) {
            cv::circle(outImg, pts[i], 3, cv::Scalar(0, 0, 255));
        }
    }
    return outImg;
}

void frameReader(cv::VideoCapture &cap) {
    while (!exitReader) {
        cap.read(frame);
    }
}

void detectionThread(fs::path modelDir, std::vector<Ptr<Tracker>> &trackers) {
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

    sleep(2);
    while (!exitDetection) {
        // std::cout << "aaaa" << endl;
        fr_mtx.lock();
        cv::Mat img = frame.clone();
        // cv::resize(img, img, img.size(), 1/2, 1/2);
        fr_mtx.unlock();
        if (!img.empty()) {
            std::vector<Face> faces;
            // std::vector<cv::Rect> boxes;
            {
                boost::timer::auto_cpu_timer t(3, "%w seconds\n");
                faces = detector.detect(img, 20.f, 0.709f);
            }

            std::cout << "Number of faces found in the supplied image - "
                      << faces.size() << std::endl;

            std::vector<rectPoints> data;

            // show the image with faces in it
            for (size_t i = 0; i < faces.size(); ++i) {
                std::vector<cv::Point> pts;
                for (int p = 0; p < NUM_PTS; ++p) {
                    pts.push_back(cv::Point(faces[i].ptsCoords[2 * p],
                                            faces[i].ptsCoords[2 * p + 1]));
                }

                auto rect = faces[i].bbox.getRect();
                auto d = std::make_pair(rect, pts);
                data.push_back(d);
                bool isNewBox = boxManager.isNewBox(rect);
                if (isNewBox) {
                    boxManager.addBox(rect, "id" + std::to_string( boxManager.size() + 1 ));
                    Ptr<Tracker> tracker;
                    tracker = TrackerKCF::create();
                    tracker->init(img, cv::Rect2d(rect));
                    trackers.push_back(tracker);
                } else {
                    std::cout << "Box is existed" << std::endl;
                }
            }
            // frame = drawRectsAndPoints(img, data);
        }
    }
}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "Usage " << argv[0] << ": "
                  << "<model-dir> ";
        return -1;
    }

    fs::path modelDir = fs::path(argv[1]);

    // Create multitracker
    // Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();
    std::vector<Ptr<Tracker>> trackers;

    std::thread detector(detectionThread, modelDir, std::ref(trackers));

    // cv::VideoCapture cap(
    //     "/home/thongpb/works/face_recognition/data/output.avi");
    cv::VideoCapture cap(0);
    std::thread reader(frameReader, std::ref(cap));

    while (true) {
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
                            FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
                    newBoxes.push_back((cv::Rect)bbox);
                } else {
                    trackers.erase(trackers.begin() + i);
                    boxManager.remove(i);
                }
            }
            boxManager.updateBoxes(newBoxes);

            cv::imshow("test", frame_dis);
            int k = cv::waitKey(1);
            if (k == 27) {
                exitReader = true;
                exitDetection = true;
                break;
            }
        }
    }
    detector.join();
    reader.join();
    return 0;
}
