#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/timer/timer.hpp>

#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "recognizer.h"
#include <mtcnn/detector.h>

namespace fs = boost::filesystem;
using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

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



class DbManager {
  private:
    std::vector<cv::Mat> features;
    std::vector<std::string> names;

    cv::Mat crr_fea;

    float L2Distance(cv::Mat &v1, cv::Mat &v2) {
        cv::Mat sub;
        cv::subtract(v1, v2, sub);
        cv::Mat l2;
        cv::multiply(sub, sub, l2);
        return cv::sum(l2).val[0];
    }

    float calDiss(cv::Mat fea){
        return L2Distance(fea, crr_fea);
    }


  public:
    DbManager(MTCNNDetector &detector, FR_MFN_Deploy &deploy,
              std::string db_dir) {
        std::vector<fs::path> subdirs;
        for (auto sub : fs::directory_iterator(fs::path(db_dir))) {
            if (fs::is_directory(sub.path())) {
                subdirs.push_back(sub);
                std::cout << sub.path().filename() << std::endl;

                for (auto f : fs::directory_iterator(sub)) {
                    if (fs::is_regular_file(f)) {
                        cv::Mat img = cv::imread(f.path().string());
                        if (!img.empty()) {
                            std::vector<Face> faces;
                            faces = detector.detect(img, 20.f, 0.709f);
                            // std::cout << "Number of faces found in the "
                            //              "supplied image - "
                            //           << faces.size() << std::endl;
                            std::vector<rectPoints> data;
                            for (size_t i = 0; i < faces.size(); ++i) {
                                std::vector<cv::Point> pts;
                                for (int p = 0; p < NUM_PTS; ++p) {
                                    pts.push_back(cv::Point(
                                        faces[i].ptsCoords[2 * p],
                                        faces[i].ptsCoords[2 * p + 1]));
                                }

                                auto rect = faces[i].bbox.getRect();
                                auto d = std::make_pair(rect, pts);
                                data.push_back(d);

                                cv::Rect r(
                                    std::max(rect.x - 20, 0),
                                    std::max(rect.y - 20, 0),
                                    std::min(rect.width + 40,
                                             img.cols - std::max(rect.x, 0)),
                                    std::min(rect.height + 40,
                                             img.rows - std::max(rect.y, 0)));

                                cv::Mat croped = img(r);
                                cv::Mat fea = deploy.forward(croped.clone());
                                // cv::imshow("croped", croped);

                                this->features.push_back(fea);
                                this->names.push_back(sub.path().filename().string());

                                break;

                            }

                            img = drawRectsAndPoints(img, data);

                            // cv::imshow("img", img);
                            // int k = cv::waitKey(1);
                            // if (k == 27) {
                            //     break;
                            // }
                        }
                    }
                }
            }
        }
    }
    
    std::vector<std::pair<std::string, float>> recommend(cv::Mat &fea, int k){
        this->crr_fea = fea;
        std::vector<float> diss;
        std::vector<std::pair<std::string, float>> results;
        diss.resize(this->features.size());
        std::transform(this->features.begin(), this->features.end(), diss.begin(), [this](cv::Mat f){return this->calDiss(f);});

        std::vector<size_t> idx(this->features.size());
        std::iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(),
             [&diss](size_t i1, size_t i2) { return diss[i1] < diss[i2]; });
        
        for (size_t i = 0; i<k; i++){
            results.push_back( make_pair(this->names[idx[i]], diss[idx[i]]));
        }
        return results;
    }


};