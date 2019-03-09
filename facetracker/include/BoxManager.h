#include<iostream>
#include<opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using vectorRect = std::vector<cv::Rect>;


class BoxManager
{
    public: 
        vectorRect boxes;
        std::vector<std::string> ids;
        double iouThreshold;
        int minArea;

    public:
        BoxManager(double iouThr, int minA){
            this->iouThreshold = iouThr;
            this->minArea = minA;
            std::cout << "Init boxmanager with iou threshold: " << iouThr << std::endl;
        }

        void setIouThreshold(double thr){
            this->iouThreshold = thr;
        }

        void updateBoxes(vectorRect crrBoxes){
            this->boxes = crrBoxes;
        }

        void addBox(cv::Rect box, std::string id){
            this->boxes.push_back(box);
            this->ids.push_back(id);
        }

        bool isNewBox(cv::Rect box){
            if (box.width * box.height < minArea){
                return false;
            }
            for (auto b : this->boxes){
                double iou = _calIOU(box, b);
                if (iou >= this->iouThreshold) {
                    return false;
                }
            }
            return true;
        }

        double _calIOU( cv::Rect box1, cv::Rect box2){
            int xA = std::max(box1.x, box2.x);
            int yA = std::max(box1.y, box2.y);
            int xB = std::min(box1.x + box1.width, box2.x + box2.width);
            int yB = std::min(box1.y + box1.height, box2.y + box2.height);

            int interArea = std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1);
            int box1Area = box1.width * box1.height;
            int box2Area = box2.width * box2.height;

            double iou = (double)interArea / double(box1Area + box2Area - interArea);
            std::cout << "IOU: " << iou << std::endl;
            return iou;
        }

        void remove(int i){
            this->boxes.erase(this->boxes.begin() + i);
            this->ids.erase(this->ids.begin() + i);
        }

        std::string getName(int i){
            return ids[i];
        }

        int size(){
            return boxes.size();
        }
};