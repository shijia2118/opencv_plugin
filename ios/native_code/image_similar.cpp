// image_similar.cpp
#include "image_similar.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace cv;

ImageSimilarity compareImageSimilarity(const char* image1Path, const char* image2Path) {
    ImageSimilarity similarityData;

    // 加载图片
    Mat image1 = imread(image1Path, IMREAD_GRAYSCALE);
    Mat image2 = imread(image2Path, IMREAD_GRAYSCALE);

    if (image1.empty() || image2.empty()) {
        similarityData.similarity = 0.0;
        return similarityData;
    }

    // 计算直方图
    Mat hist1, hist2;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    calcHist(&image1, 1, 0, Mat(), hist1, 1, &histSize, &histRange);
    calcHist(&image2, 1, 0, Mat(), hist2, 1, &histSize, &histRange);

    // 归一化直方图
    normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

    // 计算相似度（直方图相关性）
    similarityData.similarity = compareHist(hist1, hist2, HISTCMP_CORREL);

    return similarityData;
}
