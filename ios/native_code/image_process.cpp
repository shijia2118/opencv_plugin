// image_process.cpp
#include "image_process.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace cv;

/// @brief 图片相似度(直方图)
/// @param image1Path 
/// @param image2Path 
/// @return 
ImageSimilarity compareImageSimilarityHist(const char* image1Path, const char* image2Path) {
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

/// @brief 计算哈希值(感知哈希)
/// @param imagePath 
/// @return 
std::string calculatePerceptualHash(const char* imagePath){
    // 读取图像
    cv::Mat image = cv::imread(imagePath);

    if (image.empty())
    {
        return "";  // 图像读取失败，返回空的哈希值
    }

    // 将图像调整为8x8大小
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(8, 8));

    // 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);

    // 计算平均灰度值
    cv::Scalar meanValue = cv::mean(grayImage);

    // 生成感知哈希值
    std::string hash;
    for (int i = 0; i < grayImage.rows; i++)
    {
        for (int j = 0; j < grayImage.cols; j++)
        {
            if (grayImage.at<uchar>(i, j) >= meanValue.val[0])
            {
                hash += "1";
            }
            else
            {
                hash += "0";
            }
        }
    }

    return hash;
}

/// @brief 图片相似度(感知哈希)
/// @param imagePath1 
/// @param imagePath2 
/// @return 
ImageSimilarity compareImageSimilarityPhash(const char* imagePath1, const char* imagePath2) {
    ImageSimilarity similarityData;

    // 计算感知哈希值
    std::string hash1 = calculatePerceptualHash(imagePath1);
    std::string hash2 = calculatePerceptualHash(imagePath2);

    // 比较哈希值并计算相似度
    int matchingBits = 0;
    for (int i = 0; i < hash1.length(); i++){
        if (hash1[i] == hash2[i]) {
            matchingBits++;
        }
    }

    double similarity = static_cast<double>(matchingBits) / hash1.length();
    similarityData.similarity = similarity;
    return similarityData;
}




/// @brief 图片相似度(SSIM算法)
/// @param imagePath1 
/// @param imagePath2 
/// @return 
ImageSimilarity compareImageSimilaritySSIM(const char* image1Path, const char* image2Path){
    ImageSimilarity similarityData;

    // 加载图片
    cv::Mat image1 = cv::imread(image1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(image2Path, cv::IMREAD_GRAYSCALE);

    if (image1.empty() || image2.empty()) {
        // 图片加载失败，返回相似度为 0
        similarityData.similarity = 0.0;
        return similarityData;
    }

    // 将图片转换为浮点型
    cv::Mat image1f, image2f;
    image1.convertTo(image1f, CV_32F);
    image2.convertTo(image2f, CV_32F);

    // 计算均值和标准差
    cv::Scalar mean1, stdDev1;
    cv::meanStdDev(image1f, mean1, stdDev1);
    
    cv::Scalar mean2, stdDev2;
    cv::meanStdDev(image2f, mean2, stdDev2);

    // 计算方差和协方差
    cv::Mat covariance, var1, var2;
    cv::multiply(image1f - mean1, image2f - mean2, covariance);
    cv::multiply(image1f - mean1, image1f - mean1, var1);
    cv::multiply(image2f - mean2, image2f - mean2, var2);

    // 计算 SSIM
    double k1 = 0.01;
    double k2 = 0.03;
    double c1 = (k1 * 255) * (k1 * 255);
    double c2 = (k2 * 255) * (k2 * 255);
    double num = (2 * mean1.val[0] * mean2.val[0] + c1) * (2 * covariance.at<float>(0) + c2);
    double den = (mean1.val[0] * mean1.val[0] + mean2.val[0] * mean2.val[0] + c1) * (var1.at<float>(0) + var2.at<float>(0) + c2);
    if(den==0) {
        similarityData.similarity = 0;
        return similarityData;
    }

    double ssim = num / den;
    similarityData.similarity = ssim;

    return similarityData;
}


/// @brief 无参考图片时，图像清晰度
/// 边缘检测法
/// @param image 
double calculateImageBlur(const char* imagePath) {
    // 读取图像
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        return 0.0;
    }

    // 计算Canny边缘
    cv::Mat edges;
    cv::Canny(image, edges, 100, 200);

    // 计算非零像素数量
    int nonZero = cv::countNonZero(edges);

    // 返回模糊度值（非零像素数量）
    return static_cast<double>(nonZero);
}

/// @brief 获取相似图片数组
/// @param queryImagePath 
/// @param imagePaths 
/// @return 相似图片组
SimilarityArray findSimilarImages(const char* queryImagePath, const char** imagePaths, int numImagePaths) {
    SimilarityArray result;
    result.size = 0;
    result.imagePaths = new const char*[numImagePaths];

    for (int i = 0; i < numImagePaths; i++) {
        const char* imagePath = imagePaths[i];
        // 计算当前图像与指定图像的相似度
        ImageSimilarity imageSimilarity = compareImageSimilarityPhash(queryImagePath, imagePath);

        // 判断是否相似，相似度阈值设为0.9
        if (imageSimilarity.similarity >= 0.9) {
            result.imagePaths[result.size++] = imagePath; // 将相似图片路径添加到相似图片路径列表中
        }
    }

    return result; // 返回相似图片路径
}















