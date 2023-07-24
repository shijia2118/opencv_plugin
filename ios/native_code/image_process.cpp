// image_process.cpp
#include "image_process.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

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
double compareImageSimilarityPhash(const char* imagePath1, const char* imagePath2) {
    ImageSimilarity similarityData;

    // 计算感知哈希值
    std::string hash1 = calculatePerceptualHash(imagePath1);
    std::string hash2 = calculatePerceptualHash(imagePath2);

    if(hash1.empty() || hash2.empty()){
        //遇到hash值为""
        return 0.0;
    }


    // 比较哈希值并计算相似度
    int matchingBits = 0;
    for (int i = 0; i < hash1.length(); i++){
        if (hash1[i] == hash2[i]) {
            matchingBits++;
        }
    }

    double similarity = static_cast<double>(matchingBits) / hash1.length();
    return similarity;
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

    // 检查图像尺寸
    if (image1.size() != image2.size()) {
        // 图像尺寸不一致，返回相似度为 0
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


/// @brief 图像清晰度
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



/// @brief 计算2各frame相似度
/// @param frame1 
/// @param frame2 
/// @return 
double calculateFrameSimilarity(cv::Mat frame1, cv::Mat frame2) {
    cv::Mat grayFrame1, grayFrame2;

    // 转换为灰度图像
    if (frame1.channels() == 3) {
        cv::cvtColor(frame1, grayFrame1, cv::COLOR_BGR2GRAY);
    } else {
        grayFrame1 = frame1.clone();
    }
    if (frame2.channels() == 3) {
        cv::cvtColor(frame2, grayFrame2, cv::COLOR_BGR2GRAY);
    } else {
        grayFrame2 = frame2.clone();
    }

      // 调整图像尺寸为8x8
    cv::resize(grayFrame1, grayFrame1, cv::Size(8, 8));
    cv::resize(grayFrame2, grayFrame2, cv::Size(8, 8));

    // 计算平均灰度值
    double mean1 = cv::mean(grayFrame1)[0];
    double mean2 = cv::mean(grayFrame2)[0];

    // 计算平均哈希值
    uchar hash1[64], hash2[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            hash1[i * 8 + j] = (grayFrame1.at<uchar>(i, j) > mean1) ? 1 : 0;
            hash2[i * 8 + j] = (grayFrame2.at<uchar>(i, j) > mean2) ? 1 : 0;
        }
    }

    // 计算汉明距离
    int distance = 0;
    for (int i = 0; i < 64; i++) {
        if (hash1[i] != hash2[i]) {
            distance++;
        }
    }

    // 计算相似度
    double similarity = 1.0 - static_cast<double>(distance) / 64.0;

    return similarity;
}

/// @brief 峰值信噪比
/// @param I1 
/// @param I2 
/// @return  一般的取值范围20~50
double getPSNR(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    Scalar s = sum(s1);        // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

/// @brief SSIM算法
/// @param i1 
/// @param i2 
/// @return 
Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}


/// @brief 构建捕获视频帧的对象
/// @param videoPath 
/// @return 
cv::VideoCapture createVideoCapture(const std::string& videoPath) {
    cv::VideoCapture videoCapture;

#if defined(__ANDROID__)
    videoCapture = cv::VideoCapture(videoPath, cv::CAP_ANDROID);
#else
    videoCapture = cv::VideoCapture(videoPath);
#endif
    return videoCapture;
}

cv::Mat calculatePerceptualHash2(const char* imagePath) {
    // 读取图像
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        return cv::Mat();  // Return an empty cv::Mat on image read failure
    }

    // 将图像调整为8x8大小
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(8, 8));

    // 转换为灰度图像，并映射0和1为255和0
    cv::Mat grayImage;
    cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);
    grayImage = 255 - grayImage;

    return grayImage;
}



class ParallelExtractImageFeatures : public cv::ParallelLoopBody {
public:
    ParallelExtractImageFeatures(const char** imagePaths, int numImages, std::vector<cv::Mat>& features)
        : imagePaths_(imagePaths), numImages_(numImages), features_(features) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            const char* videoPath(imagePaths_[i]);
            cv::Mat hist = calculatePerceptualHash2(videoPath);
            features_[i] = hist;
        }
    }

private:
    const char** imagePaths_;
    int numImages_;
    std::vector<cv::Mat>& features_;
};

/// @brief 图片分类(Kmeans)
/// @param imagePaths 
/// @param numImages 
/// @return 
int* clusterImages(const char** imagePaths, int numImages, int K) {
    // 并行计算感知哈希值
    std::vector<cv::Mat> features(numImages);
    #pragma omp parallel for
    for (int i = 0; i < numImages; ++i) {
        features[i] = calculatePerceptualHash2(imagePaths[i]);
    }

    // 将哈希值转换为特征向量
    cv::Mat data(features.size(), features[0].cols * features[0].rows, CV_32F);
    for (int i = 0; i < features.size(); ++i) {
        for (int j = 0; j < features[i].cols * features[i].rows; ++j) {
            data.at<float>(i, j) = static_cast<float>(features[i].at<uchar>(0, j));
        }
    }

    // 使用kmeans算法对特征向量进行聚类
    cv::Mat labels;
    cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS);

    // 将聚类结果转换为标签数组并返回
    int* result = new int[labels.rows];
    for (int i = 0; i < labels.rows; ++i) {
        result[i] = labels.at<int>(i);
    }
    return result;
}




/// @brief 视频特征值(直方图)
/// @param videoPath 
/// @return 
cv::Mat extractVideoFeatures(const char* videoPath) {
    cv::VideoCapture cap = createVideoCapture(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return cv::Mat();
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frameInterval = static_cast<int>(fps) ;

    std::vector<cv::Mat> histograms;
    cv::Mat frame;
    int frameCount = 0;
    while (cap.read(frame)) {
        if (frameCount % frameInterval == 0) {
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

            int h_bins = 50, s_bins = 60;
            int histSize[] = { h_bins, s_bins };

            float h_ranges[] = { 0, 180 };
            float s_ranges[] = { 0, 256 };
            const float* ranges[] = { h_ranges, s_ranges };

            int channels[] = { 0, 1 };

            cv::Mat hist;
            cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

            histograms.push_back(hist);
        }
        frameCount++;
    }

    cv::Mat result;
    cv::Mat histResult;
    for (const auto& hist : histograms) {
        result.push_back(hist);
    }
    cv::resize(result, histResult, cv::Size(256, 1), 0, 0, cv::INTER_LINEAR);
    return histResult;
}


/// @brief 比较2个视频的相似度
/// @param videoPath1 
/// @param videoPath2 
/// @return 
double compareVideoSimilarity(const char* videoPath1, const char* videoPath2) {
    cv::Mat hist1 = extractVideoFeatures(videoPath1);
    cv::Mat hist2 = extractVideoFeatures(videoPath2);

    double similarity = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
    return similarity;
}

class ParallelExtractFeatures : public cv::ParallelLoopBody {
public:
    ParallelExtractFeatures(const char** videoPaths, int numVideos, std::vector<cv::Mat>& features)
        : videoPaths_(videoPaths), numVideos_(numVideos), features_(features) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            const char* videoPath(videoPaths_[i]);
            cv::Mat hist = extractVideoFeatures(videoPath);
            features_[i] = hist;
        }
    }

private:
    const char** videoPaths_;
    int numVideos_;
    std::vector<cv::Mat>& features_;
};


/// @brief 视频分类(Kmeans)
/// @param videoPaths 
/// @param numVideos 
/// @param K
/// @return 
int* clusterVideos(const char** videoPaths, int numVideos, int K) {
    // 提取视频特征向量
    std::vector<cv::Mat> features(numVideos);
    cv::parallel_for_(cv::Range(0, numVideos), ParallelExtractFeatures(videoPaths, numVideos, features));

    // 转换数据类型
    cv::Mat featuresMat;
    for (int i = 0; i < numVideos; i++) {
        featuresMat.push_back(features[i]);
    }
    featuresMat.convertTo(featuresMat, CV_32F);

    // 运行kmeans聚类算法
    cv::Mat labels, centers;
    cv::kmeans(featuresMat, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 5, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

     // 将聚类结果存储在动态分配的数组中
    int* result = new int[numVideos];
    for (int i = 0; i < labels.rows; i++) {
        result[i] = labels.at<int>(i);
    }

    return result;
}








