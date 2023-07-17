// image_process.h
#ifndef IMAGE_PROCESS_H_
#define IMAGE_PROCESS_H_
#ifdef __cplusplus

extern "C" {
    #endif

// 定义用于存储相似度数据的结构体
typedef struct {
    double similarity;  // 相似度
} ImageSimilarity;

typedef struct {
    const char** imagePaths; // 相似图片路径数组
    double* similarities;    // 相似度数组
    int length;              // 数组长度
} SimilarityResult;


// 比较图片相似度(直方图)
ImageSimilarity compareImageSimilarityHist(const char* image1Path, const char* image2Path);

// 计算图像模糊度的函数
double calculateImageBlur(const char* imagePath);

// 比较图片相似度(SSIM)
ImageSimilarity compareImageSimilaritySSIM(const char* imagePath1, const char* imagePath2);

// 比较图片相似度(哈希感知)
ImageSimilarity compareImageSimilarityPhash(const char* imagePath1, const char* imagePath2);

// 以图搜图(感知哈希算法)
SimilarityResult imageSearchByPerceptualHash(const char* targetImagePath, const char** queryImagePaths, int numQueryImages);

// 获取相似视频
SimilarityResult findSimilarVideos(const char* originalVideoPath, const char** videoPaths, int videoCount);


#ifdef __cplusplus
    
}
#endif

#endif  // IMAGE_SIMILAR_H_
