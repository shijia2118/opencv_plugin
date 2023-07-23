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


// 计算图像模糊度的函数
double calculateImageBlur(const char* imagePath);

// 比较图片相似度(哈希感知)
double compareImageSimilarityPhash(const char* imagePath1, const char* imagePath2);

// 以图搜图(感知哈希算法)
SimilarityResult imageSearchByPerceptualHash(const char* targetImagePath, const char** queryImagePaths, int numQueryImages);

// 获取相似视频
SimilarityResult findSimilarVideos(const char* originalVideoPath, const char** videoPaths, int videoCount);

// 图片分类(聚类)
int* clusterImages(const char** imagePaths, int numImages,int K);

// 比较2个视频相似度
double compareVideoSimilarity(const char* videoPath1, const char* videoPath2);

// 对视频进行聚类
int* clusterVideos(const char** videoPaths, int numVideos, int K);

// 比较2个视频相似度
double compareVideoSimilarity(const char* videoPath1, const char* videoPath2);

#ifdef __cplusplus
    
}
#endif

#endif  // IMAGE_SIMILAR_H_
