// image_process.h
#ifndef IMAGE_PROCESS_H_
#define IMAGE_PROCESS_H_

#ifdef __cplusplus
extern "C" 
{
#endif

// 定义用于存储相似度数据的结构体
typedef struct {
    double similarity;  // 相似度
} ImageSimilarity;

typedef struct  {
    const char** imagePaths; // 相似图片路径数组
    int size;                // 数组大小
} SimilarityArray;

// 比较图片相似度(直方图)
ImageSimilarity compareImageSimilarityHist(const char* image1Path, const char* image2Path);

// 计算图像模糊度的函数
double calculateImageBlur(const char* imagePath);

// 比较图片相似度(SSIM)
ImageSimilarity compareImageSimilaritySSIM(const char* imagePath1, const char* imagePath2);

// 比较图片相似度(哈希感知)
ImageSimilarity compareImageSimilarityPhash(const char* imagePath1, const char* imagePath2);

// 获取相似图片数组
SimilarityArray findSimilarImages(const char* queryImagePath, const char** imagePaths, int numImagePaths);


#ifdef __cplusplus
}
#endif

#endif  // IMAGE_SIMILAR_H_
