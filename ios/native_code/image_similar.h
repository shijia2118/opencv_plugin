// image_similar.h
#ifndef IMAGE_SIMILAR_H_
#define IMAGE_SIMILAR_H_

#ifdef __cplusplus
extern "C" 
{
#endif

// 定义用于存储相似度数据的结构体
typedef struct {
    double similarity;  // 相似度
} ImageSimilarity;

// 比较图片相似度的函数
ImageSimilarity compareImageSimilarity(const char* image1Path, const char* image2Path);

#ifdef __cplusplus
}
#endif

#endif  // IMAGE_SIMILAR_H_
