#include <stdint.h>
#include <opencv2/core.hpp>

#ifdef __ANDROID__
#else
#endif

#define ATTRIBUTES extern "C" __attribute__((visibility("default"))) __attribute__((used))

using namespace cv;

ATTRIBUTES char *get_string() {
    return "C++ Code from native.cpp";
}

ATTRIBUTES char *get_opencv_version() {
    return CV_VERSION;
}

