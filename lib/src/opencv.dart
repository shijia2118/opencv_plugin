import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:opencv_plugin/opencv_bindings.dart';

const String _libName = "opencv_plugin";

final DynamicLibrary _dylib = Platform.isAndroid ? DynamicLibrary.open('lib$_libName.so') : DynamicLibrary.process();

final OpencvBindings _bindings = OpencvBindings(_dylib);

class Opencv {
  ///获取图片相似度
  ///return 相似度double 0 ～ 1
  static Future<double> getImageSimilary({required String sourceUrl, required String targetUrl}) async {
    final imagePath1 = sourceUrl.toNativeUtf8();
    final imagePath2 = targetUrl.toNativeUtf8();
    return _bindings.compareImageSimilarity(imagePath1.cast(), imagePath2.cast());
  }
}
