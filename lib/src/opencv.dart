import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:opencv_plugin/opencv_bindings.dart';

const String _libName = "opencv_plugin";

final DynamicLibrary _dylib = Platform.isAndroid ? DynamicLibrary.open('lib$_libName.so') : DynamicLibrary.process();

final OpencvBindings _bindings = OpencvBindings(_dylib);

///获取图片相似度(哈希感知)
Future<double> compareImageSimilarityPhash({required String sourceUrl, required String targetUrl}) async {
  final imagePath1 = sourceUrl.toNativeUtf8();
  final imagePath2 = targetUrl.toNativeUtf8();
  double result = _bindings.compareImageSimilarityPhash(imagePath1.cast(), imagePath2.cast());
  return result;
}

///计算图片模糊度
Future<MediaDetectionResult> calculateImageBlur({required String imageUrl}) async {
  final imagePath = imageUrl.toNativeUtf8();
  double blurValue = _bindings.calculateImageBlur(imagePath.cast());
  return MediaDetectionResult(url: imageUrl, value: blurValue);
}

/// 比较2个视频相似度
Future<double> compareVideoSimilarity({required String sourceUrl, required String targetUrl}) async {
  final imagePath1 = sourceUrl.toNativeUtf8();
  final imagePath2 = targetUrl.toNativeUtf8();
  double result = _bindings.compareVideoSimilarity(imagePath1.cast(), imagePath2.cast());
  return result;
}

/// 以图搜图
/// 通过compute实现
Future<List<MediaDetectionResult>> findSimilarImages(
    {required String originUrl, required List<String> imageList}) async {
  final results = <MediaDetectionResult>[];

  // 定义计算相似度的函数
  Future<MediaDetectionResult?> compareImage(String url) async {
    final imagePath1 = originUrl.toNativeUtf8();
    final imagePath2 = url.toNativeUtf8();
    double similarValue = _bindings.compareImageSimilarityPhash(imagePath1.cast(), imagePath2.cast());
    if (similarValue > 0.8) {
      return MediaDetectionResult(url: url, value: similarValue);
    }
    return null;
  }

  // 使用compute函数执行并发操作
  final futures = imageList.map((url) => compute(compareImage, url)).toList();
  final groups = await Future.wait(futures);
  for (var group in groups) {
    if (group != null && group.url != originUrl) {
      results.add(group);
    }
  }

  return results;
}

/// 以图搜图
/// 通过c++实现
Future<List<MediaDetectionResult>> findSimilarImages2(
    {required String imageUrl, required List<String> imageList}) async {
  Completer<List<MediaDetectionResult>> completer = Completer<List<MediaDetectionResult>>();
  ReceivePort receivePort = ReceivePort();

  List<MediaDetectionResult> onResult() {
    Pointer<Utf8> originUrl = imageUrl.toNativeUtf8();
    Pointer<Pointer<Utf8>> imagePaths = convertStringListToPointer(imageList);
    SimilarityResult result =
        _bindings.imageSearchByPerceptualHash(originUrl.cast(), imagePaths.cast<Pointer<Char>>(), imageList.length);
    return pointerToStringList(result);
  }

  // 启动Isolate并传递参数
  Map<String, dynamic> param = {
    'sendPort': receivePort.sendPort,
    'onResult': onResult(),
  };
  await Isolate.spawn(isolateFunction, param);

  // 监听receivePort并处理来自Isolate的消息
  receivePort.listen((dynamic message) {
    if (message is List<MediaDetectionResult>) {
      // 收到Isolate返回的结果，完成Future
      completer.complete(message);
    }
  });
  return completer.future;
}

void isolateFunction(dynamic parameters) {
  SendPort sendPort = parameters['sendPort'];
  // 在Isolate中执行耗时任务，例如图片处理
  var result = parameters['onResult'];
  // 将结果发送回主Isolate
  sendPort.send(result);
}

/// 以视频搜视频
/// 通过compute实现
Future<List<MediaDetectionResult>> findSimilarVideos(
    {required String originUrl, required List<String> videoList}) async {
  final results = <MediaDetectionResult>[];

  // 定义计算相似度的函数
  Future<MediaDetectionResult?> compareVideo(String url) async {
    final videoPath1 = originUrl.toNativeUtf8();
    final videoPath2 = url.toNativeUtf8();
    double similarValue = _bindings.compareVideoSimilarity(videoPath1.cast(), videoPath2.cast());
    if (similarValue >= 0.8) {
      return MediaDetectionResult(url: url, value: similarValue);
    }
    return null;
  }

  // 使用compute函数执行并发操作
  final futures = videoList.map((url) => compute(compareVideo, url)).toList();
  final groups = await Future.wait(futures);
  for (var group in groups) {
    if (group != null && group.url != originUrl) {
      results.add(group);
    }
  }

  return results;
}

/// 以视频搜视频
Future<List<MediaDetectionResult>> findSimilarVideos2(
    {required String videourl, required List<String> videoUrls}) async {
  Pointer<Utf8> originUrl = videourl.toNativeUtf8();
  Pointer<Pointer<Utf8>> targetUrls = convertStringListToPointer(videoUrls);
  SimilarityResult result =
      _bindings.findSimilarVideos(originUrl.cast(), targetUrls.cast<Pointer<Char>>(), videoUrls.length);
  return pointerToStringList(result);
}

/// 图片分类(聚类算法)
Future<List<int>> clusterImages(List<String> imageList, int K) async {
  Pointer<Pointer<Utf8>> pointerImageList = convertStringListToPointer(imageList);
  Pointer<Int32> pointerResult =
      _bindings.clusterImages(pointerImageList.cast<Pointer<Char>>(), imageList.length, K) as Pointer<Int32>;
  List<int> result = convertPointerToIntList(pointerResult, imageList.length);
  return result;
}

/// 视频分类(聚类算法)
Future<List<int>> clusterVideos(List<String> videoList, int K) async {
  Pointer<Pointer<Utf8>> pointerImageList = convertStringListToPointer(videoList);
  Pointer<Int> pointerResult = _bindings.clusterVideos(pointerImageList.cast<Pointer<Char>>(), videoList.length, K);
  List<int> result = convertPointerToIntList(pointerResult as Pointer<Int32>, videoList.length);
  return result;
}

List<MediaDetectionResult> pointerToStringList(SimilarityResult similarityResult) {
  List<MediaDetectionResult> list = [];
  int length = similarityResult.length;
  if (length < 1) return list;
  var similarImagePaths = similarityResult.imagePaths;
  var similarImageValues = similarityResult.similarities;
  for (var i = 0; i < length; i++) {
    Pointer<Char> pointerChar = similarImagePaths.elementAt(i).value;
    double similarImageValue = similarImageValues.elementAt(i).value;
    String similarImagePath = convertPointerToString(pointerChar);
    MediaDetectionResult imageGroup = MediaDetectionResult(url: similarImagePath, value: similarImageValue);
    list.add(imageGroup);
  }
  return list;
}

String convertPointerToString(Pointer<Char> pointer) {
  if (pointer == nullptr) {
    return '';
  }
  final charPtr = pointer.cast<Utf8>();
  final string = charPtr.toDartString();
  return string;
}

Pointer<Pointer<Utf8>> convertStringListToPointer(List<String> stringList) {
  final length = stringList.length;
  final stringPointerList = calloc<Pointer<Utf8>>(length);

  for (var i = 0; i < length; i++) {
    final string = stringList[i];
    final charPointer = string.toNativeUtf8();
    stringPointerList[i] = charPointer.cast<Utf8>();
  }

  return stringPointerList;
}

// 将 Pointer<Int> 转换为 List<int>
List<int> convertPointerToIntList(Pointer<Int32> pointer, int count) {
  List<int> intList = pointer.asTypedList(count);
  return intList;
}

class MediaDetectionResult {
  final String url;
  final double value;
  MediaDetectionResult({required this.url, required this.value});
}
