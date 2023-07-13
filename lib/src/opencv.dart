import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:opencv_plugin/opencv_bindings.dart';

const String _libName = "opencv_plugin";

final DynamicLibrary _dylib = Platform.isAndroid ? DynamicLibrary.open('lib$_libName.so') : DynamicLibrary.process();

final OpencvBindings _bindings = OpencvBindings(_dylib);

///获取图片相似度(直方图)
Future<double> compareImageSimilarityHist({required String sourceUrl, required String targetUrl}) async {
  Completer<double> completer = Completer<double>();
  ReceivePort receivePort = ReceivePort();

  double onResult() {
    final imagePath1 = sourceUrl.toNativeUtf8();
    final imagePath2 = targetUrl.toNativeUtf8();
    double result = _bindings.compareImageSimilarityHist(imagePath1.cast(), imagePath2.cast()).similarity;
    return result;
  }

  // 启动Isolate并传递参数
  Map<String, dynamic> param = {
    'sendPort': receivePort.sendPort,
    'onResult': onResult(),
  };
  Isolate.spawn(isolateFunction, param);

  // 监听receivePort并处理来自Isolate的消息
  receivePort.listen((dynamic message) {
    if (message is double) {
      // 收到Isolate返回的结果，完成Future
      completer.complete(message);
    }
  });
  return completer.future;
}

///获取图片相似度(SSIM)
Future<double> compareImageSimilaritySSIM({required String sourceUrl, required String targetUrl}) async {
  Completer<double> completer = Completer<double>();
  ReceivePort receivePort = ReceivePort();

  double onResult() {
    try {
      final imagePath1 = sourceUrl.toNativeUtf8();
      final imagePath2 = targetUrl.toNativeUtf8();
      double result = _bindings.compareImageSimilaritySSIM(imagePath1.cast(), imagePath2.cast()).similarity;
      return result;
    } catch (e) {
      return 0;
    }
  }

  // 启动Isolate并传递参数
  Map<String, dynamic> param = {
    'sendPort': receivePort.sendPort,
    'onResult': onResult(),
  };
  Isolate.spawn(isolateFunction, param);

  // 监听receivePort并处理来自Isolate的消息
  receivePort.listen((dynamic message) {
    if (message is double) {
      // 收到Isolate返回的结果，完成Future
      completer.complete(message);
    }
  });
  return completer.future;
}

///获取图片相似度(哈希感知)
Future<double> compareImageSimilarityPhash({required String sourceUrl, required String targetUrl}) async {
  Completer<double> completer = Completer<double>();
  ReceivePort receivePort = ReceivePort();

  double onResult() {
    final imagePath1 = sourceUrl.toNativeUtf8();
    final imagePath2 = targetUrl.toNativeUtf8();
    double result = _bindings.compareImageSimilarityPhash(imagePath1.cast(), imagePath2.cast()).similarity;
    return result;
  }

  // 启动Isolate并传递参数
  Map<String, dynamic> param = {
    'sendPort': receivePort.sendPort,
    'onResult': onResult(),
  };
  Isolate.spawn(isolateFunction, param);

  // 监听receivePort并处理来自Isolate的消息
  receivePort.listen((dynamic message) {
    if (message is double) {
      // 收到Isolate返回的结果，完成Future
      completer.complete(message);
    }
  });
  return completer.future;
}

///计算图片模糊度
Future<double> calculateImageBlur({required String imageUrl}) async {
  Completer<double> completer = Completer<double>();
  ReceivePort receivePort = ReceivePort();

  double onResult() {
    Pointer<Utf8> url = imageUrl.toNativeUtf8();
    double blurScore = _bindings.calculateImageBlur(url.cast());
    return blurScore;
  }

  // 启动Isolate并传递参数
  Map<String, dynamic> param = {
    'sendPort': receivePort.sendPort,
    'onResult': onResult(),
  };
  Isolate.spawn(isolateFunction, param);

  // 监听receivePort并处理来自Isolate的消息
  receivePort.listen((dynamic message) {
    if (message is double) {
      // 收到Isolate返回的结果，完成Future
      completer.complete(message);
    }
  });
  return completer.future;
}

/// 以图搜图
Future<List<SimilarImageGroup>> findSimilarImages({required String imageUrl, required List<String> imageList}) async {
  Completer<List<SimilarImageGroup>> completer = Completer<List<SimilarImageGroup>>();
  ReceivePort receivePort = ReceivePort();

  Future<List<SimilarImageGroup>> onResult() async {
    List<SimilarImageGroup> images = [];
    for (var url in imageList) {
      if (url != imageUrl) {
        // double similarValue = await compareImageSimilarityHist(sourceUrl: imageUrl, targetUrl: url);
        // double similarValue = await compareImageSimilarityPhash(sourceUrl: imageUrl, targetUrl: url);
        double similarValue = await compareImageSimilarityPhash(sourceUrl: imageUrl, targetUrl: url);

        if (similarValue > 0.9) {
          images.add(SimilarImageGroup(url: url, value: similarValue));
        }
      }
    }
    return images;
  }

  // 启动Isolate并传递参数
  Map<String, dynamic> param = {
    'sendPort': receivePort.sendPort,
    'onResult': await onResult(),
  };
  await Isolate.spawn(isolateFunction, param);

  // 监听receivePort并处理来自Isolate的消息
  receivePort.listen((dynamic message) {
    if (message is List<SimilarImageGroup>) {
      // 收到Isolate返回的结果，完成Future
      completer.complete(message);
    }
  });
  return completer.future;
}

/// 以图搜图
Future<List<SimilarImageGroup>> searchImageByPerceptualHash(
    {required String imageUrl, required List<String> imageList}) async {
  Completer<List<SimilarImageGroup>> completer = Completer<List<SimilarImageGroup>>();
  ReceivePort receivePort = ReceivePort();

  List<SimilarImageGroup> onResult() {
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
    if (message is List<SimilarImageGroup>) {
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

List<SimilarImageGroup> pointerToStringList(SimilarityResult similarityResult) {
  List<SimilarImageGroup> list = [];
  int length = similarityResult.length;
  if (length < 1) return list;
  var similarImagePaths = similarityResult.imagePaths;
  var similarImageValues = similarityResult.similarities;
  for (var i = 0; i < length; i++) {
    Pointer<Char> pointerChar = similarImagePaths.elementAt(i).value;
    double similarImageValue = similarImageValues.elementAt(i).value;
    String similarImagePath = convertPointerToString(pointerChar);
    SimilarImageGroup imageGroup = SimilarImageGroup(url: similarImagePath, value: similarImageValue);
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

class SimilarImageGroup {
  final String url;
  final double value;
  SimilarImageGroup({required this.url, required this.value});
}
