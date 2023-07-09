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
    final imagePath1 = sourceUrl.toNativeUtf8();
    final imagePath2 = targetUrl.toNativeUtf8();
    double result = _bindings.compareImageSimilaritySSIM(imagePath1.cast(), imagePath2.cast()).similarity;
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

void isolateFunction(dynamic parameters) {
  SendPort sendPort = parameters['sendPort'];
  // 在Isolate中执行耗时任务，例如图片处理
  var result = parameters['onResult'];
  // 将结果发送回主Isolate
  sendPort.send(result);
}
