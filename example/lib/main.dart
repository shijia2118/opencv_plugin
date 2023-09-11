import 'dart:math';

import 'package:flutter/material.dart';
import 'dart:async';

import 'package:image_picker/image_picker.dart';
import 'package:isolate_manager/isolate_manager.dart';
import 'package:opencv_plugin/opencv_plugin.dart' as opencv;
import 'package:opencv_plugin_example/permittion_util.dart';

void main() {
  runApp(const MyApp());
}

@pragma('vm:entry-point')
Future<void> isoLateBlurImage(String url) async {
  final blurImage = await opencv.calculateImageBlur(imageUrl: url);
  print('>>>>>>>>>>blur==${blurImage.value}');
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? imageUrl1;
  String? imageUrl2;

  final isolateManager = IsolateManager.create(
    concurrent: 5,
    isoLateBlurImage,
    isDebug: true,
  );

  @override
  void initState() {
    super.initState();
  }

  Future getImage({required String sourceUrl, required String targetUrl}) async {
    DateTime startTime = DateTime.now();

    var result = await opencv.compareImageSimilarityPhash(
      sourceUrl: sourceUrl,
      targetUrl: targetUrl,
    );
    printDateDiff(startTime);
    print('>>>>>>>>similar===$result');
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Native Packages'),
        ),
        body: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(10),
            child: Column(
              children: [
                FilledButton(onPressed: getLostData, child: const Text('选择图片')),
                FilledButton(onPressed: getBlurImage, child: const Text('模糊图片')),
                FilledButton(onPressed: getVideo, child: const Text('选择视频')),
                FilledButton(onPressed: clusterImages, child: const Text('图片聚类')),
                FilledButton(onPressed: clusterVideos, child: const Text('视频聚类')),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Future<void> getLostData() async {
    bool result = await PermissionUtils.localGallery(context);
    if (result) {
      final ImagePicker picker = ImagePicker();
      List<XFile>? files = await picker.pickMultiImage();
      if (files.length == 10) {
        String sourceUrl = files[0].path;
        List<String> paths = [];
        for (int i = 0; i < 1000; i++) {
          XFile file = files[i % 10];
          paths.add(file.path);
        }
        DateTime startTime = DateTime.now();
        var result = await opencv.findSimilarImages(originUrl: sourceUrl, imageList: paths);
        printDateDiff(startTime);
        for (var r in result) {
          print('>>>>>>>>>>>>原图>>>>>>$sourceUrl');
          print('>>>>>>>>>相似图>>>>${r.url}');
          print('>>>>>>>>>相似度>>>>${r.value}');
        }
      } else if (files.length > 2) {
        String sourceUrl = files[0].path;
        List<String> paths = [];
        for (var file in files) {
          paths.add(file.path);
        }
        DateTime startTime = DateTime.now();
        var result = await opencv.findSimilarImages(originUrl: sourceUrl, imageList: paths);
        printDateDiff(startTime);
        for (var r in result) {
          print('>>>>>>>>>>>>原图>>>>>>$sourceUrl');
          print('>>>>>>>>>相似图>>>>${r.url}');
          print('>>>>>>>>>相似度>>>>${r.value}');
        }
      }
    }
  }

  ///计算图片模糊度
  Future<void> getBlurImage() async {
    bool result = await PermissionUtils.localGallery(context);
    if (result) {
      final ImagePicker picker = ImagePicker();
      List<XFile> files = await picker.pickMultiImage();
      if (files.length == 10) {
        List<String> paths = [];
        for (int i = 0; i < 1000; i++) {
          XFile file = files[i % 10];
          paths.add(file.path);
        }
        DateTime startTime = DateTime.now();
        printDateDiff(startTime);
        for (var url in paths) {
          isolateManager.compute(url);
        }
      }
    }
  }

  ///将图片数组聚类
  Future<void> clusterImages() async {
    bool result = await PermissionUtils.localGallery(context);
    if (result) {
      final ImagePicker picker = ImagePicker();
      List<XFile> files = await picker.pickMultiImage();
      if (files.length == 10) {
        List<String> paths = [];
        for (int i = 0; i < 1000; i++) {
          XFile file = files[i % 10];
          paths.add(file.path);
        }
        DateTime startTime = DateTime.now();
        int k = sqrt(paths.length).toInt();
        var result = await opencv.clusterImages(paths, k);
        printDateDiff(startTime);
        print('>>>>>>>>>>>>分类结果>>>>>>$result');
      }
    }
  }

  ///将视频数组聚类
  Future<void> clusterVideos() async {
    bool result = await PermissionUtils.localGallery(context);
    if (result) {
      final ImagePicker picker = ImagePicker();
      List<XFile> files = await picker.pickMultipleMedia();
      if (files.length > 2) {
        List<String> paths = [];
        for (int i = 0; i < files.length; i++) {
          XFile file = files[i % 10];
          paths.add(file.path);
        }
        DateTime startTime = DateTime.now();
        int k = sqrt(paths.length).toInt();
        var result = await opencv.clusterVideos(paths, k);
        printDateDiff(startTime);
        print('>>>>>>>>>>>>k==$k');
        print('>>>>>>>>>>>>分类结果>>>>>>$result');
      }
    }
  }

  Future<void> getVideo() async {
    bool result = await PermissionUtils.localGallery(context);
    if (result) {
      final ImagePicker picker = ImagePicker();
      List<XFile>? videos = await picker.pickMultipleMedia();

      if (videos.length == 2) {
        List<String> paths = [];
        for (var video in videos) {
          paths.add(video.path);
        }
        DateTime startTime = DateTime.now();
        var similar = await opencv.compareVideoSimilarity(sourceUrl: paths.first, targetUrl: paths[1]);
        printDateDiff(startTime);
        print('>>>>>>>>>>>>视频相似度>>>>>>$similar');
      } else {
        String sourceUrl = videos.first.path;
        List<String> paths = [];
        for (var video in videos) {
          paths.add(video.path);
        }
        DateTime startTime = DateTime.now();
        var result = await opencv.findSimilarVideos(originUrl: paths.first, videoList: paths);
        printDateDiff(startTime);
        for (var r in result) {
          print('>>>>>>>>>>>>原视频>>>>>>$sourceUrl');
          print('>>>>>>>>url>>>>>${r.url}');
          print('>>>>>>>>>value>>>>${r.value}');
        }
      }
    }
  }

  ///打印时间差
  void printDateDiff(DateTime time) {
    DateTime endTime = DateTime.now();
    Duration difference = endTime.difference(time);
    int milliseconds = difference.inMilliseconds;
    int seconds = 0;
    if (milliseconds > 1000) {
      seconds = (milliseconds / 1000).truncate();
      milliseconds = milliseconds % 1000;
    }
    print('>>>>>>>>时间差>>>>>$seconds秒$milliseconds毫秒');
  }
}
