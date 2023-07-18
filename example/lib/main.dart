import 'package:flutter/material.dart';
import 'dart:async';

import 'package:image_picker/image_picker.dart';
import 'package:opencv_plugin/opencv_plugin.dart' as opencv;
import 'package:opencv_plugin_example/permittion_util.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? imageUrl1;
  String? imageUrl2;

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
    DateTime endTime = DateTime.now();

    Duration difference = endTime.difference(startTime);
    int milliseconds = difference.inMilliseconds;
    int seconds = 0;
    if (milliseconds > 1000) {
      seconds = (milliseconds / 1000).truncate();
      milliseconds = milliseconds % 1000;
    }
    print('>>>>>>>>时间差>>>>>$seconds秒$milliseconds毫秒');
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
      if (files.length == 2) {
        String sourceUrl = files[0].path;
        String targetUrl = files[1].path;
        await getImage(sourceUrl: sourceUrl, targetUrl: targetUrl);
      } else if (files.length > 2) {
        String sourceUrl = files[0].path;
        List<String> paths = [];
        for (var file in files) {
          paths.add(file.path);
        }
        DateTime startTime = DateTime.now();
        var result = await opencv.findSimilarImages(originUrl: sourceUrl, imageList: paths);
        DateTime endTime = DateTime.now();

        Duration difference = endTime.difference(startTime);
        int milliseconds = difference.inMilliseconds;
        int seconds = 0;
        if (milliseconds > 1000) {
          seconds = (milliseconds / 1000).truncate();
          milliseconds = milliseconds % 1000;
        }
        print('>>>>>>>>时间差>>>>>$seconds秒$milliseconds毫秒');

        for (var r in result) {
          print('>>>>>>>>>>>>原图>>>>>>$sourceUrl');
          print('>>>>>>>>>相似图>>>>${r.url}');
          print('>>>>>>>>>相似度>>>>${r.value}');
        }
      }
    }
  }

  Future<void> getBlurImage() async {
    bool result = await PermissionUtils.localGallery(context);
    if (result) {
      final ImagePicker picker = ImagePicker();
      List<XFile> files = await picker.pickMultiImage();
      if (files.isNotEmpty) {
        List<String> imageList = files.map((f) => f.path).toList();
        DateTime startTime = DateTime.now();
        var blurImages = await opencv.calculateImageBlur(imageList: imageList);
        DateTime endTime = DateTime.now();
        Duration difference = endTime.difference(startTime);
        int milliseconds = difference.inMilliseconds;
        int seconds = 0;
        if (milliseconds > 1000) {
          seconds = (milliseconds / 1000).truncate();
          milliseconds = milliseconds % 1000;
        }
        print('>>>>>>>>时间差>>>>>$seconds秒$milliseconds毫秒');
        print('>>>>>>>>>>>>模糊图片长度==${blurImages.length}');
      }
    }
  }

  Future<void> getVideo() async {
    bool result = await PermissionUtils.localGallery(context);
    if (result) {
      final ImagePicker picker = ImagePicker();
      List<XFile>? videos = await picker.pickMultipleMedia();

      String sourceUrl = videos.first.path;
      List<String> paths = [];
      for (var video in videos) {
        paths.add(video.path);
      }
      DateTime startTime = DateTime.now();
      var result = await opencv.findSimilarVideos(originUrl: paths.first, videoList: paths);
      DateTime endTime = DateTime.now();

      Duration difference = endTime.difference(startTime);
      int milliseconds = difference.inMilliseconds;
      int seconds = 0;
      if (milliseconds > 1000) {
        seconds = (milliseconds / 1000).truncate();
        milliseconds = milliseconds % 1000;
      }
      print('>>>>>>>>时间差>>>>>$seconds秒$milliseconds毫秒');
      for (var r in result) {
        print('>>>>>>>>>>>>原视频>>>>>>$sourceUrl');
        print('>>>>>>>>url>>>>>${r.url}');
        print('>>>>>>>>>value>>>>${r.value}');
      }
    }
  }
}
