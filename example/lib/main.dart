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
    var result = await opencv.compareImageSimilarityPhash(
      sourceUrl: sourceUrl,
      targetUrl: targetUrl,
    );
    print('>>>>>>>>similar===$result');
  }

  Future getImageBlur({required String imageUrl}) async {
    var result = await opencv.calculateImageBlur(imageUrl: imageUrl);
    print('>>>>>>>>blur===$result');
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
      if (files.length == 1) {
        for (int i = 0; i <= 10000; i++) {
          await getImageBlur(imageUrl: files.first.path);
        }
      } else if (files.length == 2) {
        String sourceUrl = files[0].path;
        String targetUrl = files[1].path;
        await getImage(sourceUrl: sourceUrl, targetUrl: targetUrl);
      } else if (files.length > 2) {
        String sourceUrl = files[0].path;
        List<String> paths = [];
        for (var file in files) {
          paths.add(file.path);
        }
        var result = await opencv.findSimilarImages(imageUrl: sourceUrl, imageList: paths);
        for (var r in result) {
          print('>>>>>>>>url>>>>>${r.url}');
          print('>>>>>>>>>value>>>>${r.value}');
        }
      }
    }
  }
}
