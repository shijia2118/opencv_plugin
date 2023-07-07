import 'package:flutter/material.dart';
import 'dart:async';

import 'package:image_picker/image_picker.dart';
import 'package:opencv_plugin/opencv_plugin.dart';
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
    Future.delayed(const Duration(seconds: 3), () async {
      var result = await Opencv.getImageSimilary(
        sourceUrl: sourceUrl, targetUrl: targetUrl,

        // 'https://image.baidu.com/search/albumsdetail?tn=albumsdetail&word=%E6%B8%90%E5%8F%98%E9%A3%8E%E6%A0%BC%E6%8F%92%E7%94%BB&fr=albumslist&album_tab=%E8%AE%BE%E8%AE%A1%E7%B4%A0%E6%9D%90&album_id=409&rn=30',
      );
      print('>>>>>>>>result===$result');
    });
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
      if (files.length == 2) {
        print('>>>>>>file1==${files[0].path}');
        String sourceUrl = files[0].path;
        String targetUrl = files[1].path;
        await getImage(sourceUrl: sourceUrl, targetUrl: targetUrl);
      }
    }
  }
}
