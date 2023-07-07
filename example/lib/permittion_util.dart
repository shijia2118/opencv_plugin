import 'dart:io';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

class PermissionUtils {
  ///权限申请:本地相册
  static Future<bool> localGallery(BuildContext context) async {
    PermissionStatus? status;
    if (Platform.isIOS) {
      status = await Permission.photos.request();
    } else if (Platform.isAndroid) {
      status = await Permission.storage.request();
    }
    return status != null && (status.isGranted || status.isLimited);
  }
}
