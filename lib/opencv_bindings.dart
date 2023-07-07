import 'dart:ffi' as ffi;

class OpencvBindings {
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName) _lookup;

  OpencvBindings(ffi.DynamicLibrary dynamicLibrary) : _lookup = dynamicLibrary.lookup;

  OpencvBindings.fromLookup(ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName) lookup)
      : _lookup = lookup;

  /// 比较图片相似度的函数
  double compareImageSimilarity(
    ffi.Pointer<ffi.Char> image1Path,
    ffi.Pointer<ffi.Char> image2Path,
  ) {
    return _compareImageSimilarity(
      image1Path,
      image2Path,
    ).similarity;
  }

  late final _compareImageSimilarityPtr =
      _lookup<ffi.NativeFunction<ImageSimilarity Function(ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Char>)>>(
          'compareImageSimilarity');
  late final _compareImageSimilarity =
      _compareImageSimilarityPtr.asFunction<ImageSimilarity Function(ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Char>)>();
}

/// 定义用于存储相似度数据的结构体
final class ImageSimilarity extends ffi.Struct {
  /// 相似度
  @ffi.Double()
  external double similarity;
}
