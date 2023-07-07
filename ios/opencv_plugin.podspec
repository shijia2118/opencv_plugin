#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint opencv_plugin.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'opencv_plugin'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter project.'
  s.description      = <<-DESC
A new Flutter project.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '11.0'

  s.source_files = [
    'Classes/**/*',
    'native_code/**/*.{cpp,hpp,c,h}',
  ]

  s.preserve_paths = 'opencv_framework/opencv2.framework'
  s.xcconfig = { 'OTHER_LDFLAGS' => '-framework opencv2 -framework'}
  s.vendored_frameworks =  'opencv_framework/opencv2.framework'

  s.frameworks = 'AVFoundation'
  s.library = 'c++'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.pod_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  s.user_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  s.swift_version = '5.0'
end
