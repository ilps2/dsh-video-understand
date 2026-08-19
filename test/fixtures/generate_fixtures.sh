#!/bin/bash
# 生成测试夹具脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "生成测试夹具..."

# 生成 3 秒蓝色测试视频（带音频）
ffmpeg -y -f lavfi -i "color=c=blue:s=320x240:r=10:d=3" \
  -f lavfi -i "sine=frequency=440:duration=3" \
  -shortest -c:v libx264 -c:a aac \
  "$SCRIPT_DIR/blue-3s.mp4"

echo "✅ 已生成: $SCRIPT_DIR/blue-3s.mp4"

# 生成 3 秒红色测试视频（无音频）
ffmpeg -y -f lavfi -i "color=c=red:s=320x240:r=10:d=3" \
  -an -c:v libx264 \
  "$SCRIPT_DIR/red-3s-noaudio.mp4"

echo "✅ 已生成: $SCRIPT_DIR/red-3s-noaudio.mp4"

# 生成 5 秒场景切换测试视频
ffmpeg -y -f lavfi -i "color=c=blue:s=320x240:r=10:d=2.5" \
  -f lavfi -i "color=c=green:s=320x240:r=10:d=2.5" \
  -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0[outv]" \
  -map "[outv]" -c:v libx264 \
  "$SCRIPT_DIR/scene-switch-5s.mp4"

echo "✅ 已生成: $SCRIPT_DIR/scene-switch-5s.mp4"

echo "✅ 所有测试夹具生成完成"