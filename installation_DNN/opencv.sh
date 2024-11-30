# Generic tools:
sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall

# Image I/O libs
sudo apt install libjpeg-dev libpng-dev libtiff-dev

# Video/Audio Libs - FFMPEG, GSTREAMER, x264 ans so on
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
sudo apt install liblapacke-dev

# OpenCore - Adaptive Multi Rate Narrow Band (AMRNB) and Wide Band (AMRWB) speech codec
sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev

# Cameras programming interface libs
sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd ~

# GTK lib for the graphical user functionalites coming from OpenCV highghui module
sudo apt-get install libgtk-3-dev
sudo apt-get install python3-dev python3-pip
sudo apt-get install libtbb-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
