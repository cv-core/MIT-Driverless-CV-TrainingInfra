sudo apt-get update
pip2 install onnx==1.1.1 google_compute_engine Pillow pycuda==2018.1.1 numpy opencv-python
pip3 install opencv-python imgaug tqdm gnupg setuptools six typing pyyaml future pandas torch torchvision google-cloud-storage matplotlib pycuda

# Fix a wierd cuda nvcc issue, potentially not needed with future builds
sudo apt-get install gcc-5 g++-5 -y
cd /usr/lib/nvidia-cuda-toolkit/bin
sudo rm gcc g++
sudo ln -s /usr/bin/gcc-5 gcc
sudo ln -s /usr/bin/g++-5 g++

cd ~
git clone git@github.com:cv-core/PerceptionCV.git 
cd ~/PerceptionCV/CVC-YOLOv3
sudo python3 setup.py build develop
echo 'PYTHONPATH=$PYTHONPATH:/home/$USER/PerceptionCV/CVC-YOLOv3/utils/' >> ~/.bashrc
echo 'cd ~/PerceptionCV/CVC-YOLOv3' >> ~/.bashrc
