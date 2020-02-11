apt update && \
source /opt/ros/melodic/setup.bash && \
apt install vim \
tmux \
bash-completion \
git \
htop \
wget \
python3-yaml \
ros-melodic-rosserial-python \
ros-melodic-rosserial-client \
ros-melodic-xacro \
ros-melodic-joy \
ros-melodic-hardware-interface \
ros-melodic-ros-control \
ros-melodic-controller-manager \
ros-melodic-gazebo-ros-control \
ros-melodic-joint-state-controller \
ros-melodic-effort-controllers \
ros-melodic-position-controllers \
ros-melodic-diff-drive-controller \
"ros-melodic-turtlebot3*" \
python3-pip && \
pip3 install keras tensorflow matplotlib gym rospkg catkin_pkg defusedxml opencv-python scikit-image

## Burasi cv_bridge python3 icin derleyecek kisim oluyor
## python 3.6 icin calisacaktir
## cv_bridge or melodic icin 1.13.0 surumu olduguna dikkat ediniz

apt-get install python-catkin-tools python-dev python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-melodic-cv-bridge
mkdir cvbridge_workspace
cd cvbridge_workspace
catkin init
# Instruct catkin to set cmake variables
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
# Instruct catkin to install built packages into install place. It is $cvbridge_workspace/install folder
catkin config --install
# Clone cv_bridge src
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
# Find version of cv_bridge in your repository
apt-cache show ros-melodic-cv-bridge | grep Version
# Checkout right version in git repo. In our case it is 1.13.0
sed -i 's/python3/python-py36/g' src/vision_opencv/cv_bridge/CMakeLists.txt
cd src/vision_opencv/
git checkout 1.13.0
cd ../../
# Build
catkin build cv_bridge
# Extend environment with new package
source install/setup.bash --extend