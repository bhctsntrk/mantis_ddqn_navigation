apt update && \
source /opt/ros/melodic/setup.bash && \
apt install vim \
tmux \
bash-completion \
git \
htop \
wget \
net-tools \
python3-dev \
python3-yaml \
python3-empy \
python3-catkin-pkg-modules \
python3-rospkg-modules \
python3.6-tk \
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
python3-pip

pip3 install --upgrade pip
pip3 install keras 
pip3 install tensorflow-gpu
pip3 install matplotlib
pip3 install gym
pip3 install rospkg
pip3 install catkin_pkg
pip3 install defusedxml
pip3 install opencv-python
pip3 install scikit-image
pip3 install numpy
