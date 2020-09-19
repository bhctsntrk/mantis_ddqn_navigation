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
python3-tk \
pip3 install keras tensorflow matplotlib gym rospkg catkin_pkg defusedxml opencv-python scikit-image
