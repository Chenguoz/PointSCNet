# PointSCNet: Point Cloud Structure and Correlation Learning based on Space Filling Curve guided Sampling
# Abstract
The geometrical structure and internal local region relationship, such as symmetry, regular array, junction, etc., are essential for understanding a 3D shape. This paper proposes a point cloud feature extraction network, namely PointSCNet, to capture the geometrical structure information and local region correlation information of the point cloud. The PointSCNet consists of three main modules: the space filling curve guided sampling module, the information fusion module and the channel-spatial attention module. The space filling curve guided sampling module use Z-order curve coding to sample points which contain geometrical correlation. The information fusion module uses a correlation tensor and a set of skip connections to fuse the structure and correlation information. The channel-spatial attention module enhances the representation of key points and crucial feature channels to refine the network. The proposed PointSCNet is evaluated on shape classification and part segmentation tasks. The experiment results demonstrate that the PointSCNet outperforms or is on par with state-of-the-art methods by learning structure and correlation of point cloud effectively.


![image](images/example.jpg)


## Architecture

![image](images/model.jpg)
