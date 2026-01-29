# RAFDet: Range View Augmented Fusion Network for Point-Based 3D Object Detection
This is an official implementation of RAFDet.

## Framework
<p align="center"> <img src="docs/imgs/backbone_RAFDet.png" width="100%"> </p>
In recent years, point-based methods have achieved promising performance on 3D object detection task. Although effective, they still suffer from the inherent sparsity of point cloud, which makes it challenging to distinguish objects with backgrounds only relying on the view of raw point. To this end, we propose a straightforward yet effective multi-view fusion network termed RAFDet to alleviate this issue. The core idea of our method lies in combining the merits of raw point and its range view to enhance the representation learning for sparse point cloud, thus mitigating the sparsity problem and boosting the detection performance. In particular, we introduce a novel bidirectional attentive fusion module to equip sparse point with interacted fine-grained semantic clues during feature learning process. Then, we devise the range-view augmented fusion module to fully exploit the supplementary relationship between different perspectives with the aim of enhancing original point-view features. In the end, a single-stage detection head is utilized to predict final 3D bounding boxes based on the enhanced semantics. We have evaluated our method on the popular KITTI Dataset, DAIR-V2X Dataset and Waymo Open Dataset. Experimental results on the above three datasets demonstrate the effectiveness and robustness of our approach in terms of detection performance and model complexity. 

## Getting Started
You may refer to [INSTALL.md](docs/INSTALL.md) for the installation of `RAFDet` and [GETTING STARTED.md](docs/GETTING_STARTED.md) to implement this project.

## Visualization

KITTI Val
<p align="center"> <img src="docs/imgs/KITTI_Val_Visualization.png" width="100%"> </p>

KITTI Test
<p align="center"> <img src="docs/imgs/KITTI_Test_Visualization.png" width="100%"> </p>

## Citation
If you find our work useful, please cite:
```bibtex
@ARTICLE{10856371,
  author={Zheng, Zhijie and Huang, Zhicong and Zhao, Jingwen and Lin, Kang and Hu, Haifeng and Chen, Dihu},
  journal={IEEE Transactions on Multimedia}, 
  title={RAFDet: Range View Augmented Fusion Network for Point-Based 3D Object Detection}, 
  year={2025},
  volume={27},
  number={},
  pages={4167-4180},
  keywords={Feature extraction;Point cloud compression;Three-dimensional displays;Object detection;Laser radar;Convolution;Transformers;Semantics;Robustness;Representation learning;3D object detection;LiDAR;range view fusion;transformer},
  doi={10.1109/TMM.2025.3535289}
  }

@ARTICLE{10144609,
  author={Zheng, Zhijie and Huang, Zhicong and Zhao, Jingwen and Hu, Haifeng and Chen, Dihu},
  journal={IEEE Signal Processing Letters}, 
  title={DTSSD: Dual-Channel Transformer-Based Network for Point-Based 3D Object Detection}, 
  year={2023},
  volume={30},
  number={},
  pages={798-802},
  keywords={Feature extraction;Transformers;Three-dimensional displays;Object detection;Point cloud compression;Estimation;Encoding;3D object detection;attention;center estimation;point density;transformer},
  doi={10.1109/LSP.2023.3283468}
}


