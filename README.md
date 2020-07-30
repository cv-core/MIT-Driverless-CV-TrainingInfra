# Accurate Low Latency Visual Perception for Autonomous Racing: Challenges Mechanisms and Practical Solutions


<p align="center">
<img src="https://user-images.githubusercontent.com/22118253/70957091-fe06a480-2042-11ea-8c06-0fcc549fc19a.png" width="800">
</p>


This is the Pytorch side code for the accurate low latency visual perception system introduced by *[Kieran Strobel, Sibo Zhu, Raphael Chang, and Skanda Koppula. "Accurate Low Latency Visual Perception for Autonomous Racing: Challenges Mechanisms and Practical Solutions" ](https://static1.squarespace.com/static/5b79970e3c3a53723fab8cfc/t/5dd31c1eb16d2c02ed66408d/1574116397888/Accurate__Low_Latency_Visual_Perception_for_Autonomous_Racing__Challenges__Mechanisms__and_Practical_Solutions_.pdf)*. If you use the code, please cite the paper:

```
@misc{strobel2020accurate,
    title={Accurate, Low-Latency Visual Perception for Autonomous Racing:Challenges, Mechanisms, and Practical Solutions},
    author={Kieran Strobel and Sibo Zhu and Raphael Chang and Skanda Koppula},
    year={2020},
    eprint={2007.13971},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Abstract

>Autonomous racing provides the opportunity to test safety-critical perception pipelines at their limit. This paper describes the practical challenges and solutions to applying state-of-the-art computer vision algorithms to build a low-latency, high-accuracy perception system for DUT18 Driverless(DUT18D), a 4WD electric race car with podium finishes at all  Formula Driverless competitions for which it raced. The key components of DUT18D include  YOLOv3-based object detection, pose estimation and time synchronization on its dual stereovision/monovision camera setup. We highlight modifications required to adapt perception  CNNs to racing domains, improvements to loss functions used for pose estimation, and methodologies for sub-microsecond camera synchronization among other improvements. We perform  an extensive experimental evaluation of the system, demonstrating its accuracy and low-latency  in real-world racing scenarios.

<p align="center">
<img src="https://user-images.githubusercontent.com/22118253/70950893-e2de6980-202f-11ea-9a16-399579926ee5.gif" width="800">
</p>

## CVC-YOLOv3

CVC-YOLOv3 is the MIT Driverless Custom implementation of YOLOv3. 

One of our main contributions to vanilla YOLOv3 is the custom data loader we implemented:

Each set of training images from a specific sensor/lens/perspective combination is uniformly rescaled such that their landmark size distributions matched that of the camera system on the vehicle. Each training image was then padded if too small or split up into multiple images if too large.

<p align="center">
<img src="https://user-images.githubusercontent.com/22118253/69765465-09e90000-1142-11ea-96b7-370868a0033b.png" width="400">
</p>


Our final accuracy metrics for detecting traffic cones on the racing track:

| mAP | Recall | Precision |
|----|----|----|
| 89.35% | 92.77% | 86.94% |

#### CVC-YOLOv3 Dataset with *Formula Student Standard* is open-sourced ***[here](https://storage.cloud.google.com/mit-driverless-open-source/YOLO_Dataset.zip?authuser=1)***

## RektNet

RektNet is the MIT Driverless Custom Key Points Detection Network. 

<p align="center">
<img src="https://user-images.githubusercontent.com/22118253/69765965-fd65a700-1143-11ea-8804-cd1d33f2e824.png" width="800">
</p>

RektNet takes in bounding boxes outputed from CVC-YOLOv3 and outputs seven key points on the traffic cone, which is responsible for depth estimation of traffic cones on the 3D map. 
v
Our final *Depth estimation error VS Distance* graph (The **Monocular** part):

<p align="center">
<img src="https://user-images.githubusercontent.com/22118253/69766182-cc39a680-1144-11ea-9ebc-5708019ba5d2.png" width="600">
</p>

#### RektNet Dataset with *Formula Student Driverless Standard* is open-sourced ***[here](https://storage.cloud.google.com/mit-driverless-open-source/RektNet_Dataset.zip?authuser=1)***

## License

This repository is released under the Apache-2.0 license. See [LICENSE](LICENSE) for additional details.
