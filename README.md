# Accurate Low Latency Visual Perception for Autonomous Racing: Challenges Mechanisms and Practical Solutions

This is the Pytorch side code for the accurate low latency visual perception system introduced by *[Kieran Strobel, Sibo Zhu, Raphael Chang, and Skanda Koppula. "Accurate Low Latency Visual Perception for Autonomous Racing: Challenges Mechanisms and Practical Solutions" ](https://static1.squarespace.com/static/5b79970e3c3a53723fab8cfc/t/5dd31c1eb16d2c02ed66408d/1574116397888/Accurate__Low_Latency_Visual_Perception_for_Autonomous_Racing__Challenges__Mechanisms__and_Practical_Solutions_.pdf)*. If you use the code, please cite the paper:

```
BibTex Citation link will be given once the paper has been accepted
```

Abstract

>Autonomous racing provides the opportunity to test safety-critical perception pipelines at their limit. This paper describes the practical challenges and solutions to applying state-of-the-art computer vision algorithms to build a low-latency, high-accuracy perception system for DUT18 Driverless(DUT18D), a 4WD electric race car with podium finishes at all  Formula Driverless competitions for which it raced. The key components of DUT18D include  YOLOv3-based object detection, pose estimation and time synchronization on its dual stereovision/monovision camera setup. We highlight modifications required to adapt perception  CNNs to racing domains, improvements to loss functions used for pose estimation, and methodologies for sub-microsecond camera synchronization among other improvements. We perform  an extensive experimental evaluation of the system, demonstrating its accuracy and low-latency  in real-world racing scenarios.

## CVC-YOLOv3

This is the MIT Driverless Custom implementation of YOLOv3. 

One of our main contributions to vanilla YOLOv3 is the custom data loader we implemented:

Each set of training images from a specific sensor/lens/perspective combination is uniformly rescaled such that their landmark size distributions matched that of the camera system on the vehicle. Each training image was then padded if too small or split up into multiple images if too large.

<img src="https://user-images.githubusercontent.com/22118253/69765465-09e90000-1142-11ea-96b7-370868a0033b.png" width="400">


Our final accuracy metrics for detecting traffic cones on the racing track is:

```
mAP: 89.35%     Recall: 92.77%     Precision: 86.94%      
```





Training dataset including annotation for *Formula Student Driverless* is also open-sourced here(hyperlink)

## RektNet

placeholder