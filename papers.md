```mermaid
graph TB

```

# Vision

- **Real-time human pose recognition in parts from single depth images**.
Jamie Shotton, Andrew Fitzgibbon, Mat Cook, Toby Sharp, Mark Finocchio, Richard Moore, Alex Kipman, and Andrew Blake.
**CVPR 2011 best paper**.
([pdf](https://www.asc.ohio-state.edu/statistics/dmsl/BodyPartRecognition.pdf)
[project](https://www.microsoft.com/en-us/research/publication/real-time-human-pose-recognition-in-parts-from-a-single-depth-image/)
[TPAMI video](https://www.youtube.com/watch?v=ZXI6gko7kG4)
)
(Citations: **4237**)

pros:

1. Encode temporal consistency with the diff between depth values of neighboring pixels.
2. No temporal information is required.
3. Prevents overfitting with large datasets and bagging.
4. Depth invariance by depth value normalization.
5. Meanshift runs in parallel.
6. Real-time.
7. The body parsing overhead is beneficial to downstream tasks, e.g., tracking init, recovery from failure.
8. Random forest provides some interpretability.
9. Use synthetic datasets to easily increase dataset size.
10. Handles some self-occlusion.
11. Ops (diff between depths) easily coded with GPU.

cons:

1. Doesn't report performance when people are close to background wall (small diff between fore-/back-ground). 
2. The body parsing (segmentation) overhead makes the process slow.
3. Fig 7 shows there's still a huge gap between predicted human parts and g.t.
4. 3D feature is interpretable, can be visualized and satisfies geometric properties (can apply rotation matrix to)

- **Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation**.
Rhodin, Helge, Mathieu Salzmann, and Pascal Fua.
**ECCV2018**.
([pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Helge_Rhodin_Unsupervised_Geometry-Aware_Representation_ECCV_2018_paper.pdf))
(Citations:194)

pros:
1. decouple the learning of appearance, geometry, background
2. may be applied to general rigid objects
3. with the same idea of exchanging features, can decouple more, e.g., body shape, expression, hair color, etc, as long as paired images are provided.

cons:
1. comparison is not fair, has been unsupervisedly trained on many images (didn't report)
2. Hard to generalize to image dataset. For single image dataset, finding paired images is difficult.
3. Suitable to fixed camera position, so we can get the background.
4. appearance is not well captured
5. 3D pose estimation is still semi-supervised learning instead of unsupervised learning
6. geometry reconstruction is limited by training data: pitch is not well learned


- **Nerf: Representing scenes as neural radiance fields for view synthesis**.
Mildenhall, Ben, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng.
**ECCV2020**.
([pdf](http://openaccess.thecvf.com/content/CVPR2021/papers/Pumarola_D-NeRF_Neural_Radiance_Fields_for_Dynamic_Scenes_CVPR_2021_paper.pdf))
(Citations:761)

techniques:

1. volume density
2. volume rendering

pros 

1. simple and powerful MLP
2. sampling technique is efficient
3. position encoding to capture higher frequency in images, prevent over smoothing
4. model is small (5MB)
5. doesn't require 3D model as label
6. well capture specularities, view dependent
7. can edit by adding objects (volumn rendering, ray tracing)

cons

1. represent multiple scenes with 1 network
2. requires a large amount of images to prevent possible occlusion (can solve with GAN?)
3. convergence speed is slow (at least 100k iterations and 1 day on V100)
4. cannot finetune scene objects, just a naive representation.

- **Scene representation networks: Continuous 3d-structure-aware neural scene representations**.
Sitzmann, Vincent, Michael Zollhöfer, and Gordon Wetzstein.
**NeurIPS2019**.
([pdf](https://proceedings.neurips.cc/paper/2019/file/b5dc4e5d9b495d0196f61d45b26ef33e-Paper.pdf)
[project](https://www.vincentsitzmann.com/srns/)
[tutorial](https://www.youtube.com/watch?v=Or9J-DCDGko))
(Citations: 415)

pros:

1. LSTM differentiable ray marching
2. hypernetwork encodes the structure for the same kind of objects, and few-shot leaning for new scenes
3. hypernetwork, encode multiple scenes with the same network
4. scene features may benefit downstream tasks, e.g., 3D segmentation/localization/object detection
   
cons:

1. didn't use positional encoding and hard to capture high frequency signals
2. hard to guarantee for different camera pose and the same ray direction, the intersection position will be the same
3. didn't provide a way of disentangling latent feature
4. L2 loss may make the visual effect over-smooth
5. stuck with small holes (chair) which are hard to perform ray matching

- **Barf: Bundle-adjusting neural radiance fields**.
Lin, Chen-Hsuan, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey.
**ICCV2021**.
([pdf](http://openaccess.thecvf.com/content/ICCV2021/papers/Lin_BARF_Bundle-Adjusting_Neural_Radiance_Fields_ICCV_2021_paper.pdf))
(Citations: 27)

pros:

1. doesn't require accurate camera parameters
2. the mask idea supports coarse to fine training, and we don't need to change the structure of model
3. good transition from 2D example to 3D case

cons:

1. doesn't speed up training
2. maybe can customize the gradient update function (multiplying true gradient with regularization terms), bc when alpha is large, the gradient is still large
3. with camera poses, can it improve?

- **nerfies: deformable neural radiance fields.**.
park, keunhong, utkarsh sinha, jonathan t. barron, sofien bouaziz, dan b. goldman, steven m. seitz, and ricardo martin-brualla.
**ICCV2021**.
([pdf](https://arxiv.org/pdf/2011.12948)
[project](https://nerfies.github.io/))
(Citations: 159)

pros:

1. use deformation field to find a canonical view, which aligns the same object with minor movement
2. background regularization forces the background to be static
3. the robust error function reduces the influence of outliers because gradient of large values is small
4. training data is easily accessible

cons:

1. training time is long: a week on 8 V100
2. The background regularization cannot model dynamic background
3. cannot handle topological transformation
4. quality depends on SfM
5. visualization of latent code
6. fast motion

- **Deepsdf: Learning continuous signed distance functions for shape representation.**.
Park, Jeong Joon, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove.
**CVPR2019**.
([pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf))
(Citations:984)

pros:

1. continuous signed distance field is a good representation (easy to calc normal)
2. autodecoder prevents bottleneck of encoder
3. because it defines closed surfaces, can apply ray-tracing to render the shape

cons:

1. reference time is long 9s, 1e3 slower than AtlasNet (voxel hashing?)
2. hard to get real-world perfect data (train with synthetics dataset ShapeNet)
3. hard to edit the shape
4. mostly show symmetric and smooth shape, didn't show complex shapes e.g. tree with leaves, gloves, clothes

- **Grasping field: Learning implicit representations for human grasps**.
Karunratanakul, Korrawe, Jinlong Yang, Yan Zhang, Michael J. Black, Krikamol Muandet, and Siyu Tang.
**3DV2020**.
([pdf](https://arxiv.org/pdf/2008.04451.pdf?ref=https://githubhelp.com)
[code](https://github.com/korrawe/grasping_field))
(Citations:38)

techniques:
1. [chamfer distance](https://www.youtube.com/watch?v=P4IyrsWicfs)

common: 

1. delta in SDF
2. feed feature to deep layers

pros 

1. distance field representation, enforce physical constraints, e.g. no inter-penetration
2. 3d reconstruction from single view rgb image
3. sdf for motiple objects

cons

1. In real applications, objects may not be scanned 360 degrees. May be only partially scanned
2. hard to extend to different end effectors
3. doesn't consider material of objects, e.g., knife
4. why do we want to learn the static grasp

. "" In European conference on computer vision, pp. 536-551. Springer, Cham, 2014.

- **Learning 6d object pose estimation using 3d object coordinates**.
Brachmann, Eric, Alexander Krull, Frank Michel, Stefan Gumhold, Jamie Shotton, and Carsten Rother.
**ECCV2014**.
([pdf](https://link.springer.com/content/pdf/10.1007/978-3-319-10605-2_35.pdf))
(Citations:567)

terms

1. [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm) get optimal rotation mat between two sets of points

pros

1. learn class and location together, makes use of common image features
2. robust, use top 3 hypothesis and ave acc is 96%
3. use depth, robust against lighting conditions
4. depth, coordinates, and object class are good loss functions
5. robust against occlusion

cons

1. need 3D model of the object
2. |class| * space is hard to scale

- **inerf: Inverting neural radiance fields for pose estimation**.
Yen-Chen, Lin, Pete Florence, Jonathan T. Barron, Alberto Rodriguez, Phillip Isola, and Tsung-Yi Lin.
**IROS2021**.
([pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636708&casa_token=TM8xMncdrzUAAAAA:T2JSYhRXeLBSiSG48T1YQQRlA0LCSkUFVV-Hduwna24YEQmUP9beSUxwqx4pRD_2QI5g_vw4GA&tag=1)
[project](https://yenchenlin.me/inerf/))
(Citations:46)

terms

1. [Aperture problem](https://www.youtube.com/watch?v=vVGorOxMh8w)
2. [morphological](https://www.youtube.com/watch?v=d1we_yqUASg)

pros 

1. paper is easy to read, apply a simple idea to solve an important problem
2. using dilation and interest point to find interest region is interesting
3. possiblity to perform semi-supervised learning with few labeled data to train nerf
4. can be applied to different objects within the same category

cons

1. not robust against lighting and occlusion
2. training takes too long, not real-time
3. a good result still relies on a good nerf, whic means labeled data
4. is using 5 images for comparison enough?

# robotics
- **learning agile robotic locomotion skills by imitating animals**.
peng, xue bin, erwin coumans, tingnan zhang, tsang-wei lee, jie tan, and sergey levine.
**rss2020 best paper**.
([pdf](https://arxiv.org/pdf/2004.00784.pdf)
[project](https://xbpeng.github.io/projects/robotic_imitation/index.html)
[code](https://github.com/google-research/motion_imitation)
[video](https://www.youtube.com/watch?v=lkyh6uucwry))
(citations: 122)

techniques:

1. inverse-kinematics: 
   [live coding](https://www.youtube.com/watch?v=hbgdqyy8biw)
   [talk](https://graphics.cs.wisc.edu/gleicherassets/talks/1998_07_retarget-both.pdf)
   [video](https://www.youtube.com/watch?v=vn-vvzmggec)
   [paper](https://sci-hub.ru/https://dl.acm.org/doi/pdf/10.1145/280814.280820)
2.  generalized coordinates
3.  root position

pros:

1.  with the author's prior [work](https://xbpeng.github.io/projects/sfv/index.html) 
   "Reinforcement Learning of Physical Skills from Videos", we can leverage videos in-the-wild.
2. Inverse kinemetics is applicable to collecting human motion data to train human-like 2-foot robots.
3. Honestly state the shortcomings, e.g., not able to learn more dynamic behaviors.

cons:

1. How to interact with changing environment? Searching $z$ to adapt to environment works for a certain kind of environment. What if the robot goes from indoor to outdoor?
3. Not able to learn more dynamic behaviors such as large jumps and runs.
4. The behaviors learned by our policies are currently not
    as stable as the best manually-designed controllers.
5. Accessing training data is expensive. Pets may not do the actions that  researchers want (e.g., backflip). Hard to capture any animals, e.g., snakes, birds. Cannot learn from videos in-the-wild.
6. Lack of creativity? Combination of prior work.
7. Inverse kinemetics is only applicable to robots with similar structures. Hard to collect data for robots with fewer or more legs.
8. In reward function, $\exp(-x)$, when $x$ is large, derivative is small. Maybe leads to slow convergence?
9. From Robust to Adaptive (Before), there is not significant improvement? Does mean finetuning based on environment is more important than training with random environment? Is the encoder necessary? 

- **Nothing but geometric constraints: A model-free method for articulated object pose estimation**.
Liu, Qihao, Weichao Qiu, Weiyao Wang, Gregory D. Hager, and Alan L. Yuille.
**arXiv2020**.
([pdf](https://arxiv.org/pdf/2012.00088.pdf))
(Citations:7)

pipeline:

1. optic flow
2. estimate R
3. refine optic flow
4. refine transformation
5. estimate final transformation

techniques

1. epipolar geometry:
[course](https://www.youtube.com/watch?v=6kpBqfgSPRc&list=PLELvrS4qfPZ7h_pgw-_X7D1ijScmsk965&index=2)
2. RANSAC
[paper](https://dl.acm.org/doi/pdf/10.1145/358669.358692?casa_token=oNOGfxwunJIAAAAA:qYYOVTu9vOFmDqwdCO73Isrh6pf0aeGFD1czZLIkeo91xF7ikQQKx_zsqmRqmxW_ELJYeNnVh9kJ)
[video](https://www.youtube.com/watch?v=9D5rrtCC_E0)
[video2](https://www.youtube.com/watch?v=Cu1f6vpEilg)
3. rotation is orthogonal
[video](https://www.youtube.com/watch?v=arSFML-2_Os)

pros:

1. design epipolar constraint as a loss, and background pixels (where the constraint can be applied) dominate the image
2. use geometric (image coordinate) and pixel value constraints.
3. many optimization: optimization optic flow with epipolar constraint; optimize rotation matrix with depth estimation.
4. Table1: generalizable to other objects
5. To transfer to differnet objects, only need to retrain RANSAC and EM for part segmentation.
6. Table3: robust to depth estimation

cons: 

1. epipolar constraint assumes moderate change between frames -- sample rate should be large enough to train the model -- large sample rates may lead to overfitting
2. epipolar loss fails when camera is fixed
3. errors accumulate as video length increases
4. photometric difference loss is not robust to lignting condition (shadow)
5. in 3.3, authors say the camera is still
6. R is calculated from optic flow, to minimize the epipolar loss. Then use R to update optic flow. (may work because if optic flow not good, R not good, loss large; if optic flow good, loss small)
7. not clear whether to estimate joint angle or change of joint angle
8. didn't talk about what if depth estimation is not good

- **imap: Implicit mapping and positioning in real-time.**.
Sucar, Edgar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison.
**ICCV2021**.
([pdf](http://openaccess.thecvf.com/content/ICCV2021/papers/Sucar_iMAP_Implicit_Mapping_and_Positioning_in_Real-Time_ICCV_2021_paper.pdf))
(Citations:21)

pros:

1. real-time, online
2. efficient pixel sampling and key-frame sampling
3. fill in the holes
4. model size is small
5. differentialble depth estimation

cons:

1. texture doesn't look sharp, over-smooth
2. hole completion of complicated scenes is not that good, the interpolation is over-smooth (room-1, Fig7)
3. experiments didn't say robustness against depth estimation

- **3d neural scene representations for visuomotor control**.
Li, Yunzhu, Shuang Li, Vincent Sitzmann, Pulkit Agrawal, and Antonio Torralba.
**CoRL2021**.
([pdf](https://proceedings.mlr.press/v164/li22a/li22a.pdf)
[project](https://3d-representation-learning.github.io/nerf-dy/))

generalization:

1. image
2. action
3. scene

(Citations: 5)

pros:

1. triplet loss enforces state representation to be robust against camera position
2. 2-step makes training more stable
3. self supervision in auto-decoding for state representation optimization
4. learns long-term actions: pour the water and tilt the bottle back


cons:

1. lack of high frequency info for futural frames: L2 loss, averaging image features
2. What's the difference between training and goal image?
3. 1 network per action
4. memory of state in long sequence?
5. do we need 2-step?


- **Parallel tracking and mapping for small AR workspaces**.
Klein, Georg, and David Murray.
**ISMAR2007**.
([pdf](https://ieeexplore.ieee.org/iel5/4509974/4538818/04538852.pdf))
(Citations: 4963)

pros

1. recover from failure
2. no NN and promising result, stable robust
3. robust against occlusion
4. parallel design
5. image pyramid: robust against size 

shaky camera
real time 
SDE tricks
parallel 
games
single camera

painting

cons:

cannot control by multiple people
repeated texture
only main plain, 
space is limited 
now we have depth
calibration
no interaction


geometry/object detection, property of the object, semantic
bridge lightweight, fast, robust
interaction



1. corner detector, may fail due to motion blur
2. not robust against repeated texture
3. not applicable when scene is not static

- **Learning Continuous Environment Fields via Implicit Functions**.
Li, Xueting, Shalini De Mello, Xiaolong Wang, Ming-Hsuan Yang, Jan Kautz, and Sifei Liu.
**ICLR2022Poster**.
([pdf](https://arxiv.org/pdf/2111.13997))
(Citations:0)

pros
1. model scene by reaching distance. can use gradient to update the position
2. to make the next decision, only need to explore a small neighbor of the current location


cons
1. not symmetric as for current location and goal. (f(x, y) = f(y, x))
2. in 2d case, feature is not robust against color (we only want structure). But not a problem for 3D point cloud
3. need more interaction between walkable area and reaching distance field.  
   | target       grad(x->y) > grad(x->z)
 x | y
 z |
   wall

---------
4. Assume bird view. What if we only have image from the camera on the robot?


- **Objectfolder: A dataset of objects with implicit visual, auditory, and tactile representations.**.
Gao, Ruohan, Yen-Yu Chang, Shivani Mall, Li Fei-Fei, and Jiajun Wu.
**CoRL2021**.
([pdf](https://arxiv.org/pdf/2109.07991)
[project](https://ai.stanford.edu/~rhgao/objectfolder/))
(Citations:4)

pros

1. Multisensory dataset, train a model to preceive the world through vision, audio, and touch.
2. represent objects by NN, dense query
3. audio for object recognition is interesting
4. similarity between features of different modalities is interesting

cons

1. small dataset
2. why not mesh?
3. didn't talk about image rendering time
4. most are household objects like bowls, chair, desks

. "" arXiv preprint arXiv:2104.01542 (2021).
- **Synergies between affordance and geometry: 6-dof grasp detection via implicit representations**.
Jiang, Zhenyu, Yifeng Zhu, Maxwell Svetlik, Kuan Fang, and Yuke Zhu.
**arxiv2021**.
([pdf](http://www.google.com/url?q=http%3A%2F%2Farxiv.org%2Fabs%2F2104.01542&sa=D&sntz=1&usg=AOvVaw22MAuYVtUJOze1hcMq8xGA)
[project](https://sites.google.com/view/rpl-giga2021))
(Citations: 15)

pros:

1. modeling the input and output: output is the orientation given a position query, instead of (position, orientation) pair
2. also interesting to map from grid to 3 canonical planes
3. interesting to inpaint the planes by UNet
4. also interesting to query grid feature from canonical planes

cons:

1. how does arm move to avoid collision
2. why don't just trilinear interpolation
3. doesn't update as the arm moves

# Template
- ** **.
.
** **.
([pdf]())
(Citations: )

<div class="boxed" style="background-color:yellow; width:20px">&nbsp</div> 
contribution
<div class="boxed" style="background-color:green; width:20px">&nbsp</div>
method
<div class="boxed" style="background-color:blue; width:20px">&nbsp</div>
result
<div class="boxed" style="background-color:pink; width:20px">&nbsp</div>
settings
<div class="boxed" style="background-color:red; width:20px">&nbsp</div>
question
