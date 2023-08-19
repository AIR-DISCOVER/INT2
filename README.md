<h1 align="center">INT2: Interactive Trajectory Prediction at Intersections [ICCV-2023]</h1>

<p align="center">
<a href="https://int2.cn/"><img  src="docs/icon/Website-INT2.svg" ></a>
<a href="https://arxiv.org/"><img  src="docs/icon/arXiv-Paper.svg" ></a>
<a href="https://arxiv.org/"><img  src="docs/icon/publication-Paper.svg" ></a>
<a href="https://github.com/AIR-DISCOVER/INT2/blob/main/LICENSE"><img  src="docs/icon/license-MIT.svg"></a>
<a href="https://youtu.be/KNkuakDvgVc"><img  src="docs/icon/youtube-demo.svg"></a>
</p>
<h3 align="center">This is the official repository of the paper <a href='"https://arxiv.org'>INT2: Interactive Trajectory Prediction at Intersections.</a></h3>

<h4 align="center"><em><a href="https://github.com/BJHYZJ">Zhijie Yan</a>, 
<a href="https://github.com/Philipflyg">Pengfei Li</a>, 
<a href="">Zheng Fu</a>, 
<a href="#">Shaocong Xu</a>, 
<a href="#">Yongliang Shi</a>,
<a href="https://github.com/cxx226">Xiaoxue Chen</a>, 
<a href="#">Yuhang Zheng</a>, 
<a href="#">Yang Li</a>, 
<a href="https://scholar.google.com/citations?user=NAt3vgcAAAAJ&hl=en">Tianyu Liu</a>, 
<a href="#">Chuxuan Li</a>, 
<a href="#">Nairui Luo</a>, 
<a href="#">Xu Gao</a>, 
<a href="https://air.tsinghua.edu.cn/info/1046/1769.htm">Yilun Chen</a>, 
<a href="https://shi.buaa.edu.cn/wangzuoxu/zh_CN/index.htm">Zuoxu Wang</a>, 
<a href="#">Yifeng Shi</a>, 
<a href="#">Pengfei Huang</a>, 
<a href="https://github.com/0nhc">Zhengxiao Han</a>, 
<a href="https://air.tsinghua.edu.cn/info/1012/1222.htm">Jirui Yuan</a>, 
<a href="https://air.tsinghua.edu.cn/info/1046/1635.htm">Jiangtao Gong</a>, 
<a href="https://air.tsinghua.edu.cn/info/1046/1199.htm">Guyue Zhou</a>, 
<a href="https://hangzhaomit.github.io/">Hang Zhao</a>, 
<a href="https://sites.google.com/view/fromandto">Hao Zhao</a></em></h4>

<h5 align="center">Sponsored by Baidu Inc. through <a href="https://www.apollo.auto/">Apollo</a>-<a href="https://air.tsinghua.edu.cn/en/">AIR</a> Joint Research Center.</h5>

<h4 align="center">
<a href='https://arxiv.org'>arXiv</a> | <a href='https://int2.cn'>INT2 Website</a> | <a href='https://int2.cn/download'>Dataset Download</a> | <a href='https://youtu.be/KNkuakDvgVc'>Video</a>
</h4>

<a align="docs/images/pic1.png"><img src="docs/images/pic1.png"></a>

## Abstract
Motion forecasting is an important component in autonomous driving systems. One of the most challenging problems in motion forecasting is interactive trajectory prediction, whose goal is to jointly forecasts the future trajectories of interacting agents.
To this end, we present a large-scale interactive trajectory prediction dataset named <strong>INT2</strong> for <strong>INT</strong>eractive trajectory prediction at <strong>INT</strong>ersections. 
INT2 includes 612,000 scenes, each lasting 1 minute, containing up to 10,200 hours of data. 
The agent trajectories are auto-labeled by a high-performance offline temporal detection and fusion algorithm, whose quality is further inspected by human judges. Vectorized semantic maps and traffic light information are also included in INT2.
Additionally, the dataset poses an interesting domain mismatch challenge. 
For each intersection, we treat rush-hour and non-rush-hour segments as different domains.
We benchmark the best open-sourced interactive trajectory prediction method on INT2 and Waymo Open Motion, under in-domain and cross-domain settings.

<img src='docs/images/teaser.png'>




## <strong><i>🚀 News</i></strong>

><strong>[coming soon]</strong>: INT2 Motion Prediciton Challenge 2023 and INT2 Interactive Motion Prediction Challenge 2023 in this <a href="https://int2.cn/challenges">challenges page</a>.
>
><strong>[2023-8-9]</strong>: The INT2 Dataset, Benchmark, Visulization toolbox and Interaction filter toolbox are released in this <a href="https://github.com/AIR-DISCOVER/INT2">code-base page</a>.
>
><strong>[2023-8-8]:</strong> The INT2 Dataset Website are open in this <a href="https://int2.cn">website page</a>.
> 


## Getting Started

- **<strong><a href='docs/INSTALL.md'>Installation</a></strong>**
- **<strong><a href='docs/DOWNLOADING.md'>Downloading</a></strong>** 


## Dataset Structure
We processed the data in a data format similar to <a href="https://waymo.com/open/data/motion/">WOMD</a>.
```
INT2_Dataset/
    ├──hdmap
    │      ├──LANE
    │      │     ├──has_traffic_control      # Whether the road is controlled by traffic signal lights.
    │      │     ├──lane_type                # The type of road.
    │      │     ├──turn_direction           # Whether the road have a turning direction.
    │      │     ├──is_intersection          # Whether the road is an intersection.                                          
    │      │     ├──left_neighbor_id         # The ID of the adjacent lane on the left side.                
    │      │     ├──right_neighbor_id        # The ID of the adjacent lane on the right side.                
    │      │     ├──predecessors             # The lane ID that follows the current lane.            
    │      │     ├──successors               # The lane ID reached after crossing the current lane.          
    │      │     ├──centerline               # The centerline of the lane.          
    │      │     ├──left_boundary            # The left boundary of the lane.             
    │      │     └──right_boundary           # The right boundary of the lane.              
    │      ├──STOPLINE                          
    │      │     └──centerline               # The stop line.         
    │      ├──CROSSWALK                         
    │      │     └──polygon                  # The outer boundary line of the crosswalk.       
    │      ├──JUNCTION                         
    │      │     └──polygon                  # The outer boundary line of the junction.       
    │      └──MAP_RANGE                      # The extent of the intersection."    
    │            ├──x_start                            
    │            ├──x_end                           
    │            ├──y_start                            
    │            └──y_end                            
    └──interaciton_scenario                          
           ├──SCENARIO_ID                    # The name of the scene, named as "start time - end time".     
           ├──MAP_ID                         # The hdmap ID corresponding to the scene
           ├──DATA_ACQUISITION_TIME                          
           │     ├──begin                    # The start time of data segment collection, specified precisely to the month, day, hour, minute, and second.     
           │     │      ├──day                          
           │     │      ├──hour                          
           │     │      ├──minute                          
           │     │      ├──second                          
           │     │      └──weekday                          
           │     └──end                      # The end time of data segment collection, specified precisely to the month, day, hour, minute, and second.
           │            ├──day                          
           │            ├──hour                          
           │            ├──minute                          
           │            ├──second                          
           │            └──weekda                          
           ├──TIMESTAMP_SCENARIO             # The complete timestamp on the complete scene.            
           ├──AGENT_INFO                          
           │     ├──object_id                # An integer ID for each object.         
           │     ├──object_type              # An integer type for each object (Vehicle, Pedestrian, or Cyclist)           
           │     ├──object_sub_type          # An integer type for each object (CYCLIST, MOTORCYCLIST, TRICYCLIST et al.)               
           │     └──state                          
           │            ├──position_x        # The x coordinate of each object at each time step.                 
           │            ├──position_y        # The y coordinate of each object at each time step.                 
           │            ├──position_z        # The z coordinate of each object at each time step.                 
           │            ├──theta             # The theta coordinate of each object at each time step.            
           │            ├──velocity_x        # The x component of the object velocity at each time step.                 
           │            ├──velocity_y        # The y component of the object velocity at each time step.                 
           │            ├──length            # The length of each object at each time step.             
           │            ├──width             # The width of each object at each time step.            
           │            ├──height            # The height of each object at each time step.             
           │            └──valid             # A valid flag for all elements of features AGENT_INFO/state/XX. If set to 1, the element is populated with valid data, otherwise it is populated with -1.            
           ├──TRAFFIC_LIGHTS_INFO                          
           │     ├──tf_mapping_lane_id       # Road ID controlled by traffic signal lights.                  
           │     ├──tf_state_valid           # A valid flag for all elements of features TRAFFIC_LIGHTS_INFO/XX. If set to 1, the element is populated with valid data, otherwise it is populated with -1.              
           │     └──tf_state                 # The state of each traffic light at each time step.        
           └──INTERACTION_INFO                          
                 ├──interested_agents        # The ID of interested agents.                 
                 └──interaction_pair_info                         
                        ├──influencer_id     # The ID of influencer agent.                    
                        ├──reactor_id        # The ID of reactor agent.                 
                        ├──influencer_type   # The type of influencer agent.                      
                        ├──reactor_type      # The type of reactor agent.	                   
                        ├──coexistence_time  # The time when both influencer agent and reactor agent coexist.                       
                        └──interaction_time  # The index corresponding to the time during which there is interaction between influencer agent and reactor agent.                       
```

for details, please refer to the <a href="https://int2.cn/doc">dataset documentation</a>.

## Interaction Define Pipeline
We propose an algorithm that enables us to efficiently mine our vast dataset for interactions of research value.
<img src="docs/images/main.png">

INT2 includes interactions between vehicles-vehicles, vechile-cyclist, and vehicle-pedestrian:
<img src='docs/images/interactions_data_examples.png'>

Retrieve the interaction within the scenario dataset:

```
python interaction_filter.py --scenario_path int2_dataset_example/scenario/0/010213250706-010213264206.pickle --output_dir
 int2_dataset_example/interaction_scenario/complete_scenario
```

Split the complete interactive scenario into interactive scenarios with a length of 9.1 seconds:

```
python split_interaction.py --interaction_scenario_path int2_dataset_example/interaction_scenario/complete_scenario/0/010213250706-010213264206.pickle --output_dir int2_dataset_example/interaction_scenario/split_scenario
```

## Visualization
The visualization of the complete interactive scenario:

```
python vis_interaction_scenario.py --scenario_path int2_dataset_example/interaction_scenario/complete_scenario/0/010213250706-010213264206.pickle
```

The results will be saved by default in the output/visualization folder, including an XML file in <a href="https://gitlab.lrz.de/tum-cps/commonroad-scenarios/-/blob/master/documentation/XML_commonRoad_2020a.pdf">CommonRoad format</a>, frame-by-frame visualization images, and a complete video.

The visualization of the interactive scenario segments split into 9.1-second lengths.

```
python vis_split_interaction_scenario.py --scenario_path int2_dataset_example/interaction_scenario/complete_scenario/0/010
213250706-010213264206.pickle
```

multiple xml format files, visualization images, and videos with a length of 9.1 seconds will be saved by default in the 
 ```output/visualization``` folder

## Calculate Collision
We report collision rates so that they function as baselines for potential trajectory generation (instead of trajectory forecasting) applications. Generated trajectories should be as collision-free as possible, under the criteria mentioned above. To calculate collision:

```
python calculate_collision.py --scenario_path int2_dataset_example/scenario/0/010213250706-010213264206.pickle --hdmap_dir int2_dataset_example/hdmap
```
we rasterize both agents and road elements, where agents are represented as rectangles and the road elements are decomposed into a combination of triangles. We use the IOU criteria to detect collisions between agents by computing the overlap between their corresponding rectangles. We also detect collisions between agents and road elements by checking if the rectangles overlap with the road element triangles. The collision rate equals the number of collisions divided by the total number of agent-agent pairs or agent-boundary pairs, you can find it in the <a href="utils/collision_utils.py">code<a>.


## Benchmark
We used <a href="https://github.com/Tsinghua-MARS-Lab/M2I">M2I</a> and <a href="https://github.com/sshaoshuai/MTR">MTR</a> as benchmarks for our dataset.

If you want to use them, please refer to

- <a href="docs/START_M2I.md"><strong>M2I with INT2</strong></a>
- <a href="docs/START_MTR.md"><strong>MTR with INT2</strong></a>


Quantitative results of M2I on our INT2 dataset.
<img src='docs/images/model_results.png'>

Quantitative results of MTR on our INT2 dataset.

Comming soon.


## Citation
If you find this work useful in your research, please consider cite: 

```
TODO.
```

<!-- ```
@article{yan2023int2,
  title={INT2: Interactive Trajectory Prediction at Intersections},
  author={Yan, Zhijie and Li, Pengfei and Fu, Zheng and Xu, Shaocong and Shi, Yongliang and Chen, Xiaoxue and Zheng, Yuhang and Li, Yang and Liu, Tianyu and Li, Chuxuan and Luo, Nairui and Gao, Xu and Chen, Yilun and Wang, Zuoxu and Shi, Yifeng and Huang, Pengfei and Han, Zhengxiao and Yuan, Jirui and Gong, Jiangtao and Zhou, Guyue and Zhao, Hang and Zhao, Hao},
  journal={International Conference on Computer Vision},
  year={2023}
}
``` -->


## Reference
- Waymo open motion dataset: <a href="https://github.com/waymo-research/waymo-open-dataset">https://github.com/waymo-research/waymo-open-dataset</a>
- Commonroad: <a href="https://commonroad.in.tum.de/getting-started">https://commonroad.in.tum.de/getting-started</a>
- M2I: <a href="https://github.com/Tsinghua-MARS-Lab/M2I">https://github.com/Tsinghua-MARS-Lab/M2I</a>
- MTR: <a href="https://github.com/sshaoshuai/MTR">https://github.com/sshaoshuai/MTR</a>