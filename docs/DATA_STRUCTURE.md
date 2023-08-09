## Data structure
```
INT2_Dataset/
    ├──hdmap
    │      ├──LANE
    │      │     ├──has_traffic_control
    │      │     ├──lane_type
    │      │     ├──turn_direction
    │      │     ├──is_intersection
    │      │     ├──left_neighbor_id
    │      │     ├──right_neighbor_id
    │      │     ├──predecessors
    │      │     ├──successors
    │      │     ├──centerline
    │      │     ├──left_boundary
    │      │     └──right_boundary
    │      ├──STOPLINE
    │      │     └──centerline
    │      ├──CROSSWALK
    │      │     └──polygon
    │      ├──JUNCTION
    │      │     └──polygon
    │      └──MAP_RANGE 
    │            ├──x_start  
    │            ├──x_end 
    │            ├──y_start  
    │            └──y_end  
    └──interaciton_scenario
           ├──SCENARIO_ID
           ├──MAP_ID
           ├──DATA_ACQUISITION_TIME
           │     ├──begin
           │     │      ├──day
           │     │      ├──hour
           │     │      ├──minute
           │     │      ├──second
           │     │      └──weekday
           │     └──begin
           │            ├──day
           │            ├──hour
           │            ├──minute
           │            ├──second
           │            └──weekda
           ├──TIMESTAMP_SCENARIO
           ├──AGENT_INFO
           │     ├──object_id
           │     ├──object_type
           │     ├──object_sub_type
           │     └──state
           │            ├──position_x
           │            ├──position_y
           │            ├──position_z
           │            ├──theta
           │            ├──velocity_x
           │            ├──velocity_y
           │            ├──length
           │            ├──width
           │            ├──height
           │            └──valid
           ├──TRAFFIC_LIGHTS_INFO
           │     ├──tf_mapping_lane_id
           │     ├──tf_state_valid
           │     └──tf_state
           └──INTERACTION_INFO
                 ├──interested_agents
                 └──interaction_pair_info
                        ├──influencer_id
                        ├──reactor_id
                        ├──influencer_type
                        ├──reactor_type
                        ├──coexistence_time
                        └──interaction_time
```