# INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Polygon, ShapeGroup
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType,DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State as StateTupleFactory
from commonroad_dc.pycrcc import CollisionChecker
from commonroad_dc.boundary import construction
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object, create_collision_checker
import numpy as np
import Polygon.Utils
from commonroad.geometry.shape import Polygon, ShapeGroup
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario

def construct_boundary_checker(scenario: Scenario) -> CollisionChecker:
    build =  ['section_triangles', 'triangulation']#['simple_triangles']
    boundary = construction.construct(scenario, build)
    road_boundary_shape_list = []
    initial_state = None
    for r in boundary['triangulation'].unpack():
        initial_state = StateTupleFactory(position=np.array([0, 0]), orientation=0.0, time_step=0)
        p = Polygon(np.array(r.vertices()))
        road_boundary_shape_list.append(p)
    
    road_bound = StaticObstacle(obstacle_id=scenario.generate_object_id(),
                                obstacle_type=ObstacleType.ROAD_BOUNDARY,
                                obstacle_shape=ShapeGroup(road_boundary_shape_list),
                                initial_state=initial_state)
    collision_checker = CollisionChecker()
    collision_checker.add_collision_object(create_collision_object(road_bound))
    return collision_checker


def boundary_collision(scenario: Scenario,
                       ob: DynamicObstacle) -> bool:
    collision_checker = construct_boundary_checker(scenario)
    collision_object = create_collision_object(ob.prediction)
    if collision_checker.collide(collision_object):
        return True
    return False

def boundary_collision_(collision_checker,ob):
    collision_object = create_collision_object(ob.prediction)
    if collision_checker.collide(collision_object):
        return True
    return False

def obstacle_collision(scenario: Scenario,
                       ob: DynamicObstacle) -> bool:
    collision_checker = create_collision_checker(scenario)
    collision_object = create_collision_object(ob.prediction)
    if collision_checker.collide(collision_object):
        return True
    return False

def obstacle_collision_(collision_checker,
                       ob: DynamicObstacle) -> bool:
    collision_object = create_collision_object(ob.prediction)
    if collision_checker.collide(collision_object):
        return True
    return False


def scenerio_collision(scenario):
    boundary_collision_checker = construct_boundary_checker(scenario)
    ob_collision_checker = create_collision_checker(scenario)
    ob_collision_num = 0
    boundary_collision_num = 0
    total_num = 0

    ob_list = scenario.obstacles.copy() 
    for ob in ob_list:
        if type(ob) !=  DynamicObstacle:
            continue
        
        #temp = ob.obstacle_shape
        #ob._obstacle_shape = Rectangle(0.1,0.1,center=temp.center,orientation=temp.orientation)
        # print(ob.obstacle_id)
        if boundary_collision_(boundary_collision_checker,ob):
            boundary_collision_num += 1
        
        scenario.remove_obstacle(ob)
        if obstacle_collision(scenario,ob):
            ob_collision_num +=1
        total_num +=1
    ob_collision_rate = ob_collision_num / total_num
    bd_collision_rate = boundary_collision_num / total_num    
    return ob_collision_rate, bd_collision_rate

def boundary_collision(file_path):
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    ob_collision_rate, bd_collision_rate = scenerio_collision(scenario)
    return ob_collision_rate, bd_collision_rate

if __name__ == '__main__':
    file_path = "../output/visualization/8/010213355106-010213364106/010213355106-010213364106.xml"
    ob_collision_rate, bd_collision_rate = boundary_collision(file_path)
    print(f"ob_collision_rate: {ob_collision_rate}, bd_collision_rate: {bd_collision_rate}")

