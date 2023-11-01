import os
import math 

import json
import cv2
import numpy as np
import concurrent.futures

from cubercnn import data

dataset_paths_to_json = [
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/ARKitScenes_test.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/Hypersim_test.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/KITTI_test.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/nuScenes_test.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/Objectron_test.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/SUNRGBD_test.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/nuScenes_train.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/nuScenes_val.json",
                         '/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/ARKitScenes_val.json',
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/ARKitScenes_train.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/KITTI_val.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/KITTI_train.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/Objectron_train.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/Objectron_val.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/SUNRGBD_val.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/SUNRGBD_train.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/Hypersim_train.json",
                         "/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/Hypersim_val.json"
                         
                         
                         ]
random_viz=[82832, 22614, 62026, 91811, 95492, 76316, 58799, 78530, 3948, 70911, 91512, 12651, 73805, 78970, 23736, 24024, 59951, 62390, 62115, 30130, 59065, 90928, 26150, 26678, 99332, 83640, 45866, 789, 3849, 10961, 14213, 43600, 75966, 37421, 98746, 50351, 43486, 48008, 78018, 96401, 17299, 20857, 22126, 63541, 54640, 62215, 33034, 16523, 42331, 13884, 74541, 94169, 51382, 55436, 52553, 7212, 64597, 68707, 63137, 56117, 98691, 11949, 24211, 16908, 68500, 57360, 54354, 70290, 80617, 89269, 26942, 77651, 53521, 43046, 8631, 69627, 74967, 39392, 63160, 42062, 89183, 95760, 80356, 84561, 23239, 68976, 52252, 73995, 51944, 53524, 7956, 8702, 51736, 46754, 17578, 25891, 76179, 47618, 55760, 95530]
dataset_paths_to_json = ["/research/cvl-zhiyuan/Txt23d/omni3d/datasets/Omni3D/nuScenes_val.json"]


class Object:
    def __init__(self):
        pass

class BoundingBox:
    def __init__(self, label, center):
        self.label = label
        self.center = center
        

def assign_unique_names(objects):
    object_counts = {}
    for obj in objects:
        label = obj.type
        if label not in object_counts:
            object_counts[label] = 1
        else:
            object_counts[label] += 1
        obj.unique_name = f"{ordinal(object_counts[label])} {label}"

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def describe_position(box1, box2):
    x1, _, z1 = box1.center
    x2, _, z2 = box2.center
    description = f"The {box1.label} is"
    
    if z1 < z2:
        position = "in front of"
    elif z1 > z2:
        position = "behind"
    else:
        position = ""

    if x1 < x2:
Untitled document           position = "to the right of"

    description += f" {position} the {box2.label}."
    return description

def generate_scene_description(objects):
    assign_unique_names(objects)
    
    scene_description = []
    
    for i, obj1 in enumerate(objects):
        box1 = BoundingBox(obj1.unique_name, obj1.t)
        for j, obj2 in enumerate(objects):
            if i != j:
                box2 = BoundingBox(obj2.unique_name, obj2.t)
                scene_description.append(describe_position(box1, box2))
    
    return scene_description



'''def readLabels(label_dir):
    objects = []
    with open(label_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            obj = Object()
            obj.type = data[0]
            if obj.type=="trash":
                obj.type= "trash can"
            #print(obj.type=="chair")
            obj.x1, obj.y1, obj.x2, obj.y2 = map(float, data[1:5])
            obj.t = list(map(float, data[5:8]))
            obj.l, obj.w, obj.h = map(float, data[8:11])
            obj.angle1, obj.angle2 = map(float, data[11:13])
            if obj.type in ['chair', 'table', 'cabinet', "trash can", 'display', 'bookshelf', 'sofa', 'bathtub', 'bed']:  
                objects.append(obj)
    return objects'''

'''def computeBox3D(dim, center, orientation_1, orientation_2):
    heading_angle = np.arctan2(orientation_2, orientation_1)
    R = rotz(heading_angle)
    
    l = dim[0]
    w = dim[1]
    h = dim[2]

    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,w,w,-w,-w,-w,-w]
    z_corners = [h,h,-h,-h,h,h,-h,-h]
    corners_3d = np.dot(R, np.array([x_corners, y_corners, z_corners])/2)
    corners_3d += np.array(center).reshape((3,1))
    
    return corners_3d.T, heading_angle
def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[s, 0, -c],
                     [0, 1, 0], 
                     [c, 0, s]])
def compute_heading_angle(orientation_1, orientation_2):
    return np.atan2(orientation_2, orientation_1)'''
                     
                     
def mat2euler(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    #singular = sy < 1e-6

    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], sy)
    z = math.atan2(R[1, 0], R[0, 0])

    return np.array([x, y, z])

# Calculates Rotation Matrix given euler angles.
    """_summary_

    Returns:
        _type_: _description_
    """# adopted from https://www.learnopencv.com/rotation-matrix-to-euler-angles/

def euler2mat(euler):

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(euler[0]), -math.sin(euler[0])],
                    [0, math.sin(euler[0]), math.cos(euler[0])]
                    ])

    R_y = np.array([[math.cos(euler[1]), 0, math.sin(euler[1])],
    Untitled document
    R_z = np.array([[math.cos(euler[2]), -math.sin(euler[2]), 0],
                    [math.sin(euler[2]), math.cos(euler[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def computeBox3D(center, dim, R):
    euler = mat2euler(np.array(R))
    
    R = euler2mat(euler)
    l, w, h = dim
    z_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, w, w, -w, -w, -w, -w]
    x_corners = [h, h, -h, -h, h, h, -h, -h]
    corners_3d = np.dot(R, np.array([z_corners, y_corners, x_corners])/2)
    corners_3d += np.array(center).reshape((3, 1))
    return corners_3d.T, euler






def normalize_and_round_boxes(box_list):
    #print(len(box_list))
    all_points = np.vstack(box_list)
    min_values = np.min(all_points, axis=0)
    max_values = np.max(all_points, axis=0)
    #print(min_values)
    #print(max_values)
    #aaaa
    scale_factors = 100 / np.max(max_values - min_values)
    normalized_boxes = []
    for box in box_list:
        normalized_box = (box - min_values) * scale_factors
        normalized_boxes.append(normalized_box)
    rounded_boxes = [np.round(box).astype(int) for box in normalized_boxes]
    
    
    return rounded_boxes, scale_factors, min_values

def project_3d_bbox_to_image(K, bbox3D_cam):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    bbox2D_proj = [[int(fx * (X / Z) + cx), int(fy * (Y / Z) + cy)] for X, Y, Z in bbox3D_cam]
    return bbox2D_proj


def draw_bounding_box(image, bbox2D_proj):
    for i in range(0, 4):
        cv2.line(image, tuple(bbox2D_proj[i]), tuple(bbox2D_proj[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.line(image, tuple(bbox2D_proj[i + 4]), tuple(bbox2D_proj[(i + 1) % 4 + 4]), (0, 255, 0), 2)
        cv2.line(image, tuple(bbox2D_proj[i]), tuple(bbox2D_proj[i + 4]), (0, 255, 0), 2)
        
'''class BoundingBox:
    def __init__(self, label, center):
        self.label = label
        self.center = center

def describe_position(box1, box2):
    x1, _, z1 = box1.center
    x2, _, z2 = box2.center
    description = f"The {box1.label} is"
    
    if z1 < z2:
        position = "in front of"
    elif z1 > z2:
        position = "behind"
    else:
        position = ""

    if x1 < x2:
        if position:
            position += " and to the left of"
        else:
            position = "to the left of"
    elif x1 > x2:
        if position:
            position += " and to the right of"
Untitled document
                for obj in objects:
                    coners, heading_angle = computeBox3D(
                        [obj.l, obj.w, obj.h], obj.t, obj.angle1, obj.angle2)
                    scene_data[scene]['boxes'].append(coners)
                

    scale_factors_dic = {}
    for scene, data in scene_data.items():
        #print(data["boxes"][0].shape)
  
        normalized_boxes, scale_factors, min_values = normalize_and_round_boxes(data['boxes'])
    
        scale_factors_dic[scene]={
            "scale_factors": scale_factors,
            "min_values": min_values
        }
        

    
    
    descriptions = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            objects = readLabels(filepath)
            scene = filename.split('_')[0] + '_' + filename.split('_')[1]
            
            if len(objects) > 1:
                language_objects = []
                
                for obj in objects:
                    dim = [obj.l, obj.w, obj.h]
                    center = obj.t
                    scale_factors=scale_factors_dic[scene]["scale_factors"]
                    min_values=scale_factors_dic[scene]["min_values"]
                    #print(scale_factors)
                    #print(min_values)
                    
                    coners, heading_angle = computeBox3D(
                    [obj.l, obj.w, obj.h], obj.t, obj.angle1, obj.angle2)
                    dim = np.round(np.array(dim) * scale_factors).astype(int).tolist()
           
                    center = np.round((np.array(center)-min_values) * scale_factors).astype(int).tolist()
                    
                    dim = np.round(dim, 2).tolist()
                    center =np.round(center,2).tolist() 
            
                    [obj.l, obj.w, obj.h]=dim
                    obj.t = center 
                    #print(obj.t)
                    language_objects.append({
                        "scene_id": filename[:-4],
                        "object_id": filename[:-4] + "_" + obj.type,
                        "label": obj.type,
                        
                        "dimensions": dim,
                        "center": center,
                        "heading_angle": heading_angle.tolist()
                    })
                
                scene_descriptions = generate_scene_description(objects)
                descriptions.append({
                    "id": filename[:-4],
                    "scene_id": filename[:-4],
                    "scene_descriptions": scene_descriptions,
                    "objects": language_objects
                })
    descriptions = sorted(descriptions, key=lambda x: x['id'])
    return descriptions
def transform_box2_to_box1(box2):
    return [
        box2[7],  # A2 -> H2
        box2[4],  # B2 -> E2
        box2[0],  # C2 -> A2
        box2[3],  # D2 -> D2
        box2[6],  # E2 -> G2
        box2[5],  # F2 -> F2
        box2[1],  # G2 -> B2
        box2[2]   # H2 -> C2
    ]
    
    Untitled document boxes for selected images
            all_bbox3D_cam = []
            
            for selected_img_data in selected_imgs:
                annIds = dataset.getAnnIds(imgIds=selected_img_data['id'])
                anns = dataset.loadAnns(annIds)
                #print(anns[2]["bbox3D_cam"])
                for ann in anns:
        
                    if  ann["valid3D"]:
            
                        bbox3D_cam_value = ann["bbox3D_cam"]
            
        
                        all_bbox3D_cam.append(bbox3D_cam_value)
            #print(all_bbox3D_cam)
            # Normalize bounding boxes for the selected images
            if len(all_bbox3D_cam)==0:
                #print("no len")
                return
            normalized_boxes, scale_factors, min_values = normalize_and_round_boxes(all_bbox3D_cam)
            
            
            annIds = dataset.getAnnIds(imgIds=img_data['id'])
            anns = dataset.loadAnns(annIds)
            image_path = os.path.join("/research/cvl-zhiyuan/Txt23d/omni3d/datasets", img_data['file_path'])
            image = cv2.imread(image_path)
            
            #all_bbox3D_cam = [ann["bbox3D_cam"] for ann in anns]
            #normalized_boxes, scale_factors, min_values = normalize_and_round_boxes(all_bbox3D_cam)
            
            min_value = float('inf')  # Set to positive infinity initially
            min_box = None    
            objects = []
            language_objects = []
            
            for index,ann in enumerate(anns):
                if ann["valid3D"] is False:
                    continue
                if ann["behind_camera"]:
                    continue
                if len(anns)==0:
                    continue
                if len(language_objects)>30:
                    continue
                if ann["category_name"] not in [ 'books', 'chair', 'lamp', 'cabinet', 'table', 'car', 'sofa', 'pillow', 'pedestrian', 'window', 'barrier', 'picture', 'truck', 'stationery', 'blinds', 'shelves', 'clothes', 'sink', 'traffic cone', 'desk', 'box', 'door', 'bed', 'television', 'floor mat', 'counter', 'oven', 'bookcase', 'refrigerator', 'shoes', 'fireplace', 'trailer', 'machine', 'mirror', 'cup', 'bathtub', 'stove', 'laptop', 'bottle', 'toilet', 'bus', 'towel', 'bicycle', 'cereal box', 'motorcycle']:
                    continue
                
                bbox3D_cam = ann["bbox3D_cam"]
                R_cam = ann["R_cam"]
                C_center = ann["center_cam"]
                Dimension = ann["dimensions"]
                K = img_data['K']
                
                
                
                #defining the obj 
                obj = Object()
                obj.type = ann["category_name"]
                obj.t = C_center
                obj.l, obj.w, obj.h = Dimension
                
                
                #print(bbox3D_cam)
                coners3d,heading_angle=computeBox3D(C_center,Dimension,R_cam)
                #print(scale_factors)
                #print(min_values)
                #print(Dimension)
                #print(C_center)
                #print(ann["valid3D"])
                
                #dim = np.round(np.array(Dimension) * scale_factors).astype(int).tolist()
                #center = np.round((np.array(C_center)-min_values) * scale_factors).astype(int).tolist()
                #bbox3D_cam = np.round((np.array(bbox3D_cam)-min_values) * scale_factors).astype(int).tolist()
                dim = np.round(Dimension, 2).astype(np.float).tolist()
                center =np.round(C_center,2).astype(np.float).tolist() 
                
                
                [obj.l, obj.w, obj.h]=dim
                obj.t = center
                
                #print(coners3d.tolist())
                current_value = C_center[1] + Dimension[1]/2
                if current_value < min_value:
                    min_value = current_value
                    min_box = index 
                if idx in random_viz:  
                    bbox2D_proj = project_3d_bbox_to_image(np.array(K), np.array(transform_box2_to_box1(coners3d.tolist())))
                    draw_bounding_box(image, bbox2D_proj)
                    #print(bbox3D_cam)
                    #print(transform_box2_to_box1(coners3d.tolist()))
                
                language_objects.append({
                            "scene_id": image_path,
                            "object_id": image_path + "_" + obj.type,
                            "label": obj.type,
                            
                            "dimensions": dim,
                            "center": center,
                            "heading_angle": heading_angle.tolist(),
                            "bbox3D_cam":bbox3D_cam
                        })
                
                #print(language_objects)
                objects.append(obj)
            if idx in random_viz:
                
                cv2.imwrite(f"test_imgs_new/Projected3DBoundingBox_{idx}.jpg", image)
            
            scene_descriptions = generate_scene_description(objects)
            descriptions.append({
                        "id": idx,
                        "scene_id": image_path,
                        "scene_descriptions": scene_descriptions,
                        "objects": language_objects
                    }) 
        
            return
        except Exception as e:
            print(e)
            return
        
        #ground_points=computeground3D(anns[min_box]["center_cam"],anns[min_box]["dimensions"],anns[min_box]["R_cam"])
        #print(ground_points.tolist())
        #print(min_box)
        
        
        
    
    num_threads = 16  # You can adjust this based on your system's capabilities

# Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Use list comprehension to submit tasks to the executor
    # The executor.map() function will parallelize the processing
        executor.map(lambda x: process_image(*x), enumerate(imgs))
    return descriptions
if __name__ == "__main__":
    
    descriptions = process_directory()
    with open('/research/cvl-zhiyuan/Txt23d/omni3d/data_box_final_3.json', 'w') as outfile:
        json.dump(descriptions, outfile, indent=4)