import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops

from multiprocessing import shared_memory

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def droid_visualization(ht, wd, buffer_size, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    #roid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.02

    torch.cuda.synchronize()

    # shared memory 접근, 공유 메모리가 없으면 오류 발생
    try:
        shm_poses = shared_memory.SharedMemory(name='shm_poses')
        shm_intrinsics = shared_memory.SharedMemory(name='shm_intrinsics')
        shm_counter = shared_memory.SharedMemory(name='shm_counter')
        #shm_disps = shared_memory.SharedMemory(name='shm_disps')
        shm_disps_up = shared_memory.SharedMemory(name='shm_disps_up')        
        shm_dirty = shared_memory.SharedMemory(name='shm_dirty')
        shm_images = shared_memory.SharedMemory(name='shm_images')
    except FileNotFoundError:
        print("Shared memory segment not found.")
        return

    # shared memory 데이터를 torch 텐서로 변환 (shared_변수로 원본 참조)
    shared_poses = np.ndarray((buffer_size, 7), dtype=np.float32, buffer=shm_poses.buf)
    shared_intrinsics = np.ndarray((buffer_size, 4), dtype=np.float32, buffer=shm_intrinsics.buf)
    shared_counter = np.ndarray((1,), dtype=np.int32, buffer=shm_counter.buf)
    #shared_disps = np.ndarray((buffer_size, ht//8, wd//8), dtype=np.float32, buffer=shm_disps.buf)
    shared_disps_up = np.ndarray((buffer_size, ht, wd), dtype=np.float32, buffer=shm_disps_up.buf)    
    shared_dirty = np.ndarray((buffer_size,), dtype=np.bool_, buffer=shm_dirty.buf)
    shared_images = np.ndarray((buffer_size, 3, ht, wd), dtype=np.uint8, buffer=shm_images.buf)

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        
        with torch.no_grad():
            current_counter = shared_counter[0]
            #print(f'visualization.py counter : {current_counter}')
            #import time
            #time.sleep(3)

            #return

            #torch.cuda.synchronize()

            #print(f'shape of shaered dirty: {shared_dirty.shape}')
            poses_device = torch.tensor(shared_poses).to(device)
            intrinsics_device = torch.tensor(shared_intrinsics).to(device) * 8
            #disps_device = torch.tensor(shared_disps).to(device)
            disps_up_device = torch.tensor(shared_disps_up).to(device)
            dirty_device = torch.tensor(shared_dirty).to(device)
            images_device = torch.tensor(shared_images).to(device)

            dirty_index, = torch.where(dirty_device)  # dirty flag가 True인 인덱스

            #print(f'shared dirty: {shared_dirty}')
            #print(f'pose device: {poses_device}')

            if len(dirty_index) == 0:
                return

            print(f'dirty index: {dirty_index}')

            shared_dirty[dirty_index.cpu()] = False

            # dirty flag 업데이트
            #shared_dirty[dirty_index] = False


            # shared memory에서 필요한 데이터 추출
            poses = torch.index_select(poses_device, 0, dirty_index)
            #disps = torch.index_select(disps_device, 0, dirty_index)
            disps_up = torch.index_select(disps_up_device, 0, dirty_index)
            

            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(images_device, 0, dirty_index)
            #images = images.cpu()[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
            images = images.cpu()[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) / 255.0            
            #points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics_device[0]).cpu()
            points = droid_backends.iproj(SE3(poses).inv().data, disps_up, intrinsics_device[0]).cpu()

            #thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
            thresh = droid_visualization.filter_thresh * torch.ones_like(disps_up.mean(dim=[1, 2]))

            # Depth 필터링
            #count = droid_backends.depth_filter(poses_device, disps_device, intrinsics_device[0], dirty_index, thresh)
            count = droid_backends.depth_filter(poses_device, disps_up_device, intrinsics_device[0], dirty_index, thresh)

            #import time
            #time.sleep(5)
            #return

            #print(f'count: {count}')

            count = count.cpu()
            #disps = disps.cpu()
            disps_up = disps_up.cpu()

            #asks = ((count >= 2) & (disps > .5 * disps.mean(dim=[1, 2], keepdim=True)))
            masks = ((count >= 2) & (disps_up > .5 * disps_up.mean(dim=[1, 2], keepdim=True)))


            #print(f'poses: {SE3(poses).inv().data}')
            #print(f'disps: {disps}')
            #print(f'count: {count}')
            #print(f'masks: {masks}')
            #print(f'intrinsics: {shared_intrinsics[0]}')
    
            #points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics_device[0]).cpu()

            #print(f'reconstructed points: {points}')            
            
            #import time
            #time.sleep(5)
            #return

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    #print(f'remove cam{ix}')
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    #print(f'remove point{ix}')
                    del droid_visualization.points[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                
                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            #print('atm code!!!')
            # 카메라 파라미터가 있는지 확인하고 창 크기 비교
            #cam_intrinsic = cam.intrinsic
            #window_width = vis.get_render_option().width
            #window_height = vis.get_render_option().height
            #window_width, window_height = vis.get_window_size()

            #print(f"Camera width: {cam_intrinsic.width}, Window width: {960}")
            #print(f"Camera height: {cam_intrinsic.height}, Window height: {540}")

            #print(cam_intrinsic)

            # 창 크기와 카메라 매개변수가 일치하는지 확인
            #if cam_intrinsic.width != window_width or cam_intrinsic.height != window_height:
            #    print("Camera parameters and window size do not match!")

            # 메모리 해제 과정
            #del poses_device, intrinsics_device, disps_device, dirty_device, images_device

            # 가비지 수거 호출
            #gc.collect()

            # GPU 캐시 비우기
            #torch.cuda.empty_cache()

            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

            #import time
            #time.sleep(5)

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()
