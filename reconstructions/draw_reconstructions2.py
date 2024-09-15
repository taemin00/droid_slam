import open3d as o3d
import numpy as np
import quaternion
import cv2
from scipy.spatial.transform import Rotation as R


data_path = 'result'

disps = np.load(data_path + '/disps.npy')
images = np.load(data_path + '/images.npy')
intrinsics = np.load(data_path + '/intrinsics.npy')
poses = np.load(data_path + '/poses.npy')

print(f'disps shape: {disps.shape}')
print(f'iamges shape: {images.shape}')
print(f'intrinsics.shape: {intrinsics.shape}')
print(f'poses.shape: {poses.shape}')

#print(f'last disp: {disps[frame_index]}')
#print(f'last image: {images[frame_index]}')
#print(f'all intrinsics: {intrinsics}')
#print(f'all poses: {poses}')

point_clouds = []

#for frame_index in range(10,64,10):
for frame_index in range(0, len(images)):
    depth_map = (1/disps[frame_index])
    #depth_map[depth_map > 100] = 100
    depth_min = depth_map.min()
    depth_max = depth_map.max()

    if depth_max > 5:
        continue

    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)  # 0 ~ 1로 정규화
    depth_map_normalized = (depth_map_normalized * 255).astype(np.uint8)  # 0 ~ 255로 변환
    color_map = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    print(f'depth min: {depth_map.min()}, depth max: {depth_map.max()}')
    #cv2.imshow('depth', color_map)
    #cv2.waitKey(0)

    rgb_image = images[frame_index].transpose(1, 2, 0)
    print(f'rgb min: {rgb_image.min()}, rgb max: {rgb_image.max()}')

    #cv2.imshow('image', rgb_image)
    #cv2.waitKey(0)

    combined_image = np.hstack((rgb_image, color_map))

    #cv2.imshow('rgb and depth', combined_image)
    #cv2.waitKey(0)

    fx = 428.3200
    fy = 431.3600
    cx = 256.0800
    cy = 198.0800

    #fx, fy, cx, cy = tuple(intrinsics[-1].tolist())

    print(f'fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}')

    # depth scale factor
    point_cloud = o3d.geometry.PointCloud()

    u, v = np.meshgrid(range(images.shape[3]), range(images.shape[2]))
    print(f'u shape: {u.shape}, vshape: {v.shape}')
    print(f'dips shape: {disps.shape}')

    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map


    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    print(f'shape of point cloud: {points.shape}')

    colors = rgb_image.reshape((-1, 3)) / 255

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    voxel_size = 0.01  # 다운샘플링 할 voxel 크기
    point_cloud = point_cloud.voxel_down_sample(voxel_size)
    print(f'num downsampled points: {len(point_cloud.points)}')

    pose = poses[frame_index]
    print(f'camera pose: {pose}')

    # 평행 이동 (첫 번째 3개 값)
    translation = pose[:3]

    # 쿼터니언 (마지막 4개 값)
    quaternion = pose[3:]

    # 쿼터니언을 회전 행렬로 변환
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # 4x4 변환 행렬 생성
    transformation_matrix = np.eye(4)  # 단위 행렬로 초기화
    transformation_matrix[:3, :3] = rotation_matrix  # 회전 행렬 설정
    transformation_matrix[:3, 3] = translation  # 평행 이동 벡터 설정

    inverse_transformation = np.linalg.inv(transformation_matrix)
    point_cloud.transform(inverse_transformation)

    point_clouds.append(point_cloud)

    #or i in range(poses.shape[0]):
        # homogeneous transform matrix로 변환
    #   pose_matrix = np.eye(4)
    #   pose_matrix[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(poses[i, 3:]))
    #   pose_matrix[:3, 3] = poses[i, :3]

        # point_cloud 객체의 위치 및 방향 설정
    #   point_cloud.transform(pose_matrix)


o3d.visualization.draw_geometries(point_clouds)
