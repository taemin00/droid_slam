import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    #fx/=2
    #fy/=2
    #cx/=2
    #cy/=2

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    print(f'num image inputs: {len(image_list)}')

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        #image = cv2.resize(image, (320, 240))

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        #h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        #w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        h1 = int(h0 * np.sqrt((240 * 320) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((240 * 320) / (h0 * w0)))

        #print(f'new width: {w1}, new height: {h1}')

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        #print(f'new intrinsics: {intrinsics}')

        yield t, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)

    print('intrinsics')
    print(intrinsics)
    print('poses')
    print(poses)
    print('disp')
    print(disps)
    
def show_point_clouds(droid):
    import open3d as o3d
    import droid_backends
    from lietorch import SE3
    #import pickle
    #import copy

    video = droid.video
    
    #with open('video_backup.pkl', 'wb') as f:
    #    video_copy = copy.deepcopy(video)
    #    pickle.dump(video_copy, f)


    # = droid.video.counter.value
    #stamps = droid.video.tstamp[:t].cpu().numpy()
    #mages = droid.video.images[:t].cpu().numpy()
    #isps = droid.video.disps_up[:t].cpu().numpy()
    #oses = droid.video.poses[:t].cpu().numpy()
    #ntrinsics = droid.video.intrinsics[:t].cpu().numpy()

    # Visualizer 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    filter_thresh = 0.05

    def create_point_actor(points, colors):
        """ open3d point cloud from numpy array """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    with torch.no_grad():
        with video.get_lock():
            dirty_index, = torch.where(video.dirty.clone())
            print(f'dirty indexs: {dirty_index}')

        if len(dirty_index) == 0:
            print('length of dirty index is 0')
            return

        #video.dirty[dirty_index] = False

        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)
        #Ps = SE3(poses).inv().matrix().cpu().numpy()

        images = torch.index_select(video.images, 0, dirty_index)
        images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            
        count = droid_backends.depth_filter(
            video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

        for i in range(len(dirty_index)):
            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                
            ## add point actor ###
            point_actor = create_point_actor(pts, clr)
            vis.add_geometry(point_actor)

        # 렌더링 및 시각화 업데이트
        vis.run()  # 창을 열고 렌더링을 시작합니다.
        vis.destroy_window()  # 창을 닫습니다.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[480, 640])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    it = 0

    for (t, image, intrinsics) in (image_stream(args.imagedir, args.calib, args.stride)):
        print(f'it: {it}')
        it += 1
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

        t = droid.video.counter.value
        #tstamps = droid.video.tstamp[t-1].cpu().numpy()
        #images = droid.video.images[t-1].cpu().numpy()
        #disps = droid.video.disps_up[t-1].cpu().numpy()
        #poses = droid.video.poses[t-1].cpu().numpy()
        #intrinsics = droid.video.intrinsics[t-1].cpu().numpy()
        
        dirty_index, = torch.where(droid.video.dirty.clone())

        #print(f'demo.py - t: {t}, dirty index: {dirty_index}')
        #print(f'tstmp: {tstamps}')
        #print(f'image: {images}')
        #print(f'disp: {disps}')
        #print(f'{t} poses(demo.py): {poses}')
        #print(f'intrinsic: {intrinsics}')


    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))

    #show_point_clouds(droid)

