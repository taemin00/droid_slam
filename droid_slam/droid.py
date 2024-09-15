import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

from multiprocessing import shared_memory

class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        self.ht = args.image_size[0]
        self.wd = args.image_size[1]
        self.buffer_size = args.buffer

        try:
            self.shm_poses = shared_memory.SharedMemory(name='shm_poses')
            self.shm_intrinsics = shared_memory.SharedMemory(name='shm_intrinsics')
            self.shm_counter = shared_memory.SharedMemory(name='shm_counter')
            #self.shm_disps = shared_memory.SharedMemory(name='shm_disps')
            self.shm_disps_up = shared_memory.SharedMemory(name='shm_disps_up')            
            self.shm_dirty = shared_memory.SharedMemory(name='shm_dirty')
            self.shm_images = shared_memory.SharedMemory(name='shm_images')
        except FileNotFoundError:
            self.shm_poses = shared_memory.SharedMemory(create=True, size=self.buffer_size * 7 * 4, name='shm_poses')
            self.shm_intrinsics = shared_memory.SharedMemory(create=True, size=self.buffer_size * 4 * 4, name='shm_intrinsics')
            self.shm_counter = shared_memory.SharedMemory(create=True, size=4, name='shm_counter')
            #self.shm_disps = shared_memory.SharedMemory(create=True, size=self.buffer_size * (self.ht//8) * (self.wd//8) * 4, name='shm_disps')
            self.shm_disps_up = shared_memory.SharedMemory(create=True, size=self.buffer_size * self.ht * self.wd * 4, name='shm_disps_up')
            self.shm_dirty = shared_memory.SharedMemory(create=True, size=self.buffer_size, name='shm_dirty')
            self.shm_images = shared_memory.SharedMemory(create=True, size=self.buffer_size * 3 * self.ht * self.wd, name='shm_images')  # 이미지 공유 메모리

        # numpy 배열로 변환
        self.poses = np.ndarray((self.buffer_size, 7), dtype=np.float32, buffer=self.shm_poses.buf)
        self.intrinsics = np.ndarray((self.buffer_size, 4), dtype=np.float32, buffer=self.shm_intrinsics.buf)
        self.counter = np.ndarray((1,), dtype=np.int32, buffer=self.shm_counter.buf)
        #self.disps = np.ndarray((self.buffer_size, self.ht//8, self.wd//8), dtype=np.float32, buffer=self.shm_disps.buf)
        self.disps_up = np.ndarray((self.buffer_size, self.ht, self.wd), dtype=np.float32, buffer=self.shm_disps_up.buf)
        self.dirty = np.ndarray((self.buffer_size,), dtype=np.bool_, buffer=self.shm_dirty.buf)
        self.images = np.ndarray((self.buffer_size, 3, self.ht, self.wd), dtype=np.uint8, buffer=self.shm_images.buf)  # 이미지 추가


        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.dirty, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            #self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer = Process(target=droid_visualization, args=(self.ht, self.wd, self.buffer_size, ))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        self.old_t = -1

    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            t = self.video.counter.value        
   
            #if t > 0 and t % 10 == 0:
            #    self.backend(3)

            #dirty_index, = torch.where(self.video.dirty.clone())
            #print(f'droid.py!!! - dirty index: {dirty_index}')
            # 공유 메모리 업데이트
            
            if t > self.old_t:
                if t > 0 and t % 100 == 0:
                    torch.cuda.empty_cache()
                    print("#" * 32)
                    self.backend(10)

                    #torch.cuda.empty_cache()
                    #print("#" * 32)
                    #self.backend(12)

                    #self.video.disps[:] = torch.zeros_like(self.video.disps)
                    #self.video.poses[:] = torch.zeros_like(self.video.poses)

                #print(f'droid.py - update shared memory!!!(t={t})')
                np.copyto(self.poses, self.video.poses.cpu().numpy())
                np.copyto(self.intrinsics, self.video.intrinsics.cpu().numpy())
                self.counter[0] = self.video.counter.value
                #np.copyto(self.disps, self.video.disps.cpu().numpy())
                np.copyto(self.disps_up, self.video.disps_up.cpu().numpy())
                np.copyto(self.dirty, self.video.dirty.cpu().numpy())  # dirty도 업데이트
                np.copyto(self.images, self.video.images.cpu().numpy())  # 이미지도 업데이트
                self.old_t = t

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

