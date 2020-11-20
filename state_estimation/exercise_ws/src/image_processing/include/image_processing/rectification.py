import numpy as np
import cv2
from image_geometry import PinholeCameraModel
from .ground_projection_geometry import Point

class Rectify:
    """
    Handles the Rectification operations.

    """

    def __init__(self, camera_info):
        self.ci = camera_info
        self.pcm = PinholeCameraModel()
        self.pcm.fromCameraInfo(self.ci)
        self._rectify_inited = False
        self._distort_inited = False



    def rectify_point(self, pixel):
        return Point(*list(self.pcm.rectifyPoint([pixel.x, pixel.y])))

    def _init_rectify_maps(self):
        W = self.pcm.width
        H = self.pcm.height
        mapx = np.ndarray(shape=(H, W, 1), dtype='float32')
        mapy = np.ndarray(shape=(H, W, 1), dtype='float32')
        mapx, mapy = cv2.initUndistortRectifyMap(self.pcm.K, self.pcm.D, self.pcm.R,
                                                 self.pcm.P, (W, H),
                                                 cv2.CV_32FC1, mapx, mapy)
        self.mapx = mapx
        self.mapy = mapy
        self._rectify_inited = True

    def rectify(self, cv_image_raw, interpolation=cv2.INTER_NEAREST):
        ''' Undistort an image.
            To be more precise, pass interpolation= cv2.INTER_CUBIC
        '''
        if not self._rectify_inited:
            self._init_rectify_maps()
        #
        #        inter = cv2.INTER_NEAREST  # 30 ms
        #         inter = cv2.INTER_CUBIC # 80 ms
        #         cv_image_rectified = np.zeros(np.shape(cv_image_raw))
        cv_image_rectified = np.empty_like(cv_image_raw)
        res = cv2.remap(cv_image_raw, self.mapx, self.mapy, interpolation,
                        cv_image_rectified)
        return res

    def distort(self, rectified):
        if not self._rectify_inited:
            self._init_rectify_maps()
        if not self._distort_inited:
            self.rmapx, self.rmapy = invert_map(self.mapx, self.mapy)
            self._distort_inited = True
        distorted = np.zeros(np.shape(rectified))
        res = cv2.remap(rectified, self.rmapx, self.rmapy, cv2.INTER_NEAREST, distorted)
        return res

    def rectify_full(self, cv_image_raw, interpolation=cv2.INTER_NEAREST, ratio=1):
        '''
            Undistort an image by maintaining the proportions.
            To be more precise, pass interpolation= cv2.INTER_CUBIC
            Returns the new camera matrix as well.
        '''
        W = int(self.pcm.width * ratio)
        H = int(self.pcm.height * ratio)
        #        mapx = np.ndarray(shape=(H, W, 1), dtype='float32')
        #        mapy = np.ndarray(shape=(H, W, 1), dtype='float32')
        print('K: %s' % self.pcm.K)
        print('P: %s' % self.pcm.P)

        #        alpha = 1
        #        new_camera_matrix, validPixROI = cv2.getOptimalNewCameraMatrix(self.pcm.K, self.pcm.D, (H, W), alpha)
        #        print('validPixROI: %s' % str(validPixROI))

        # Use the same camera matrix
        new_camera_matrix = self.pcm.K.copy()
        new_camera_matrix[0, 2] = W / 2
        new_camera_matrix[1, 2] = H / 2
        print('new_camera_matrix: %s' % new_camera_matrix)
        mapx, mapy = cv2.initUndistortRectifyMap(self.pcm.K, self.pcm.D, self.pcm.R,
                                                 new_camera_matrix, (W, H),
                                                 cv2.CV_32FC1)
        cv_image_rectified = np.empty_like(cv_image_raw)
        res = cv2.remap(cv_image_raw, mapx, mapy, interpolation,
                        cv_image_rectified)
        return new_camera_matrix, res

    def invert_map(mapx, mapy):
        H, W = mapx.shape[0:2]
        rmapx = np.empty_like(mapx)
        rmapx.fill(np.nan)
        rmapy = np.empty_like(mapx)
        rmapy.fill(np.nan)

        for y, x in itertools.product(range(H), range(W)):
            tx = mapx[y, x]
            ty = mapy[y, x]

            tx = int(np.round(tx))
            ty = int(np.round(ty))

            if (0 <= tx < W) and (0 <= ty < H):
                rmapx[ty, tx] = x
                rmapy[ty, tx] = y

        # fill holes
        #     if False:

        fill_holes(rmapx, rmapy)

        #     D = 4
        #     for y, x in itertools.product(range(H), range(W)):
        #         v0 = max(y-D, 0)
        #         v1 = max(y+D, H-1)
        #         u0 = max(x-D, 0)
        #         u1 = max(x+D, W-1)
        #
        #         rmapx[y,x] = np.median(rmapx[v0:v1,u0:u1].flatten())
        #         rmapy[y,x] = np.median(rmapy[v0:v1,u0:u1].flatten())

        return rmapx, rmapy

    def fill_holes(rmapx, rmapy):
        H, W = rmapx.shape[0:2]

        nholes = 0

        R = 2
        F = R * 2 + 1

        def norm(x):
            return np.hypot(x[0], x[1])

        deltas0 = [(i - R - 1, j - R - 1) for i, j in itertools.product(range(F), range(F))]
        deltas0 = [x for x in deltas0 if norm(x) <= R]
        deltas0.sort(key=norm)

        def get_deltas():
            #         deltas = list(deltas0)
            #
            return deltas0

        holes = set()

        for i, j in itertools.product(range(H), range(W)):
            if np.isnan(rmapx[i, j]):
                holes.add((i, j))

        while holes:
            nholes = len(holes)
            nholes_filled = 0

            for i, j in list(holes):
                # there is nan
                nholes += 1
                for di, dj in get_deltas():
                    u = i + di
                    v = j + dj
                    if (0 <= u < H) and (0 <= v < W):
                        if not np.isnan(rmapx[u, v]):
                            rmapx[i, j] = rmapx[u, v]
                            rmapy[i, j] = rmapy[u, v]
                            nholes_filled += 1
                            holes.remove((i, j))
                            break

            #         print('holes %s filled: %s' % (nholes, nholes_filled))
            if nholes_filled == 0:
                break
