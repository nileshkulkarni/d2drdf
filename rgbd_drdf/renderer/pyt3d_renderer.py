from copy import deepcopy
from math import degrees

import numpy as np
import torch

import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVOrthographicCameras,
    PerspectiveCameras,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer,
)
from pytorch3d.structures import Pointclouds

SMPL_OBJ_COLOR_LIST = [
    [0.65098039, 0.74117647, 0.85882353],  # SMPL
    [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
]


class PointRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"

    def __init__(
        self,
        cameras,
        image_size=1200,
        faces_per_pixel=1,
        device="cuda:0",
        blur_radius=0.003,
        lights=None,
        materials=None,
        max_faces_per_bin=50000,
    ):
        self.image_size = image_size
        self.faces_per_pixel = faces_per_pixel
        self.max_faces_per_bin = max_faces_per_bin  # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights = (
            lights
            if lights is not None
            else PointLights(
                ((0.5, 0.5, 0.5),),
                ((0.5, 0.5, 0.5),),
                ((0.05, 0.05, 0.05),),
                ((0, -2, 0),),
                device,
            )
        )

        self.materials = materials
        self.cameras = cameras

    def setup_renderer(self, cameras, radius=0.001):
        # for sillhouette rendering
        sigma = 1e-4

        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            # faces_per_pixel=self.faces_per_pixel,
            # clip_barycentric_coords=False,
            # max_faces_per_bin=self.max_faces_per_bin,
            points_per_pixel=8
            # bin_size=None
        )

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PulsarPointsRenderer(rasterizer=rasterizer, n_channels=4).to(
            self.device
        )
        return renderer

    def set_camera(self, cameras):
        self.cameras = cameras
        return

    def render(self, points, ret_mask=False, **kwargs):
        ## disk radius is a function of distance to points
        points_tensor, points_normals, points_features = points.get_cloud(0)
        radius = 0.01 + 0.02 * (points_tensor[:, 2] / 10)

        renderer = self.setup_renderer(self.cameras, radius=radius)

        # images = renderer(points, **kwargs)
        images = renderer(
            points,
            gamma=(1e-4,),
            zfar=(11.0,),
            znear=(0.1,),
            radius_world=True,
            bg_col=torch.tensor(
                [0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
            ),
            # dtype=torch.float32,
            # device=self.device
        )

        # print(images.shape)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :].cpu().detach().numpy(), mask > 0
        return images[0, ..., :].cpu().detach().numpy()


class Pyt3DWrapperPointCloud:
    def __init__(
        self,
        image_size,
        device="cuda:0",
        colors=SMPL_OBJ_COLOR_LIST,
        dataset="matterport",
    ):
        self.colors = deepcopy(colors)
        self.device = device
        self.image_size = image_size
        if dataset in ["matterport", "scannet"]:
            self.front_camera = self.get_matterport_camera(
                image_size=image_size, device=device
            )
        elif dataset in [
            "taskonomy",
            "threedf",
        ]:
            self.front_camera = self.get_taskonomy_camera(
                image_size=image_size, device=device
            )
        else:
            assert False, "unkown dataset"
        self.front_cameras = self.get_matteport_3d_novel_cameras(
            image_size=image_size, device=device
        )
        self.renderer = PointRendererWrapper(
            cameras=self.front_camera, image_size=image_size, device=device
        )
        self.camera_fn = Pyt3DWrapperPointCloud.get_camera_fn(
            image_size=image_size, device=device
        )
        self.matterport_camera_params = self.get_matterport_camera_params()
        self.taskonomy_camera_params = self.get_taskonomy_camera_params()

    @staticmethod
    def get_camera_fn(
        image_size,
        device,
    ):
        def get_camera(RT):
            return Pyt3DWrapperPointCloud.get_matterport_camera(
                image_size=image_size, device=device, RT=RT
            )

        return get_camera

    @staticmethod
    def get_taskonomy_camera_params():
        camera_params = {}
        camera_params["fx"] = 464.6562363
        camera_params["fy"] = 464.6562363
        camera_params["cx"] = 256
        camera_params["cy"] = 256
        return camera_params

    @staticmethod
    def get_matterport_camera_params():
        camera_params = {}
        camera_params["fx"] = 1075.1
        camera_params["fy"] = 1075.1
        camera_params["cx"] = 629.7
        camera_params["cy"] = 522.3
        return camera_params

    @staticmethod
    def get_taskonomy_camera(
        image_size,
        device="cuda:0",
        RT=None,
    ):
        camera_params = {}
        camera_params["fx"] = 464.6562363
        camera_params["fy"] = 464.6562363
        camera_params["cx"] = 256
        camera_params["cy"] = 256

        # def camera_fn(image_size, device='cuda:0', RT=RT):
        return Pyt3DWrapperPointCloud.get_matterport_camera(
            image_size, device=device, RT=RT, camera_params=camera_params
        )

    @staticmethod
    def get_matterport_camera(image_size, device="cuda:0", RT=None, camera_params=None):
        R, T = torch.eye(3), torch.zeros(3)
        # R[0, 0] = R[1, 1] = -1  # pytorch3d y-axis up, need to rotate to kinect coordinate
        R[0, 0] = R[1, 1] = 1
        RT_pytorch3d = torch.eye(4)
        RT_pytorch3d[:3, :3] = R
        RT_pytorch3d[:3, 3] = T
        if RT is not None:
            RT_local = RT * 1
            newRT = torch.matmul(RT_local, RT_pytorch3d)
            R = newRT[:3, :3]
            T = newRT[:3, 3]
            breakpoint()

        R = R.unsqueeze(0).to(device)
        T = T.unsqueeze(0).to(device)
        # fx, fy = 979.7844, 979.840  # focal length
        # cx, cy = 1018.952, 779.486  # camera centers
        # fx, fy = 1075.1, 1075.8
        if camera_params is None:
            fx, fy = 1075.1, 1075.1
            cx, cy = 629.7, 522.3
        else:
            fx = camera_params["fx"]
            fy = camera_params["fy"]
            cx = camera_params["cx"]
            cy = camera_params["cy"]
        tx = (1280 - image_size[1]) / 2

        color_w, color_h = image_size[1], image_size[0]  # kinect color image size
        cx = cx - tx
        cy = cy

        ## scale cx cy to ndc

        cx, cy = 2 * (color_w - cx) / color_w - 1, 2 * (color_h - cy) / color_h - 1
        cx = cx
        cy = cy
        # cx = 0
        # cy = 0

        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        # focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor(
            [2 * fx / image_size[1], (2 * fy / image_size[0])], dtype=torch.float32
        ).unsqueeze(0)

        # focal_length = (2 * fx / image_size[1], )
        pyt3d_version = pytorch3d.__version__
        if True:
            if pyt3d_version >= "0.6.0":
                cam = PerspectiveCameras(
                    focal_length=focal_length,
                    principal_point=cam_center,
                    image_size=(image_size,),
                    device=device,
                    R=R,
                    T=T,
                )
            else:
                assert False, "should not come here"
                cam = PerspectiveCameras(
                    focal_length=focal_length,
                    principal_point=cam_center,
                    image_size=((color_w, color_h),),
                    device=device,
                    R=R,
                    T=T,
                )
        if False:
            cam = FoVOrthographicCameras(
                device=device,
                R=R,
                T=T,
                znear=0.1,
            )
        return cam

    @staticmethod
    def get_matteport_3d_novel_cameras(
        image_size,
        device="cuda:0",
    ):
        R, T = torch.eye(3), torch.zeros(3)
        R[0, 0] = R[1, 1] = 1

        R = R.unsqueeze(0).to(device)
        T = T.unsqueeze(0).to(device)
        # fx, fy = 979.7844, 979.840  # focal length
        # cx, cy = 1018.952, 779.486  # camera centers
        # fx, fy = 1075.1, 1075.8

        fx, fy = 1075.1, 1075.1
        cx, cy = 629.7, 522.3
        tx = (1280 - image_size[1]) / 2

        color_w, color_h = image_size[1], image_size[0]  # kinect color image size
        cx = cx - tx
        cy = cy

        transforms = []
        for step in range(-5, 5):
            az = -5 * step
            el = np.abs(step) * (-5) * 0
            step_cam = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=-5, elev=el, azim=az, degrees=True, at=((0, 0, 5),)
            )
            transforms.append(step_cam)
        ## scale cx cy to ndc
        cx, cy = 2 * (color_w - cx) / color_w - 1, 2 * (color_h - cy) / color_h - 1
        cx = cx
        cy = cy
        # cx = 0
        # cy = 0

        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        # focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor(
            [2 * fx / image_size[1], (2 * fy / image_size[0])], dtype=torch.float32
        ).unsqueeze(0)

        # focal_length = (2 * fx / image_size[1], )
        cameras = []
        for tfs in transforms:
            R = tfs[0].to(device)
            T = tfs[1].to(device)
            cam = PerspectiveCameras(
                focal_length=focal_length,
                principal_point=cam_center,
                image_size=(image_size,),
                device=device,
                R=R,
                T=T,
            )
            cameras.append(cam)

        return cameras

    @staticmethod
    def get_gif_cameras(image_size, camera_params, device="cuda:0"):
        R, T = torch.eye(3), torch.zeros(3)
        R[0, 0] = R[1, 1] = 1

        R = R.unsqueeze(0).to(device)
        T = T.unsqueeze(0).to(device)
        # fx, fy = 979.7844, 979.840  # focal length
        # cx, cy = 1018.952, 779.486  # camera centers
        # fx, fy = 1075.1, 1075.8

        fx, fy = camera_params["fx"], camera_params["fy"]
        cx, cy = camera_params["cx"], camera_params["cy"]

        tx = (image_size[1] - image_size[1]) / 2

        color_w, color_h = image_size[1], image_size[0]  # kinect color image size
        cx = cx - tx
        cy = cy

        transforms = []

        num_steps = 60
        total_az_var = 10.0 ## this was 20
        az_step = total_az_var / num_steps

        total_el_var = 10.0 ## this was 20

        el_step = total_az_var / num_steps

        for step in range(0, -num_steps, -1):
            az = -(az_step) * step
            el = 0
            step_cam = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=-5, elev=el, azim=az, degrees=True, at=((0, 0, 5),)
            )
            transforms.append(step_cam)

        for step in range(-num_steps, num_steps, 1):
            az = -(az_step) * step
            el = 0
            step_cam = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=-5, elev=el, azim=az, degrees=True, at=((0, 0, 5),)
            )
            transforms.append(step_cam)

        for step in range(num_steps, -num_steps, -1):
            az = -az_step * step
            el = (num_steps - np.abs(step)) * (-1 * el_step)
            step_cam = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=-5, elev=el, azim=az, degrees=True, at=((0, 0, 5),)
            )
            transforms.append(step_cam)

        for step in range(-num_steps, 0, 1):
            az = -az_step * step
            el = 0
            step_cam = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=-5, elev=el, azim=az, degrees=True, at=((0, 0, 5),)
            )
            transforms.append(step_cam)

        ## scale cx cy to ndc
        cx, cy = 2 * (color_w - cx) / color_w - 1, 2 * (color_h - cy) / color_h - 1
        cx = cx
        cy = cy

        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        # focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor(
            [2 * fx / image_size[1], (2 * fy / image_size[0])], dtype=torch.float32
        ).unsqueeze(0)

        # focal_length = (2 * fx / image_size[1], )
        cameras = []
        for tfs in transforms:
            R = tfs[0].to(device)
            T = tfs[1].to(device)
            cam = PerspectiveCameras(
                focal_length=focal_length,
                principal_point=cam_center,
                image_size=(image_size,),
                device=device,
                R=R,
                T=T,
            )
            cameras.append(cam)

        return cameras

    def render_point_cloud_gif(
        self, point_clouds, point_clouds_color, RT=None, camera_params={}
    ):
        if point_clouds_color is not None:
            colors_lst = point_clouds_color
        else:
            color_lst = []
            for cx in range(len(point_clouds)):
                colors = deepcopy(self.colors)[cx]
                colors = np.array(colors)[None] + 0 * point_clouds[cx]
                color_lst.append(colors)
        render_pcl = self.prepare_render(point_clouds, colors_lst)
        gamma = 0.1
        device = self.device
        if not camera_params:
            camera_params = self.matterport_camera_params

        cameras = self.get_gif_cameras(
            image_size=self.image_size, camera_params=camera_params
        )
        images_all = []

        for cam in cameras:
            self.renderer.set_camera(cameras=cam)
            images = self.renderer.render(
                points=render_pcl,
                gamma=(gamma,),
                zfar=(45.0,),
                znear=(0.1,),
                radius_world=True,
                bg_col=torch.tensor(
                    [0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device
                ),
                dtype=torch.float32,
                device=device,
            )
            images_all.append(images)
        return images_all

    def render_point_clouds(self, point_clouds, point_clouds_color, RT=None):
        if point_clouds_color is not None:
            colors_lst = point_clouds_color
        else:
            color_lst = []
            for cx in range(len(point_clouds)):
                colors = deepcopy(self.colors)[cx]
                colors = np.array(colors)[None] + 0 * point_clouds[cx]
                color_lst.append(colors)
        render_pcl = self.prepare_render(point_clouds, colors_lst)
        gamma = 0.1
        device = self.device
        cameras = [self.front_camera]
        if RT:
            cam = self.camera_fn(RT=torch.FloatTensor(RT))
        cameras = self.front_cameras
        images_all = []
        for cam in cameras:
            self.renderer.set_camera(cameras=cam)
            images = self.renderer.render(
                points=render_pcl,
                gamma=(gamma,),
                zfar=(45.0,),
                znear=(0.1,),
                radius_world=True,
                bg_col=torch.tensor(
                    [0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
                ),
                dtype=torch.float32,
                device=device,
            )
            images_all.append(images)
        # import imageio
        # for i in range(len(images_all)):
        #     imageio.imsave('test/test_{}.png'.format(i), images_all[i])
        # breakpoint()

        return images_all

    def prepare_render(self, point_cloud_lst, colors_lst):

        out_point_clouds = []

        points = []
        rgb_feat = []
        for pcl, color in zip(point_cloud_lst, colors_lst):
            verts = torch.Tensor(pcl).to(self.device)
            rgb = torch.Tensor(color).to(self.device)
            # rgb = torch.Tensor(pcl * 0 + color).to(self.device)
            points.append(verts)
            rgb_feat.append(rgb)

        points = torch.cat(points, axis=0)
        rgb_feat = torch.cat(rgb_feat, axis=0)
        point_cloud = Pointclouds(points=[points], features=[rgb_feat])
        return point_cloud


from matplotlib import cm

from rgbd_drdf.utils import geometry_utils


def render_pcl(pcl, RT, py3d_renderer):
    ## Transform meshes to camera coordiante frame.
    pcl_local = geometry_utils.transform_points(pcl.transpose(), RT).transpose()
    color_map = cm.get_cmap("inferno")
    depth_colors = color_map(pcl_local[:, 2] / 10)
    if True:
        pcl_local[:, 0] = -1 * pcl_local[:, 0]
        pcl_local[:, 1] = -1 * pcl_local[:, 1]
        rendered = py3d_renderer.render_point_clouds(
            point_clouds=[pcl_local], point_clouds_color=[depth_colors[:, 0:3]], RT=None
        )
    if False:
        rendered = py3d_renderer.render_point_clouds(
            point_clouds=[pcl],
            point_clouds_color=[depth_colors[:, 0:3]],
            RT=np.linalg.inv(RT),
        )
    return rendered


def compute_normals(pcd, center=np.array([0, 0, 0])):
    import open3d as o3d

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=15)
    )
    # normals = np.asarray(pcd.normals)
    normals = np.asarray(pcd.normals)
    normals = align_normals(np.asarray(pcd.points), normals, center)
    normals = np.clip(normals, a_min=-1, a_max=1)
    return normals


def align_normals(points, normals, center):
    direction = center[None, 0:3] - points
    dot_p = np.sign(np.sum(normals * direction, axis=1))
    normals = normals * dot_p[:, None]
    return normals


import open3d as o3d


def render_pcl_w_normals(pcl, RT, py3d_renderer):
    ## Transform meshes to camera coordiante frame.
    pcl_local = geometry_utils.transform_points(pcl.transpose(), RT).transpose()
    color_map = cm.get_cmap("inferno")

    depth_colors = color_map(pcl_local[:, 2] / 10)
    camera_frame_pcd = o3d.geometry.PointCloud()
    camera_frame_pcd.points = o3d.utility.Vector3dVector(pcl_local)
    normals = compute_normals(camera_frame_pcd)
    ## for matterport

    normals = normals[:, (1, 0, 2)]
    normals[:, 0] = -1 * normals[:, 0]
    # normals[:, 1] = -1 * normals[:, 1]
    normals[:, 2] = -1 * normals[:, 2]
    normal_colors = (normals + 1) / 2
    normal_colors = np.concatenate(
        [normal_colors, normal_colors[:, 0:1] * 0 + 1], axis=-1
    )
    if True:
        pcl_local[:, 0] = -1 * pcl_local[:, 0]
        pcl_local[:, 1] = -1 * pcl_local[:, 1]
        rendered = py3d_renderer.render_point_clouds(
            point_clouds=[pcl_local], point_clouds_color=[normal_colors], RT=None
        )
    if False:
        rendered = py3d_renderer.render_point_clouds(
            point_clouds=[pcl],
            point_clouds_color=[depth_colors[:, 0:3]],
            RT=np.linalg.inv(RT),
        )
    return rendered


def render_pcl_w_normals_w_visiblity(
    pcl, visibility, image_point_colors, RT, py3d_renderer
):
    ## Transform meshes to camera coordiante frame.
    pcl_local = geometry_utils.transform_points(pcl.transpose(), RT).transpose()
    color_map = cm.get_cmap("inferno")

    depth_colors = color_map(pcl_local[:, 2] / 10)
    camera_frame_pcd = o3d.geometry.PointCloud()
    camera_frame_pcd.points = o3d.utility.Vector3dVector(pcl_local)
    normals = compute_normals(camera_frame_pcd)
    ## for matterport

    normals = normals[:, (1, 0, 2)]
    normals[:, 0] = -1 * normals[:, 0]
    # normals[:, 1] = -1 * normals[:, 1]
    normals[:, 2] = -1 * normals[:, 2]
    normal_colors = (normals + 1) / 2
    # if True:
    # pcl_colors = visibility[:,None] * depth_colors[:, 0:3] + (1 - visibility[:,None]) * normal_colors

    if visibility.max() > 1 or True:
        first_hits = visibility[:, None] == 0  ## has the intersection index.
        print("using modern visibility")
        pcl_colors = (
            first_hits * image_point_colors[:, 0:3]
            + np.logical_not(first_hits) * normal_colors
        )
    else:
        assert False, "should not use this visiblity"
        pcl_colors = (
            visibility[:, None] * image_point_colors[:, 0:3]
            + (1 - visibility[:, None]) * normal_colors
        )
    pcl_colors = np.concatenate([pcl_colors, pcl_colors[:, 0:1] * 0 + 1], axis=-1)
    if True:
        pcl_local[:, 0] = -1 * pcl_local[:, 0]
        pcl_local[:, 1] = -1 * pcl_local[:, 1]
        rendered = py3d_renderer.render_point_clouds(
            point_clouds=[pcl_local], point_clouds_color=[pcl_colors], RT=None
        )
    if False:
        rendered = py3d_renderer.render_point_clouds(
            point_clouds=[pcl],
            point_clouds_color=[depth_colors[:, 0:3]],
            RT=np.linalg.inv(RT),
        )
    return rendered


def render_pcl_w_normals_w_visiblity_gif(
    pcl, visibility, image_point_colors, RT, py3d_renderer, camera_params
):
    ## Transform meshes to camera coordiante frame.
    pcl_local = geometry_utils.transform_points(pcl.transpose(), RT).transpose()
    color_map = cm.get_cmap("inferno")

    depth_colors = color_map(pcl_local[:, 2] / 10)
    camera_frame_pcd = o3d.geometry.PointCloud()
    camera_frame_pcd.points = o3d.utility.Vector3dVector(pcl_local)
    normals = compute_normals(camera_frame_pcd)
    ## for matterport

    normals = normals[:, (1, 0, 2)]
    normals[:, 0] = -1 * normals[:, 0]
    # normals[:, 1] = -1 * normals[:, 1]
    normals[:, 2] = -1 * normals[:, 2]
    normal_colors = (normals + 1) / 2
    # if True:
    # pcl_colors = visibility[:,None] * depth_colors[:, 0:3] + (1 - visibility[:,None]) * normal_colors

    if visibility.max() > 1 or True:
        first_hits = visibility[:, None] == 0  ## has the intersection index.
        print("using modern visibility")
        pcl_colors = (
            first_hits * image_point_colors[:, 0:3]
            + np.logical_not(first_hits) * normal_colors
        )
    else:
        assert False, "should not use this visiblity"
        pcl_colors = (
            visibility[:, None] * image_point_colors[:, 0:3]
            + (1 - visibility[:, None]) * normal_colors
        )
    pcl_colors = np.concatenate([pcl_colors, pcl_colors[:, 0:1] * 0 + 1], axis=-1)
    if True:
        pcl_local[:, 0] = -1 * pcl_local[:, 0]
        pcl_local[:, 1] = -1 * pcl_local[:, 1]
        rendered = py3d_renderer.render_point_cloud_gif(
            point_clouds=[pcl_local],
            point_clouds_color=[pcl_colors],
            RT=None,
            camera_params=camera_params,
        )
    return rendered
