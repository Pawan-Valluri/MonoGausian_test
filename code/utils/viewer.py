import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import math
import json

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, znear=0.01, zfar=10):
        self.image_width = W
        self.image_height = H
        self.radius = r
        self.fovy = fovy
        self.znear = znear
        self.zfar = zfar
        self.look_at = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_matrix(np.eye(3))
        self.z_sign = -1
        self.y_sign = -1
        self.dragging = False
        self.prev_mouse_pos = None

    def orbit(self, dx, dy):
        angle_x = np.radians(-0.3 * dy)
        angle_y = np.radians(-0.3 * dx)
        
        axis_x = self.rot.as_matrix()[:3, 0]
        axis_y = self.rot.as_matrix()[:3, 1]

        self.rot = R.from_rotvec(axis_y * angle_x) * self.rot
        self.rot = R.from_rotvec(axis_x * angle_y) * self.rot

    def pan(self, dx, dy):
        pan_speed = 0.005 * self.radius
        self.look_at += self.rot.apply([dx * pan_speed, -dy * pan_speed, 0])

    def zoom(self, delta):
        self.radius *= 1.1 ** (-delta)

    @property
    def intrinsics(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.image_width // 2, self.image_height // 2])

    @property
    def projection_matrix(self):
        K = self.intrinsics[None]
        h, w = self.image_height, self.image_width
        near, far = self.znear, self.zfar
        
        fx, fy, cx, cy = K[0]
        proj = np.zeros((4, 4))
        proj[0, 0] = 2 * fx / w
        proj[1, 1] = 2 * fy / h
        proj[0, 2] = (w - 2 * cx) / w
        proj[1, 2] = (h - 2 * cy) / h
        proj[2, 2] = self.z_sign * (far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = self.z_sign
        return proj

    @property
    def pose(self):
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] -= self.radius

        rot_matrix = np.eye(4, dtype=np.float32)
        rot_matrix[:3, :3] = self.rot.as_matrix()
        pose = rot_matrix @ pose

        pose[:3, 3] -= self.look_at
        pose[:, [1, 2]] *= self.y_sign
        return pose

    @property
    def world_view_transform(self):
        return np.linalg.inv(self.pose)

    @property
    def full_proj_transform(self):
        return self.projection_matrix @ self.world_view_transform

    @property
    def camera_center(self):
        return self.pose[:3, 3]

    @property
    def tanfovx(self):
        focal_x = self.intrinsics[0]
        return self.image_width / (2 * focal_x)

    @property
    def tanfovy(self):
        focal_y = self.intrinsics[1]
        return self.image_height / (2 * focal_y)

    @property
    def scale(self):
        return self.radius

    @property
    def rotation(self):
        return self.rot.as_quat()

class Mini3DViewer:
    def __init__(self, W=960, H=540):
        self.W, self.H = W, H
        self.camera = OrbitCamera(W, H)
        self.render_buffer = np.ones((H, W, 3), dtype=np.float32)

    def render(self):
        # Placeholder rendering logic.
        self.render_buffer[:, :, :] = np.random.rand(self.H, self.W, 3) * 0.5

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.camera.dragging = True
            self.camera.prev_mouse_pos = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.camera.dragging = False
            self.camera.prev_mouse_pos = None

        elif event == cv2.EVENT_MOUSEMOVE and self.camera.dragging:
            dx = x - self.camera.prev_mouse_pos[0]
            dy = y - self.camera.prev_mouse_pos[1]

            if flags & cv2.EVENT_FLAG_CTRLKEY:  # Pan
                self.camera.pan(dx, dy)
            else:  # Orbit
                self.camera.orbit(dx, dy)

            self.camera.prev_mouse_pos = (x, y)

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self.camera.zoom(delta)
        # print(self.camera.pose)
        print("##############\n", {
            "world_view_transform": self.camera.world_view_transform,
            "full_proj_transform": self.camera.full_proj_transform,
            "camera_center": self.camera.camera_center,
            "tanfovx": self.camera.tanfovx,
            "tanfovy": self.camera.tanfovy
        }, "\n##############\n")

    def run(self):
        cv2.namedWindow("3D Viewer")
        cv2.setMouseCallback("3D Viewer", self.on_mouse)

        while True:
            self.render()
            display_image = (self.render_buffer * 255).astype(np.uint8)
            cv2.imshow("3D Viewer", display_image)

            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = Mini3DViewer()
    viewer.run()
