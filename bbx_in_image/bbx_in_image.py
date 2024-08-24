import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

class bbx():
    def __init__(self, anno, image_size):

        self.l = anno["3D_dimension"]["l"]
        self.w = anno["3D_dimension"]["w"]
        self.h = anno["3D_dimension"]["h"]
        
        self.x = anno["3D_location"]["x"]
        self.y = anno["3D_location"]["y"]
        self.z = anno["3D_location"]["z"]
        self.id = anno["track_id"]

        self.image_x = image_size[0]
        self.image_y = image_size[1]
        self.rotation = self.calc_rotaional_matrix(anno["rotation"])  #TODO 
        #self.rotation = np.array(anno)
        self.corners = self.getcorners()
    

    def compute_calib(self, K, R, t):
        """
        计算相机的 calib 矩阵。
        
        参数:
        K: 内参矩阵 (3, 3)
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3, 1)
        
        返回:
        calib 矩阵 (3, 4)
        """
        # 确保 t 是列向量
        t = t.reshape(3, 1)
        
        # 组合旋转和平移，得到外参矩阵 P
        P = np.hstack((R, t))  # (3, 4)
        
        # 计算 calib 矩阵
        calib = K @ P  # (3, 3) @ (3, 4) = (3, 4)
        
        return calib
    

    def render(self, 
           axis,
           
           calib_extrinsic,
           calib_intrinsic,
           colors,

           linewidth: float = 1) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        #TODO 调用 self.corners() 方法获取边界框的角点，并使用 view_points 函数将其投影到二维平面上。如果 normalize 为 True，则进行归一化处理。
        
        #TODO 
        
        R = calib_extrinsic[:3, :3]
        T = calib_extrinsic[:3, 3]
        
        
        calib = self.compute_calib(calib_intrinsic, R, T)
        #corners = self.my_view_points(calib, R, T)
        corners = self.view_points(calib)

        if max(corners[0, :]) > self.image_x or max(corners[1,:]) > self.image_y:
            print("current bbx %s is out of image range" % self.id)
            return None
        #todo draw_rect 函数用于绘制一个矩形。它接受选择的角点和颜色作为参数，依次连接这些角点，形成一个闭合的矩形。
        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # 使用循环绘制边界框的四个侧面。每个侧面由上面和下面的角点连接而成。
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                    [corners.T[i][1], corners.T[i + 4][1]],
                    color=colors[2], linewidth=linewidth)

        # 调用 draw_rect 函数分别绘制前面（前四个角点）和后面（后四个角点）的矩形。
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0], linewidth=linewidth)
        
        center_bottom = np.mean(corners.T[[0, 1, 4, 5]], axis=0)
        axis.text(center_bottom[0], center_bottom[1], s = self.id, color='black', fontsize=12, ha='center')
    


    
    def view_points(self, calib) -> np.ndarray:
        """
        This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
        orthographic projections. It first applies the dot product between the points and the view. By convention,
        the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
        normalization along the third dimension.

        For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
        TODO For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
        For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
        all zeros) and normalize=False

        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
            The projection should be such that the corners are projected onto the first 2 axis.
        :param normalize: Whether to normalize the remaining coordinate (along the third axis).
        :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
        """

        points = self.corners   
        
        view = calib
        
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]   #TODO 这里都是ok的
        
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)  #TODO
        
        

        return points[:2, :]

    def getcorners(self) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = self.l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = self.w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = self.h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.rotation, corners)  #TODO 

        # Translate
        corners[0, :] = corners[0, :] + self.x
        corners[1, :] = corners[1, :] + self.y
        corners[2, :] = corners[2, :] + self.z

        return corners

    def calc_rotaional_matrix(self, rotation_angles):
    # 输入的旋转角度（以弧度为单位）
    # rotation_angles = {
    #     "x": 0,
    #     "y": 0,
    #     "z": 0.5834386364060438
    # }
    # 提取旋转角度
        theta_x = rotation_angles["x"]
        theta_y = rotation_angles["y"]
        theta_z = rotation_angles["z"]

        # 计算绕 X 轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        # 计算绕 Y 轴的旋转矩阵
        R_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        # 计算绕 Z 轴的旋转矩阵
        R_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵（先绕 Z 轴，再绕 Y 轴，最后绕 X 轴）
        R = R_z @ R_y @ R_x
        
        return R

def render_data(ax, image_path, anno_path, calib_extrinsic_path, calib_intrinsic_path,
                           out_path: str = None):
        image = Image.open(image_path)
        # Show image.
        ax.imshow(image)

        # Show boxes.
        with open(anno_path, 'r') as file:
            anno = json.load(file)
        
        with open(calib_extrinsic_path, 'r') as file:
            calib_extrinsic = json.load(file)
            calib_extrinsic = np.array(calib_extrinsic)
            
        with open(calib_intrinsic_path, 'r') as file:
            calib_intrinsic = json.load(file)
            calib_intrinsic = np.array(calib_intrinsic)
        
        c = np.array([255, 158, 0]) / 255.0 # Orange
        ax.set_xlim(0, image.size[0])
        ax.set_ylim(image.size[1], 0)


        ax.axis('off')
        ax.set_title("bbx in image")
        ax.set_aspect('equal')
        for bbx_anno in anno:
            box = bbx(bbx_anno, image.size)
            box.render(ax, calib_extrinsic,calib_intrinsic, colors=(c, c, c))
            # debug_img_path = os.path.join("debug", "%s.png"% str(bbx_anno["track_id"]))
            # plt.savefig(debug_img_path)
        

        if out_path is not None:
            plt.savefig(out_path)
    
# def render_sample_data(ax,                       out_path: str = None):
#         image_path = '/home/myData/storage/xmh/Dataset/V2X-Sim-2.0-mini/sweeps/CAM_FRONT_id_1/scene_5_000006.jpg'
#         anno_path = "source_path/example/box.json"
#         calib_path = "source_path/example/calib.json"
        
        
#         image = Image.open(image_path)

#         # Init axes.
#         width, height = image.size

#     # 计算 figsize
#         figsize = (width / 100, height / 100)
        


#         # Show image.
#         ax.imshow(image)

#         # Show boxes.
#         with open(anno_path, 'r') as file:
#             anno = json.load(file)
        
#         with open(calib_path, 'r') as file:
#             calib = json.load(file)
        

#         calib = np.array(calib)
#         c = np.array([255, 158, 0]) / 255.0 # Orange
        
#         for bbx_anno in anno:
#             print(bbx_anno["track_id"])
#             if bbx_anno["track_id"] == "000009":
#                 print(1)
#             box = bbx(bbx_anno)
#             box.render(ax, calib, colors=(c, c, c))
        
        
#         ax.set_xlim(0, image.size[0])
#         ax.set_ylim(image.size[1], 0)
        
#         ax.axis('off')
#         ax.set_title("bbx in image")
#         ax.set_aspect('equal')

#         if out_path is not None:
#             plt.savefig(out_path)
#         print(out_path)


if __name__ == "__main__":
    #TODO  
    """
    source_path:
        calib:
            extrinsic.json
            intrinsic.json
        frame_n:
            xxx.pcd
            xxx.jpg
            xxx.json   (annotation)
    
    output:
        frame_n:
            output.png
    """
    
    source_path = "source_path" #TODO
    calib_folder = os.path.join(source_path, "calib")
    calib_extrinsic_path = os.path.join(calib_folder, "extrinsic.json")
    calib_intrinsic_path = os.path.join(calib_folder, "intrinsic.json")

    
    frame_folders = []
    for folder in os.listdir(source_path):
        if folder.startswith('frame'):
            frame_folders.append(os.path.join(source_path, folder))

    for frame_folder in frame_folders:
        for file in os.listdir(frame_folder):
            if file.endswith('.pcd'):
                pcd_path = os.path.join(frame_folder, file)
            if file.endswith('.jpg'):
                image_path = os.path.join(frame_folder, file)
            if file.endswith('.json'):
                anno_path = os.path.join(frame_folder, file)
        
        out_path = os.path.join(frame_folder, "output.png")
        if not os.path.exists(out_path):
            os.makedirs(out_path)



        fig, axs = plt.subplots(1, 1,  figsize=(15, 6))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


        print(frame_folder)
        render_data(axs, image_path, anno_path, calib_extrinsic_path, calib_intrinsic_path, out_path)

        #render_data(image_path, anno_path, calib_path, out_path)