from heat_clip.sam_heatmap_generator import SAMHeatmapGenerator

if __name__ == "__main__":
    # 创建热力图生成器
    heatmap_gen = SAMHeatmapGenerator(
        model_path="sam2.1_b.pt",
        use_gaussian_blur=True,
        blur_times=3,
        kernel_size=101,
        sigma=20,
    )

    # 生成单张图像的热力图
    result = heatmap_gen.generate_heatmaps(
        "examples/truck.png", bboxes=[[80, 120, 800, 380]], visualize=True
    )


# # SAMHeatmapGenerator Class
# import numpy as np
# import cv2
# from ultralytics import SAM
# from typing import List, Dict, Tuple, Optional, Union, Any

# class SAMHeatmapGenerator:
#     """
#     使用SAM模型生成掩码热力图的类
#     """
#     def __init__(
#         self,
#         model_path: str = "sam2.1_b.pt",
#         use_gaussian_blur: bool = True,
#         **kwargs
#     ):
#         """
#         初始化热力图生成器

#         Args:
#             model_path: SAM模型路径或名称
#             use_gaussian_blur: 是否使用高斯模糊生成热力图
#             **kwargs: 额外参数，包括：
#                 - blur_times: 高斯模糊重复次数，默认3
#                 - kernel_size: 高斯核大小，默认101
#                 - sigma: 高斯标准差，默认20
#         """
#         # 加载SAM模型
#         self.model = SAM(model_path)
#         self.use_gaussian_blur = use_gaussian_blur

#         # 设置高斯模糊参数，如果未提供则使用默认值
#         self.kwargs = {
#             'blur_times': kwargs.get('blur_times', 3),
#             'kernel_size': kwargs.get('kernel_size', 101),
#             'sigma': kwargs.get('sigma', 20)
#         }

#     def create_heatmap_gaussian(self, mask: np.ndarray) -> np.ndarray:
#         """
#         使用高斯模糊生成热力图

#         Args:
#             mask: 二值掩码

#         Returns:
#             热力图（灰度图）
#         """
#         # 确保掩码是二值的
#         binary_mask = (mask > 0).astype(np.uint8) * 255

#         # 应用高斯模糊
#         blur_times = self.kwargs.get('blur_times', 3)
#         kernel_size = self.kwargs.get('kernel_size', 101)
#         sigma = self.kwargs.get('sigma', 20)

#         # 确保kernel_size是奇数
#         if kernel_size % 2 == 0:
#             kernel_size += 1

#         blurred = cv2.GaussianBlur(binary_mask, (kernel_size, kernel_size), sigma)
#         for _ in range(blur_times - 1):  # 已经应用了一次，所以减1
#             blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), sigma)

#         return blurred

#     def generate_heatmaps(
#         self,
#         orig_img: Union[str, Any],
#         bboxes: List[List[int]],
#         labels: Optional[List[int]] = None,
#         visualize: bool = False
#     ) -> Dict[str, Any]:
#         """
#         对图像生成热力图

#         Args:
#             orig_img: str | Image | ndarray, 图像路径或图像对象
#             bboxes: 边界框列表，每个边界框为 [x1, y1, x2, y2] 格式
#             labels: List[int] | None, 可选的类别标签列表，与bboxes一一对应
#             visualize: 是否可视化结果

#         Returns:
#             包含原始图像、掩码和热力图的字典
#         """
#         # 运行SAM模型进行推理
#         results = self.model(orig_img, bboxes=bboxes, labels=labels)

#         # 如果labels是None的状态，可能bboxes的个数与结果的个数不匹配
#         if labels is None:
#             assert len(bboxes) == len(results), "bboxes和results的长度不匹配，存在未分割的对象"

#         num_objects = len(results)

#         # 初始化结果字典
#         result_dict = {
#             "orig_img": orig_img,
#             "masks": {
#                 "foreground": [],
#                 "background": []
#             },
#             "heatmaps": {
#                 "foreground": [],
#                 "background": None
#             },
#             "class_ids": labels
#         }

#         # 初始化前景掩码并集
#         all_masks = None

#         # 处理每个检测到的对象
#         for i in range(num_objects):
#             # 获取掩码
#             mask = results[i].masks.data[i].cpu().numpy().astype(np.uint8)
#             result_dict["masks"]["foreground"].append(mask)

#             # 累积前景掩码
#             if all_masks is None:
#                 all_masks = mask.copy()
#             else:
#                 all_masks = np.logical_or(all_masks, mask).astype(np.uint8)

#             # 生成热力图
#             if self.use_gaussian_blur:
#                 heatmap = self.create_heatmap_gaussian(mask)
#             else:
#                 # 如果不使用高斯模糊，直接使用掩码作为热力图
#                 heatmap = mask * 255

#             result_dict["heatmaps"]["foreground"].append(heatmap)

#             # 可视化
#             if visualize:
#                 # cv2.imshow(f"Object {i+1} Mask", mask * 255)
#                 cv2.imshow(f"Object {i if not labels else labels[i]} Heatmap", heatmap)

#         # 创建背景掩码
#         background_mask = np.ones_like(all_masks) - all_masks
#         result_dict["masks"]["background"].append(background_mask)  # 添加背景掩码

#         # 生成背景热力图
#         if self.use_gaussian_blur:
#             background_heatmap = self.create_heatmap_gaussian(background_mask)
#         else:
#             background_heatmap = background_mask * 255

#         result_dict["heatmaps"]["background"] = background_heatmap

#         # 可视化背景
#         if visualize:
#             # cv2.imshow("Background Mask", background_mask * 255)
#             cv2.imshow("Background Heatmap", background_heatmap)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

#         return result_dict


# # 示例用法
# if __name__ == "__main__":
#     # 创建热力图生成器
#     heatmap_gen = SAMHeatmapGenerator(
#         model_path="sam2.1_b.pt",
#         use_gaussian_blur=True,
#         blur_times=3,
#         kernel_size=101,
#         sigma=20
#     )

#     # 生成单张图像的热力图
#     result = heatmap_gen.generate_heatmaps(
#         "examples/truck.png",
#         bboxes=[[80, 120, 800, 380]],
#         visualize=True
#     )


# # ultralytics SAM2 demo
# """setup
# pip install ultralytics
# """
# import numpy as np
# import cv2
# from ultralytics import SAM

# # 使用高斯模糊生成热力图
# def create_heatmap_gaussian(mask, blur_times=3, kernel_size=101, sigma=20):
#     # 确保掩码是二值的
#     binary_mask = (mask > 0).astype(np.uint8) * 255
#     # 应用高斯模糊
#     blurred = cv2.GaussianBlur(binary_mask, (kernel_size, kernel_size), sigma)
#     for _ in range(blur_times):
#         blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), sigma)
#     return blurred


# # Load a model
# model = SAM("sam2.1_b.pt")

# # Display model information (optional)
# model.info()

# # Run inference with bboxes prompt
# results = model("examples/truck.png", bboxes=[[80, 120, 800, 380]])
# num_objects = len(results)
# print(f"Number of objects detected: {num_objects}")

# # 初始化一个全 0 的掩码，用于累积所有前景掩码
# all_masks = None

# for i in range(num_objects):
#     result = results[i]
#     orig_img = result.orig_img  # 原始图像
#     mask = result.masks.data[i].cpu().numpy().astype(np.uint8)  # 将布尔掩码转换为 uint8 格式

#     # 第一个掩码直接赋值，后续掩码通过逻辑或合并
#     if all_masks is None:
#         all_masks = mask.copy()
#     else:
#         all_masks = np.logical_or(all_masks, mask).astype(np.uint8)

#     # 高斯模糊方法
#     heatmap_gaussian = create_heatmap_gaussian(mask)
#     cv2.imshow(f"Object {i+1} Gaussian Heatmap", heatmap_gaussian)

# # 创建背景掩码 (对所有前景掩码的并集取反)
# background_mask = np.ones_like(all_masks) - all_masks
# heatmap_background = create_heatmap_gaussian(background_mask)
# cv2.imshow("Background Gaussian Heatmap", heatmap_background)

# # 等待键盘输入后继续
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # CN-CLIP demo
# """setup
# pip install -e .
# """
# import torch
# from PIL import Image

# import cn_clip.clip as clip
# from cn_clip.clip import load_from_name, available_models
# print("Available models:", available_models())
# # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

# import os

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = load_from_name(name=os.path.join(os.environ['DATAPATH'], 'pretrained_weights/clip_cn_vit-b-16.pt'),
#                                    device=device,
#                                    download_root='./',
#                                    vision_model_name="ViT-B-16",
#                                    text_model_name="RoBERTa-wwm-ext-base-chinese",
#                                    input_resolution=224)
# model.eval()
# image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
# text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     logits_per_image, logits_per_text = model.get_similarity(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
