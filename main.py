# ultralytics SAM2 demo
"""setup
pip install ultralytics
pip install scipy  # 需要安装scipy用于KDE方法
"""
import numpy as np
import cv2
from ultralytics import SAM
from scipy.spatial.distance import cdist

# 使用高斯模糊生成热力图
def create_heatmap_gaussian(mask, blur_times=3, kernel_size=101, sigma=20):
    # 确保掩码是二值的
    binary_mask = (mask > 0).astype(np.uint8) * 255
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(binary_mask, (kernel_size, kernel_size), sigma)
    for _ in range(blur_times):
        blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), sigma)
    return blurred


# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

# Run inference with bboxes prompt
results = model("examples/truck.png", bboxes=[[80, 120, 800, 380]], labels=[1])
num_objects = len(results)
print(f"Number of objects detected: {num_objects}")

for i in range(num_objects):
    result = results[i]
    orig_img = result.orig_img  # 原始图像
    mask = result.masks.data[i].cpu().numpy().astype(np.uint8)  # 将布尔掩码转换为 uint8 格式

    # 高斯模糊方法
    heatmap_gaussian = create_heatmap_gaussian(mask, kernel_size=51, sigma=10)
    cv2.imshow(f"Object {i+1} Gaussian Heatmap", heatmap_gaussian)

    # 等待键盘输入后继续
    cv2.waitKey(0)

cv2.destroyAllWindows()


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
