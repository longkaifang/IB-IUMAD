#trian_split   one object


import os
import json
from glob import glob

def generate_json(dataset_root, output_json_path):
    """
    生成数据集路径的 JSON 文件，逐行写入，仅遍历 train/good/rgb 目录。

    Args:
        dataset_root (str): 数据集根目录。
        output_json_path (str): 输出的 JSON 文件路径。
    """
    # 打开输出文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        # 遍历数据集根目录
        for clsname in os.listdir(dataset_root):
            cls_dir = os.path.join(dataset_root, clsname)
            if not os.path.isdir(cls_dir):
                continue

            # 遍历 train/good/rgb 目录
            if clsname=="bagel":
                # print(clsname)
                for split in ["test"]:
                    split_dir = os.path.join(cls_dir, split)
                    # print(split_dir)
                    if not os.path.exists(split_dir):
                        continue

                    # 遍历 good 和 defect 目录
                    for label_name in ["good", "combined","contamination","crack","hole"]: #bagel
                    # for label_name in ["good", "bent","cut","thread","hole"]: #cable_gland
                    # for label_name in ["good", "combined","contamination","crack","hole","cut"]: #carrot
                    # for label_name in ["good", "combined","contamination","crack","hole"]: #cookie
                    # for label_name in ["good", "combined","contamination","cut","bent"]: #dowel
                    # for label_name in ["good", "combined","contamination","color","cut"]: #foam
                    # for label_name in ["good", "combined","contamination","cut","hole"]: #peach
                    # for label_name in ["good", "combined","contamination","cut","hole"]: #potato
                    # for label_name in ["good", "open","contamination","cut"]: #rope
                    # for label_name in ["good", "combined","contamination","cut","hole"]: #tire
                        label_dir = os.path.join(split_dir, label_name)
                        label_dir1 = os.path.join(label_dir + "/rgb")
                        # print(label_name)
                        if not os.path.exists(label_dir):
                            continue

                        # 遍历图片文件
                        for img_path in glob(os.path.join(label_dir1, "*.png")):  # 匹配 .jpg 或 .JPG
                            # 生成相对路径
                            # print(img_path)
                            xyz_path1 = img_path.replace("/rgb/", "/xyz/")
                            xyz_path2 = os.path.splitext(xyz_path1)[0] + ".tiff" 
                            gt_path1 = img_path.replace("/rgb/", "/gt/")
                            gt_path2 = os.path.splitext(gt_path1)[0] + ".png" 
                            # print(xyz_path)
                            relative_path = os.path.relpath(img_path, dataset_root)
                            relative_path1 = os.path.relpath(xyz_path2, dataset_root)
                            relative_path2 = os.path.relpath(gt_path2, dataset_root)
                            # print(label_name)
                            if label_name=="good":
                            # 构造 JSON 对象
                                json_obj = {
                                    "filename": relative_path.replace("\\", "/"),  # 统一路径格式
                                    "filename_xyz": relative_path1.replace("\\", "/"),
                                    "label": 0,  # good 图片的标签为 0
                                    "label_name": "good",  # 标签名称
                                }
                                # 将 JSON 对象写入文件，每行一个
                                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                            else:
                            # 构造 JSON 对象
                                json_obj = {
                                    "filename": relative_path.replace("\\", "/"),  # 统一路径格式
                                    "filename_xyz": relative_path1.replace("\\", "/"),
                                    "label": 1,  # good 图片的标签为 0
                                    "label_name": "defective",  # 标签名称
                                    "maskname":  relative_path2.replace("\\", "/")
                                }
                                # 将 JSON 对象写入文件，每行一个
                                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

                            

    print(f"JSON 文件已生成：{output_json_path}")


# 示例调用
dataset_root = "/home/admin1/2Tsdb/lkf/uniform-3dad/dataset/MVTec_3DAD/Data"
output_json_path = "/home/admin1/2Tsdb/lkf/uniform-3dad/dataset_split_code/MVTec3DAD_multi_test_json/bagel.json"
generate_json(dataset_root, output_json_path)