import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


# 数据集路径
data_path = r"G:\Couresware\DeepLearning_project\YOLOv8_X-ray_image_inspection_of_dangerous_goods\x-ray\HiXray-dataset"
# 目标保存路径
yolo_dataset_path = r"G:\Couresware\DeepLearning_project\YOLOv8_X-ray_image_inspection_of_dangerous_goods\x-ray\yolo_dataset"
os.makedirs(yolo_dataset_path, exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_path, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_path, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_path, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_path, 'labels', 'test'), exist_ok=True)

# 类别列表
classes = [
    'Portable_Charger_1',
    'Portable_Charger_2',
    'Mobile_Phone',
    'Cosmetic',
    'Nonmetallic_Lighter',
    'Water',
    'Tablet',
    'Laptop'
]


# 转换标签格式函数
def convert_label_format(label_path, image_width, image_height):
    new_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Error: Invalid line format in {label_path}: {line}")
                continue
            img_name, class_name, x1, y1, x2, y2 = parts
            try:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            except ValueError:
                print(f"Error: Coordinates in {label_path} are not valid integers: {line}")
                continue
            if class_name not in classes:
                print(f"Error: Unknown class {class_name} in {label_path}")
                continue
            class_id = classes.index(class_name)
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            new_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return new_labels


# 复制并转换文件函数
def copy_and_convert_files(image_path, annotation_path, image_dst_folder, label_dst_folder):
    for file in os.listdir(image_path):
        img_name = os.path.splitext(file)[0]
        img_src = os.path.join(image_path, file)
        label_src = os.path.join(annotation_path, f'{img_name}.txt')
        img_dst = os.path.join(image_dst_folder, file)
        label_dst = os.path.join(label_dst_folder, f'{img_name}.txt')
        if os.path.exists(label_src):
            try:
                shutil.copy(img_src, img_dst)
                img = Image.open(img_src)
                width, height = img.size
                new_labels = convert_label_format(label_src, width, height)
                with open(label_dst, 'w') as f:
                    f.write('\n'.join(new_labels))
            except Exception as e:
                print(f"Error processing {label_src}: {e}")


# 多线程复制和转换训练集和测试集
with ThreadPoolExecutor() as executor:
    train_image_path = os.path.join(data_path, 'train', 'train_image')
    train_annotation_path = os.path.join(data_path, 'train', 'train_annotation')
    train_image_dst = os.path.join(yolo_dataset_path, 'images', 'train')
    train_label_dst = os.path.join(yolo_dataset_path, 'labels', 'train')
    executor.submit(copy_and_convert_files, train_image_path, train_annotation_path, train_image_dst, train_label_dst)

    test_image_path = os.path.join(data_path, 'test', 'test_image')
    test_annotation_path = os.path.join(data_path, 'test', 'test_annotation')
    test_image_dst = os.path.join(yolo_dataset_path, 'images', 'test')
    test_label_dst = os.path.join(yolo_dataset_path, 'labels', 'test')
    executor.submit(copy_and_convert_files, test_image_path, test_annotation_path, test_image_dst, test_label_dst)


# 创建数据配置文件
data_yaml = f"""
train: {os.path.join(yolo_dataset_path, 'images', 'train')}
val: {os.path.join(yolo_dataset_path, 'images', 'test')}
nc: {len(classes)}
names: {classes}
"""

with open(os.path.join(yolo_dataset_path, 'xray_data.yaml'), 'w') as f:
    f.write(data_yaml)