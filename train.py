import os
import sys
import torch
from ultralytics import YOLO


def main():
    # 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # 优先加载 PyTorch 的 DLL
    torch_dir = os.path.dirname(torch.__file__)
    dll_path = os.path.join(torch_dir, 'lib')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(dll_path)

    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # --- 1. 设置本地路径变量 ---
    # 你需要根据你的实际文件夹结构修改这些路径
    # 确保使用正斜杠 '/' 或双反斜杠 '\\' 以避免路径问题
    base_path = "./"

    # data.yaml路径，确保它指向你的本地文件
    data_yaml_path = os.path.join(base_path, "yolo_dataset/xray_data.yaml")

    # 预训练模型路径
    # YOLO会自动下载yolov8n.pt，如果本地没有。
    # 如果你确实想使用自己下载的，可以取消下面这行的注释并确保路径正确。
    # pretrained_model_path = os.path.join(base_path, "pre_trained_model/yolov8n.pt")
    # model = YOLO(pretrained_model_path) # 使用本地预训练模型
    model = YOLO('./pre_train_models/yolov8n.pt')  # 或者直接让YOLO下载/使用缓存的yolov8n.pt

    # --- 2. 训练参数配置 ---
    # RTX 3060 Laptop (通常6GB VRAM), 16GB系统RAM
    # batch: 4 进一步减小，看看是否能解决问题
    # workers: 2 保持不变
    train_params = {
        'data': data_yaml_path,
        'epochs': 100,
        'batch': 4,          # RTX 3060 Laptop 建议8-16，根据显存调整
        'imgsz': 640,
        'device': 'cuda' if cuda_available else 'cpu',
        'optimizer': 'AdamW',# YOLOv8默认是AdamW或SGD，AdamW通常效果不错
        'lr0': 0.001,        # 减小学习率
        'lrf': 0.1,          # 调整最终学习率
        'augment': True,     # 自动启用基础数据增强
        'mosaic': 0.75,      # 启用马赛克增强的概率 (YOLOv8中mosaic默认在最后10个epoch关闭，此处设置可能被覆盖或有不同行为)
        'hsv_h': 0.015,      # 色调增强幅度
        'hsv_s': 0.7,        # 饱和度增强幅度
        'hsv_v': 0.4,        # 明度增强幅度
        'translate': 0.1,    # 平移增强幅度
        'scale': 0.5,        # 缩放增强幅度
        'fliplr': 0.5,       # 水平翻转概率
        'workers': 2,        # 根据你的CPU核心数和数据加载情况调整，2可能是一个不错的起点
        'project': os.path.join(base_path, "trained_models"),  # 训练输出的根目录
        'name': 'xray_detection_v1',  # 实验名称，会在此目录下创建 xray_detection_v1 文件夹
        'exist_ok': True,    # 允许覆盖同名实验文件夹下的内容
        'seed': 42           # 固定随机种子，保证结果可复现
    }

    # --- 3. 开始训练 ---
    print(f"Starting training with parameters: {train_params}")
    results = model.train(**train_params)
    print("Training completed.")
    print(f"Results saved to: {results.save_dir}")  # 显示模型和结果保存的实际路径

    # --- 4. 自动验证最佳模型（使用验证集） ---
    # model.val() 会自动加载训练过程中保存的最佳模型 (best.pt)
    # 它会使用 data_yaml_path 中定义的验证集
    print("Starting validation on the validation set...")
    metrics = model.val()  # metrics 对象包含各类评估指标，如 mAP50-95, mAP50 等
    print(f"Validation metrics: {metrics.box.map}")  # 打印 mAP50-95

    # --- 5. 测试集评估 (可选) ---
    # 确保你的 data.yaml 中已配置 'test' 集路径，并且该路径下有图像
    # 例如: test: test/images (在 xray_data.yaml 中)
    # 并且 D:/Projects/XRay_Detection/yolo_dataset/test/images 文件夹存在
    test_set_path_in_yaml = train_params['data']  # 这是yaml文件的路径
    # 检查data.yaml中是否配置了test集，并且实际文件/文件夹是否存在
    # (这一步的自动检查比较复杂，依赖yaml解析，通常YOLO内部会处理路径问题)
    # 简单起见，我们假设如果配置了test，YOLO能找到它。

    # model.val() 会自动使用在训练时使用的配置文件中定义的 'val' 或 'test' split
    # 要在测试集上评估，你需要确保 'data.yaml' 中有 'test:' 指向你的测试集图片目录
    # 并且该目录存在。
    # 如果 yaml 中有 test 集的定义：
    print("Attempting evaluation on the test set (if configured in YAML)...")
    try:
        test_results = model.val(
            split='test',       # 指定使用测试集
            save_json=True,     # 生成COCO格式的JSON评估结果
            save_hybrid=True    # 保存混合标签（图片+预测框）以分析误检
        )
        print(f"Test set metrics: {test_results.box.map}")
        print(f"Test results saved to directory: {test_results.save_dir}")
    except Exception as e:
        print(f"Could not evaluate on test set. Ensure 'test' split is defined in your '{data_yaml_path}' and the path is correct. Error: {e}")

    # --- 6. 预测示例 ---
    # 确保测试图片路径正确
    test_image_path = os.path.join(base_path, "yolo_dataset/test/images/test.jpg")  # 修改为你的测试图片路径
    # 你可能需要从你的测试集里选一张实际存在的图片，或者提供一个样例图片
    # 例如，我们假设有一张名为 "sample_test_image.jpg" 的图片在项目根目录下
    # test_image_path = os.path.join(base_path, "sample_test_image.jpg")  # 替换为你的图片

    if os.path.exists(test_image_path):
        print(f"Running prediction on: {test_image_path}")
        # 使用训练后加载的最佳模型进行预测
        # model.predict() 会自动加载 'best.pt' (如果当前 model 实例是刚训练完的)
        # 或者你可以显式加载: model = YOLO(os.path.join(train_params['project'], train_params['name'], 'weights/best.pt'))
        prediction_results = model.predict(
            source=test_image_path,
            conf=0.25,        # 置信度阈值 (YOLO默认0.25，可以按需调整)
            iou=0.45,         # NMS IoU阈值 (YOLO默认0.7，你之前是0.45)
            save=True,        # 保存带标注的图片
            save_txt=True,    # 保存txt格式的标签结果
            save_conf=True,   # 在txt标签中保存置信度
            exist_ok=True,    # 如果输出文件夹已存在，则覆盖
            project=os.path.join(base_path, "predictions"),  # 指定预测结果保存的根目录
            name="predict_run1"  # 预测结果将保存在 predictions/predict_run1 中
        )
        print(f"Prediction results saved. Check 'predictions/predict_run1' directory.")
    else:
        print(f"Test image not found at: {test_image_path}. Skipping prediction.")

    # --- 7. 模型导出 ---
    # 导出目录
    export_dir = os.path.join(base_path, "trained_models")
    os.makedirs(export_dir, exist_ok=True)

    # 导出ONNX格式 (加载训练好的最佳模型)
    # best_model_path = os.path.join(train_params['project'], train_params['name'], 'weights/best.pt')
    # best_model = YOLO(best_model_path) # 确保加载的是最佳模型

    print("Exporting model to ONNX format...")
    try:
        # model 实例在训练和验证后，通常指的是最后或最佳模型
        onnx_path = model.export(
            format='onnx',
            imgsz=640,
            dynamic=False,   # 固定输入尺寸
            simplify=True,   # 简化ONNX模型
            opset=12,        # ONNX版本
            # file 参数已弃用，会自动命名或使用默认。若要指定，用在YOLO()加载时
        )
        # export() 返回导出模型的路径, 我们重命名并移动到期望位置
        # onnx_path 是一个 pathlib.Path 对象
        final_onnx_name = 'xray_detector_model.onnx'  # 你之前用的是 dog_classifier.onnx
        final_onnx_path = os.path.join(export_dir, final_onnx_name)
        # 如果 onnx_path 是 Path 对象，使用 os.replace
        if os.path.exists(str(onnx_path)):  # onnx_path 是 export() 的返回值
            os.replace(str(onnx_path), final_onnx_path)
            print(f"Model exported to ONNX: {final_onnx_path}")
        else:
            print(f"Failed to find exported ONNX model at {onnx_path}")

    except Exception as e:
        print(f"Error exporting ONNX model: {e}")

    # 保存PyTorch格式 (通常 best.pt 和 last.pt 会在训练时自动保存在 runs/detect/trainX/weights/ 目录下)
    # model.save() 是一个辅助函数，可以将当前模型状态保存到指定路径
    # 这里我们假设 model 指向的是训练后的最佳模型 (YOLO对象在 .train() 后会更新其内部模型状态)
    pytorch_model_path = os.path.join(export_dir, 'xray_detector_model.pt')  # 你之前用的是 dog_classifier.pt
    model.save(pytorch_model_path)
    print(f"PyTorch model saved to: {pytorch_model_path}")

    print("\nLocal training and processing workflow completed!")
    print(f"All trained models and results should be in: {os.path.join(train_params['project'], train_params['name'])}")
    print(f"Exported models should be in: {export_dir}")


if __name__ == "__main__":
    main()