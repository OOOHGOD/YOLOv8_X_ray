import os
import requests


def download_model(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"成功下载模型到 {save_path}")
    except requests.RequestException as e:
        print(f"下载失败: {e}")


if __name__ == "__main__":
    # 创建保存模型的目录
    save_dir = './pre_train_models'
    os.makedirs(save_dir, exist_ok=True)

    # 定义模型的下载链接
    model_urls = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
         "yolov7-tiny.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
        "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
    }

    # 下载每个模型
    for model_name, url in model_urls.items():
        save_path = os.path.join(save_dir, model_name)
        download_model(url, save_path)
