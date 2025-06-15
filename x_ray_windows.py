import sys
import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, \
    QLineEdit, QCheckBox, QComboBox, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import time


class DetectionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        try:
            self.model = YOLO(
                r"G:\Couresware\DeepLearning_project\YOLOv8_X-ray_image_inspection_of_dangerous_goods\x-ray-new\x-ray\trained_models\xray_detection_v1\weights\best.pt")
        except Exception as e:
            print(f"模型加载失败: {e}")
        self.image_paths = []
        self.current_index = 0
        self.all_results = []

    def initUI(self):
        self.setWindowTitle('基于YOLOv8的安检X光危险品检测识别系统')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: lightblue;")

        # 大标题
        title_label = QLabel('基于YOLOv8的安检X光危险品检测识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 30px;")

        # 左侧区域
        left_layout = QVBoxLayout()

        # X光图像展示区
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label)

        # 处理后图像展示区
        self.processed_image_label = QLabel()
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.processed_image_label)

        # 检测结果与位置信息表格
        table_title = QLabel('检测结果与位置信息')
        table_title.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(table_title)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels(['序号', '文件路径', '目标编号', '类别', '置信度', '坐标位'])
        # 设置表格大小
        self.result_table.setFixedSize(600, 300)  # 可根据需要调整大小
        left_layout.addWidget(self.result_table)

        # 右侧区域
        right_layout = QVBoxLayout()

        # 检测参数设置
        param_title = QLabel('检测参数设置')
        param_title.setStyleSheet("font-size: 20px;")
        right_layout.addWidget(param_title)

        conf_threshold_layout = QHBoxLayout()
        conf_threshold_label = QLabel('置信度阈值')
        self.conf_threshold_input = QLineEdit('0.25')
        conf_threshold_layout.addWidget(conf_threshold_label)
        conf_threshold_layout.addWidget(self.conf_threshold_input)
        right_layout.addLayout(conf_threshold_layout)

        iou_threshold_layout = QHBoxLayout()
        iou_threshold_label = QLabel('交并比阈值')
        self.iou_threshold_input = QLineEdit('0.70')
        iou_threshold_layout.addWidget(iou_threshold_label)
        iou_threshold_layout.addWidget(self.iou_threshold_input)
        right_layout.addLayout(iou_threshold_layout)

        # 检测结果
        result_title = QLabel('检测结果')
        result_title.setStyleSheet("font-size: 20px;")
        right_layout.addWidget(result_title)

        show_label_checkbox = QCheckBox('显示标签名称与置信度')
        right_layout.addWidget(show_label_checkbox)

        total_target_layout = QHBoxLayout()
        total_target_label = QLabel('总目标数')
        self.total_target_count = QLabel('0')
        total_target_layout.addWidget(total_target_label)
        total_target_layout.addWidget(self.total_target_count)
        right_layout.addLayout(total_target_layout)

        time_taken_layout = QHBoxLayout()
        time_taken_label = QLabel('用时')
        self.time_taken = QLabel('0.00 s')
        time_taken_layout.addWidget(time_taken_label)
        time_taken_layout.addWidget(self.time_taken)
        right_layout.addLayout(time_taken_layout)

        target_choice_layout = QHBoxLayout()
        target_choice_label = QLabel('目标选择')
        self.target_choice_combo = QComboBox()
        self.target_choice_combo.addItem('全部')
        target_choice_layout.addWidget(target_choice_label)
        target_choice_layout.addWidget(self.target_choice_combo)
        right_layout.addLayout(target_choice_layout)

        self.detail_result_label = QLabel()
        right_layout.addWidget(self.detail_result_label)

        # 操作按钮
        button_layout = QHBoxLayout()
        open_image_button = QPushButton('打开图片')
        open_folder_button = QPushButton('打开文件夹')
        next_round_button = QPushButton('下一轮次')
        save_button = QPushButton('保存')
        exit_button = QPushButton('退出')

        button_layout.addWidget(open_image_button)
        button_layout.addWidget(open_folder_button)
        button_layout.addWidget(next_round_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(exit_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)

        middle_layout = QHBoxLayout()
        middle_layout.addLayout(left_layout)
        middle_layout.addLayout(right_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # 按钮信号与槽连接
        open_image_button.clicked.connect(self.open_image)
        open_folder_button.clicked.connect(self.open_folder)
        next_round_button.clicked.connect(self.next_round)
        save_button.clicked.connect(self.save_results)
        exit_button.clicked.connect(self.close)

    def open_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.jpg *.jpeg)')
        if image_path:
            self.image_paths = [image_path]
            self.current_index = 0
            self.all_results = []
            self.process_images()

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder_path:
            self.image_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(root, file))
            self.current_index = 0
            self.all_results = []
            self.process_images()

    def next_round(self):
        if self.current_index >= len(self.image_paths):
            print("已经没有更多图片了。")
            return
        next_images = self.image_paths[self.current_index:self.current_index + 9]
        self.current_index += 9
        self.process_images(next_images)

    def process_images(self, images=None):
        if images is None:
            images = self.image_paths[:9]
        total_time = 0
        total_targets = 0
        self.result_table.setRowCount(0)

        # 获取阈值参数
        try:
            conf_threshold = float(self.conf_threshold_input.text())
        except ValueError:
            conf_threshold = 0.25  # 默认值
            self.conf_threshold_input.setText('0.25')

        try:
            iou_threshold = float(self.iou_threshold_input.text())
        except ValueError:
            iou_threshold = 0.70  # 默认值
            self.iou_threshold_input.setText('0.70')

        for image_path in images:
            try:
                start_time = time.time()
                img = cv2.imread(image_path)
                if img is None:
                    print(f"图像读取失败: {image_path}")
                    continue

                # 使用获取的阈值参数进行推理
                results = self.model(img, conf=conf_threshold, iou=iou_threshold)

                end_time = time.time()
                time_taken = end_time - start_time
                total_time += time_taken

                # 更新用时显示
                self.time_taken.setText(f"{time_taken:.2f} s")

                total_targets += len(results[0].boxes)

                for i, box in enumerate(results[0].boxes):
                    cls = results[0].names[int(box.cls)]
                    conf = float(box.conf)
                    xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
                    row_position = self.result_table.rowCount()
                    self.result_table.insertRow(row_position)
                    self.result_table.setItem(row_position, 0,
                                              QTableWidgetItem(str(len(self.all_results) + row_position + 1)))
                    self.result_table.setItem(row_position, 1, QTableWidgetItem(image_path))
                    self.result_table.setItem(row_position, 2, QTableWidgetItem(str(i + 1)))
                    self.result_table.setItem(row_position, 3, QTableWidgetItem(cls))
                    self.result_table.setItem(row_position, 4, QTableWidgetItem(f'{conf * 100:.2f}%'))
                    self.result_table.setItem(row_position, 5, QTableWidgetItem(f'({xmin}, {ymin}, {xmax}, {ymax})'))
                    self.all_results.append(
                        [len(self.all_results) + row_position + 1, image_path, i + 1, cls, f'{conf * 100:.2f}%',
                         f'({xmin}, {ymin}, {xmax}, {ymax})'])

                # 显示最后一张图像
                if image_path == images[-1]:
                    # 显示原始图像
                    height, width, channel = img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_img)
                    # 可根据需要调整图片缩放大小
                    self.image_label.setPixmap(
                        pixmap.scaled(400, 300, Qt.KeepAspectRatio))

                    # 显示处理后的图像
                    processed_img = img.copy()
                    for box in results[0].boxes:
                        cls = results[0].names[int(box.cls)]
                        conf = float(box.conf)
                        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(processed_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        cv2.putText(processed_img, f'{cls} {conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (255, 0, 0), 2)

                    height, width, channel = processed_img.shape
                    bytes_per_line = 3 * width
                    q_processed_img = QImage(processed_img.data, width, height, bytes_per_line,
                                             QImage.Format_RGB888).rgbSwapped()
                    processed_pixmap = QPixmap.fromImage(q_processed_img)
                    # 可根据需要调整图片缩放大小
                    self.processed_image_label.setPixmap(
                        processed_pixmap.scaled(400, 300, Qt.KeepAspectRatio))

            except Exception as e:
                print(f"检测过程中出现错误: {e}")

        self.total_target_count.setText(str(total_targets))
        self.time_taken.setText(f'{total_time:.3f} s')

    def save_results(self):
        df = pd.DataFrame(self.all_results, columns=['序号', '文件路径', '目标编号', '类别', '置信度', '坐标位'])
        save_dir = r'.\保存历史检测记录'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, 'detection_results.xlsx')
        try:
            df.to_excel(file_path, index=False)
            print(f"检测结果已保存为 {file_path}")
        except Exception as e:
            print(f"保存结果时出现错误: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = DetectionUI()
    ui.show()
    sys.exit(app.exec_())
