from utils_algorithm.load_config import read_yaml
from detection_algorithm.yolo.detected import Detect
#from detection_algorithm.yolov5.detect import run
from pathlib import Path
import cv2 as cv
import time


current_path = Path(__file__).resolve().parents[0]  # 当前文件路径文件路径
cfg_path = str(Path.joinpath(current_path, 'config.yaml'))
cfg = read_yaml(cfg_path)
yolo_yaml = cfg['yolo']['classes']


def process(image):

    boot_path = Path(__file__).resolve().parents[0]
    weight_path = str(Path.joinpath(boot_path, 'weights/pytorch/yolov5s.pt'))
    v5_6 = "/home/winston/torch/yolov5s.pt"
    time_start = time.perf_counter()
    detects = Detect(v5_6, 160, 160, yolo_yaml, 0.3, 0.75, 20, 'cpu')

    data = detects.resize_zoom(image)
    frame, _, _ = detects.predict_(data, 'cpu', image, (0, 2, 3, 5))
    # 计算运行时间
    time_end = time.perf_counter()
    run_time = time_end - time_start
    run_time_ms = run_time * 1000
    print("run_time:" + str(run_time_ms))
    cv.imshow("1", image)
    cv.waitKey(0)
    return frame


if __name__ == '__main__':
    """im_path = str(Path.joinpath(current_path, '00004.jpg'))
    image = cv.imread(im_path)
    # process(image)
    
    """
    boot_path = Path(__file__).resolve().parents[2]
    im_path = str(Path.joinpath(current_path, 'bus.jpg'))
    weight_path = (Path.joinpath(current_path, 'weights/pytorch/yolov5n.pt'))
    detect1 = Detect(weight_path, (640, 640), 0.5, 0.5, 100, 'cpu', None, 'data/coco128.yaml', False, False, False, False, False, False)
    path = './11.jpeg'
    img = cv.imread(im_path)
    # 传入一张图片
    # img = detect1.resize_zoom(img)
    #im, _ = detect1.detect(img)
    detect_image, _ = detect1.detect(img)
    cv.namedWindow("detected", cv.WINDOW_FREERATIO)
    cv.imshow("detected", detect_image)
    cv.waitKey(0)
    cv.imwrite("fishing.jpeg", detect_image)
    # detect1.detect(img)
    # print(image.shape, im.shape)
    """cap = cv.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        if frame is not None:
            frame, _ = detect1.detect(frame)
            cv.imshow("Detected Window", frame)
            if cv.waitKey(50) == 27:
                break"""