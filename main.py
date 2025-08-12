import Camera, Detect, Debug, UI,  my_serial
import toml
import sys
from PyQt5.QtWidgets import  QApplication
from post_process import Post_process

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 读取参数
    config = toml.load("./config/config.toml")

    # 创建线程
    print("start to init threads")
    debuger_thread = None
    if config["debug"]:
        debuger_thread = Debug.Debuger()
    camera_thread = Camera.Camera(debuger_thread)
    if config["detect"]:
        detector_thread = Detect.Detector(debuger_thread)
    post_process_thread = Post_process.Post_processor(debuger_thread)

    if config["use_serial"]:
        serial_thread = my_serial.Serial(debuger_thread)
    print("thread init finished!")

    main_window_thread = UI.Main_window(post_process_thread, debuger_thread)
    main_window_thread.restart_camera.connect(camera_thread.restart)
    # 移动窗口到左上角 
    main_window_thread.move(0, 0)
    main_window_thread.show()

    # 启动线程
    camera_thread.start()
    if config["detect"]:
        detector_thread.start()
    main_window_thread.start()
    post_process_thread.start()
    if config["use_serial"]:
        serial_thread.start()
    if config["debug"]:
        debuger_thread.start()

    # 阻塞主程序等待线程运行完
    app.exec_()
    camera_thread.end()
    if config["detect"]:
        detector_thread.end()
    main_window_thread.end()
    post_process_thread.end()
    if config["use_serial"]:
        serial_thread.end()
    if config["debug"]:
        debuger_thread.end()

    camera_thread.wait()
    if config["detect"]:
        detector_thread.wait()
    main_window_thread.wait()
    post_process_thread.wait()
    if config["use_serial"]:
        serial_thread.wait()
    if config["debug"]:
        debuger_thread.wait()
