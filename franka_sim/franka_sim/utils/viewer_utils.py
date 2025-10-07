import mujoco.viewer
import subprocess
import time


class DualMujocoViewer:
    def __init__(
            self,
            model,
            data,
            key_callback=None
    ):
        self.model = model
        self.data = data
        self.viewer_1 = None
        self.viewer_2 = None
        self.key_callback = key_callback

    def __enter__(self):
        self.launch()
        self.set_window_positions()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def launch(self):
        self.viewer_1 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback
        )
        self.viewer_2 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback
        )

    def set_window_positions(self):
        time.sleep(1)
        try:
            output = subprocess.check_output(['wmctrl', '-l']).decode('utf-8')
            window_ids = []
            for line in output.splitlines():
                if 'MuJoCo' in line:
                    window_ids.append(line.split()[0])

            if len(window_ids) >= 2:
                screen_info = subprocess.check_output(['xrandr']).decode('utf-8')
                for line in screen_info.splitlines():
                    if '*' in line:
                        width, height = map(int, line.split()[0].split('x'))
                        break

                half_width = width // 2
                half_height = height // 2

                # 设置第一个窗口在屏幕左上角
                x1 = 0
                y1 = 0
                width1 = half_width
                height1 = half_height
                subprocess.call(['wmctrl', '-i', '-r', window_ids[0], '-e', f'0,{x1},{y1},{width1},{height1}'])

                # 设置第二个窗口在屏幕右上角
                x2 = half_width
                y2 = 0
                width2 = half_width
                height2 = half_height
                subprocess.call(['wmctrl', '-i', '-r', window_ids[1], '-e', f'0,{x2},{y2},{width2},{height2}'])
        except Exception as e:
            print(f"设置窗口位置时发生错误: {e}")
            
    def is_running(self):
        return self.viewer_1.is_running() and self.viewer_2.is_running()

    def sync(self):
        if self.viewer_1:
            self.viewer_1.sync()
        if self.viewer_2:
            self.viewer_2.sync()

    def close(self):
        import glfw
        if self.viewer_1:
            self.viewer_1.close()
        if self.viewer_2:
            self.viewer_2.close()
        glfw.terminate()
