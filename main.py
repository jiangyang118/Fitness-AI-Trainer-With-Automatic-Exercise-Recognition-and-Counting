import streamlit as st
# from Audio import text_to_speech, get_audio
import cv2
import tempfile
import ExerciseAiTrainer as exercise
from chatbotDeepseek import chat_ui
# from chatbot import chat_ui
import time
import logging
from contextlib import AbstractContextManager
from typing import Union

# === 健壮性封装与监控：VideoCapture 安全包装 ===
class SafeVideoCapture(AbstractContextManager):
    """对 cv2.VideoCapture 的安全封装，带自动释放与基础监控（帧数/耗时/FPS）。"""
    def __init__(self, source: Union[int, str]):
        # 记录输入源（0/1 表示摄像头索引；字符串表示视频文件路径）
        self.source = source
        self.cap = None           # 底层 OpenCV 句柄
        self.ok = False           # 打开是否成功
        self.frames = 0           # 成功读取的帧数
        self.total_read_secs = 0  # 读取帧的累计耗时（秒）
        self._t0 = None           # 打开时间，用于计算运行时长
        try:
            # 打开视频源：文件或摄像头
            self.cap = cv2.VideoCapture(self.source)
            self.ok = bool(self.cap and self.cap.isOpened())
            if self.ok:
                self._t0 = time.time()
        except Exception as e:
            logging.exception("SafeVideoCapture 打开失败: %s", e)
            self.ok = False

    def __enter__(self):
        # 允许 with 语法使用
        return self

    def __exit__(self, exc_type, exc, tb):
        # 退出上下文保证释放资源；返回 False 让异常继续抛出（Streamlit 会展示）
        self.release()
        return False

    def read(self):
        """安全读取一帧，并记录耗时与帧计数。"""
        if not self.ok or self.cap is None:
            return False, None
        t1 = time.time()
        ret, frame = self.cap.read()
        t2 = time.time()
        if ret:
            self.frames += 1
            self.total_read_secs += (t2 - t1)
        return ret, frame

    def release(self):
        """安全释放底层资源。"""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as e:
            logging.warning("释放 VideoCapture 失败: %s", e)
        finally:
            self.cap = None
            self.ok = False

    # 监控指标便捷属性
    @property
    def elapsed(self) -> float:
        """从打开到当前的总时长（秒）。"""
        return 0.0 if self._t0 is None else (time.time() - self._t0)

    @property
    def avg_read_ms(self) -> float:
        """平均每帧读取耗时（毫秒）。"""
        return 0.0 if self.frames == 0 else (self.total_read_secs / self.frames) * 1000.0

    @property
    def fps(self) -> float:
        """粗略估算 FPS（读取成功帧数 / 总时长）。"""
        el = self.elapsed
        return 0.0 if el <= 0 else (self.frames / el)

def render_capture_metrics(cap_obj: SafeVideoCapture, location: str = "sidebar"):
    """在页面上渲染监控指标；location 可为 'sidebar' 或 'main'。"""
    container = st.sidebar if location == "sidebar" else st
    container.markdown("### 🎯 视频源监控")
    c1, c2, c3 = container.columns(3)
    c1.metric("累计帧数", cap_obj.frames)
    c2.metric("平均读帧耗时", f"{cap_obj.avg_read_ms:.1f} ms")
    c3.metric("估算 FPS", f"{cap_obj.fps:.2f}")
    container.caption(f"运行时长：{cap_obj.elapsed:.1f} s")


# （可选）保留：用于实验性“用户自行选择 视频/摄像头”的旧入口；当前 Video/WebCam 流程未使用
def get_capture():
    src = st.radio("选择视频源", ["上传视频", "摄像头"], horizontal=True)
    cap = None
    if src == "上传视频":
        f = st.file_uploader("上传视频", type=["mp4","mov","avi","mkv"])
        if f:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t.write(f.read()); t.flush()
            cap = cv2.VideoCapture(t.name)
    else:
        cap = cv2.VideoCapture(0)  # 如有多摄像头可做成下拉选择

    if not cap or not cap.isOpened():
        st.warning("⚠️ 未获取到有效视频源，请先上传视频或允许摄像头权限。")
        return None
    return cap


def main():
    # 页面基础设置
    st.set_page_config(page_title='Fitness AI Coach', layout='centered')
    st.title('Fitness AI Coach')

    # 顶部功能选择
    options = st.sidebar.selectbox('Select Option', ('Video', 'WebCam', 'Auto Classify', 'chatbot'))

    # 聊天：DeepSeek
    if options == 'chatbot':
        st.markdown('-------')
        st.markdown("The chatbot can make mistakes. Check important info.")
        chat_ui()

    # ========== 离线视频分析（上传视频优先；未上传回退摄像头） ==========
    if options == 'Video':
        st.markdown('-------')
        st.write('## Upload your video and select the correct type of Exercise to count repetitions')
        st.write("")
        st.write('Please ensure you are clearly visible and facing the camera directly. This will help the AI accurately track your movements.')

        st.sidebar.markdown('-------')

        # 选择动作类型（影响后续调用哪个算法）
        exercise_options = st.sidebar.selectbox(
            'Select Exercise', ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )
        st.sidebar.markdown('-------')

        # 侧边栏上传视频
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
        tfflie = tempfile.NamedTemporaryFile(delete=False)

        # 展示“输入视频”的占位（这里直接展示临时文件路径，上传后会被写入）
        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)
        st.markdown('## Input Video')
        st.video(tfflie.name)
        st.markdown('-------')

        # 输出分析区域
        st.markdown(' ## Output Video')

        # === 统一数据源：优先使用侧栏上传的视频；否则回退摄像头 ===
        if video_file_buffer is not None:
            try:
                tfflie.write(video_file_buffer.read())
                tfflie.flush()
                source: Union[int, str] = tfflie.name
            except Exception as e:
                st.error(f"写入临时文件失败：{e}")
                st.stop()
        else:
            source = 0  # 回退到默认摄像头

        # 用安全包装管理整个生命周期
        with SafeVideoCapture(source) as safe_cap:
            if not safe_cap.ok:
                st.warning("⚠️ 无法打开视频源，请检查文件有效性或授予摄像头权限。")
                st.stop()
            cap = safe_cap.cap  # 传给现有算法

            try:
                if exercise_options == 'Bicept Curl':
                    # 二头弯举：左右臂计数
                    exer = exercise.Exercise()
                    counter, stage_right, stage_left = 0, None, None
                    exer.bicept_curl(cap, is_video=True, counter=counter,
                                     stage_right=stage_right, stage_left=stage_left)

                elif exercise_options == 'Push Up':
                    # 俯卧撑：面向正面或左侧
                    st.write("The exercise need to be filmed showing your left side or facing frontally")
                    exer = exercise.Exercise()
                    counter, stage = 0, None
                    exer.push_up(cap, is_video=True, counter=counter, stage=stage)

                elif exercise_options == 'Squat':
                    # 深蹲：计数与阶段判断
                    exer = exercise.Exercise()
                    counter, stage = 0, None
                    exer.squat(cap, is_video=True, counter=counter, stage=stage)

                elif exercise_options == 'Shoulder Press':
                    # 肩上推举
                    exer = exercise.Exercise()
                    counter, stage = 0, None
                    exer.shoulder_press(cap, is_video=True, counter=counter, stage=stage)
            finally:
                # 无论成功失败都输出监控指标到侧边栏
                render_capture_metrics(safe_cap, location="sidebar")

    # ========== 自动识别 & 计数（内置分类） ==========
    elif options == 'Auto Classify':
        st.markdown('-------')
        st.write('Click button to start automatic exercise classification and repetition counting')
        st.markdown('-------')
        st.write("Please ensure you are clearly visible and facing the camera directly. This will help the AI accurately track your movements.")
        auto_classify_button = st.button('Start Auto Classification')

        if auto_classify_button:
            time.sleep(2)  # 简单延迟，提升体验
            exer = exercise.Exercise()
            # 内部应自行打开摄像头/视频并完成分类与计数
            exer.auto_classify_and_count()

    # ========== WebCam（在线摄像头） ==========
    elif options == 'WebCam':
        st.markdown('-------')
        st.sidebar.markdown('-------')

        # 选择动作类型（与 Video 一致，但数据源固定为摄像头）
        exercise_general = st.sidebar.selectbox(
            'Select Exercise', ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )

        st.write(' 点击按钮开始基于摄像头的训练（请确保授权摄像头权限）')
        start_button = st.button('Start Exercise')

        if start_button:
            time.sleep(1)  # 短暂延迟让摄像头“热身”
            ready = True

            # 统一使用 SafeVideoCapture(0) 打开默认摄像头
            # 使用 with 确保异常/完成后自动释放摄像头资源
            with SafeVideoCapture(0) as safe_cap:
                if not safe_cap.ok:
                    st.warning("⚠️ 无法打开摄像头，请检查权限或设备连接。")
                    st.stop()
                cap = safe_cap.cap

                try:
                    # 和 Video 分支一致，直接复用算法接口
                    if exercise_general == 'Bicept Curl':
                        while ready:
                            exer = exercise.Exercise()
                            counter, stage_right, stage_left = 0, None, None
                            # 这里不传 is_video=True，沿用原作者 webcam 版本接口
                            exer.bicept_curl(cap, counter=counter, stage_right=stage_right, stage_left=stage_left)
                            break

                    elif exercise_general == 'Push Up':
                        while ready:
                            exer = exercise.Exercise()
                            counter, stage = 0, None
                            exer.push_up(cap, counter=counter, stage=stage)
                            break

                    elif exercise_general == 'Squat':
                        while ready:
                            exer = exercise.Exercise()
                            counter, stage = 0, None
                            exer.squat(cap, counter=counter, stage=stage)
                            break

                    elif exercise_general == 'Shoulder Press':
                        while ready:
                            exer = exercise.Exercise()
                            counter, stage = 0, None
                            exer.shoulder_press(cap, counter=counter, stage=stage)
                            break
                finally:
                    # WebCam 模式下同样输出监控指标
                    render_capture_metrics(safe_cap, location="sidebar")


if __name__ == '__main__':
    # 简单日志配置：你可按需调整级别/格式或输出到文件
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    # 可选：加载自定义样式
    def load_css():
        try:
            with open("static/styles.css", "r") as f:
                css = f"<style>{f.read()}</style>"
                st.markdown(css, unsafe_allow_html=True)
        except FileNotFoundError:
            # 没有样式文件也不影响运行
            pass

    main()