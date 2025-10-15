import tkinter as tk
from tkinter import messagebox, ttk
import threading
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import ollama
import time

# --- 1. 默认配置区域 ---
CONFIG = {
    # Whisper 模型配置
    "stt_model_size": "distil-large-v3.5",
    "device": "cuda",
    "compute_type": "int8",
    # 语言配置
    "source_language_code": "ko",
    "source_language_name": "韩语",
    "target_language": "中文",
    # Ollama 模型配置
    "ollama_model": 'gemma2:9b-instruct-q4_K_M',
    # Ollama 提示词模板
    "ollama_prompt_template": "Translate the following {source_language} text to {target_language}. "
                              "Only output the translated content, without any explanation or original text. "
                              "Text: '{text}'",
    # 音频捕获配置
    "audio_device_name_substring": "CABLE Output",
    "sample_rate": 16000,
    "interval_seconds": 2,
    # 字幕窗口配置
    "window_width": 1200,
    "background_color": 'black',
    "original_text_color": '#AAAAAA',
    "translated_text_color": '#FFFF00',
    "font_family": 'Microsoft YaHei',
    "font_size": 28,
    "font_weight": 'bold'
}

# 推荐的 Ollama 模型列表
OLLAMA_MODEL_CHOICES = [
    'gemma2:9b-instruct-q4_K_M',
    'gemma3:8b',
    'gemma3:12b',
    'gemma3:27b',
    'qwen2.5:7b',
    'llama3:8b'
]


# --- 2. 设置窗口实现 (已升级) ---
class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, current_config, apply_callback):
        super().__init__(parent)
        self.title("设置")
        self.geometry("600x480")
        self.configure(bg='#2b2b2b')
        self.parent = parent
        self.config = current_config.copy()
        self.apply_callback = apply_callback
        self.entries = {}

        # 设置项映射
        settings = {
            "stt_model_size": "Whisper 模型大小:",
            "ollama_model": "Ollama 模型:",
            "source_language_name": "源语言 (名称):",
            "source_language_code": "源语言 (代码):",
            "target_language": "目标语言:",
            "ollama_prompt_template": "Ollama 提示词模板:"
        }

        frame = tk.Frame(self, bg='#2b2b2b')
        frame.pack(padx=20, pady=20, fill='both', expand=True)

        for key, label_text in settings.items():
            row = tk.Frame(frame, bg='#2b2b2b')
            row.pack(fill='x', pady=5)

            label = tk.Label(row, text=label_text, width=20, anchor='w', fg='white', bg='#2b2b2b')
            label.pack(side='left')

            if key == "ollama_prompt_template":
                entry = tk.Text(row, height=5, width=40, fg='white', bg='#3c3f41', insertbackground='white')
                entry.insert('1.0', self.config.get(key, ''))
            elif key == "ollama_model":
                # --- 升级：使用 Combobox 允许选择或输入 ---
                entry = ttk.Combobox(row, width=38, values=OLLAMA_MODEL_CHOICES)
                entry.set(self.config.get(key, ''))
            else:
                entry = tk.Entry(row, width=40, fg='white', bg='#3c3f41', insertbackground='white')
                entry.insert(0, self.config.get(key, ''))

            entry.pack(side='left', fill='x', expand=True)
            self.entries[key] = entry

        save_button = tk.Button(self, text="保存并应用", command=self.save_and_apply, bg='#007acc', fg='white',
                                relief='flat', font=('Arial', 10, 'bold'))
        save_button.pack(pady=20, ipadx=10)

        self.transient(self.parent)
        self.grab_set()
        self.parent.wait_window(self)

    def save_and_apply(self):
        new_config = self.config.copy()
        for key, entry in self.entries.items():
            if isinstance(entry, tk.Text):
                new_config[key] = entry.get('1.0', 'end-1c').strip()
            elif isinstance(entry, ttk.Combobox):
                new_config[key] = entry.get().strip()
            else:
                new_config[key] = entry.get().strip()

        if self.apply_callback:
            self.apply_callback(new_config)

        self.destroy()


# --- 3. 字幕窗口实现 (已升级：支持进度条) ---
class SubtitleWindow:
    def __init__(self, initial_config, apply_settings_callback):
        self.config = initial_config
        self.apply_settings_callback = apply_settings_callback

        self.root = tk.Tk()
        self.root.title("实时翻译字幕")
        self.root.geometry(
            f'{self.config["window_width"]}x150+{(self.root.winfo_screenwidth() - self.config["window_width"]) // 2}+100')
        self.root.configure(bg=self.config['background_color'])
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", self.config['background_color'])

        font_settings = (self.config['font_family'], self.config['font_size'], self.config['font_weight'])

        self.label_original = tk.Label(self.root, text="...", bg=self.config['background_color'],
                                       fg=self.config['original_text_color'], font=font_settings,
                                       wraplength=self.config["window_width"] - 40, justify='center')
        self.label_original.pack(pady=(10, 5), fill='x')

        self.label_translated = tk.Label(self.root, text="正在初始化...", bg=self.config['background_color'],
                                         fg=self.config['translated_text_color'], font=font_settings,
                                         wraplength=self.config["window_width"] - 40, justify='center')
        self.label_translated.pack(pady=(5, 10), fill='x')

        # --- 新增：进度条组件 (默认隐藏) ---
        self.progress_frame = tk.Frame(self.root, bg=self.config['background_color'])
        # 配置进度条样式
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", troughcolor=self.config['background_color'], background='#007acc', thickness=5)

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal",
                                            length=self.config["window_width"] - 100, mode="determinate",
                                            style="TProgressbar")
        self.progress_bar.pack(pady=(0, 10))

        self.settings_button = tk.Label(self.root, text="⚙️", bg=self.config['background_color'], fg="white",
                                        font=('Arial', 16), cursor="hand2")
        self.settings_button.place(relx=1.0, rely=0.0, x=-5, y=5, anchor='ne')
        self.settings_button.bind("<Button-1>", self.open_settings)

        self.root.bind("<ButtonPress-1>", self.start_move)
        self.root.bind("<ButtonRelease-1>", self.stop_move)
        self.root.bind("<B1-Motion>", self.do_move)

    def open_settings(self, event):
        SettingsWindow(self.root, self.config, self.apply_settings_callback)

    def start_move(self, event):
        self.x, self.y = event.x, event.y

    def stop_move(self, event):
        self.x, self.y = None, None

    def do_move(self, event):
        deltax, deltay = event.x - self.x, event.y - self.y
        x, y = self.root.winfo_x() + deltax, self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    # --- 进度条控制方法 ---
    def show_progress(self, message, mode='determinate'):
        """显示进度条并设置模式"""

        def _show():
            self.label_translated.config(text=message)
            self.progress_bar.config(mode=mode)
            if mode == 'indeterminate':
                self.progress_bar.start(10)
            else:
                self.progress_bar.stop()
                self.progress_bar['value'] = 0
            self.progress_frame.pack(pady=5)
            self._adjust_height()

        self.root.after(0, _show)

    def update_progress(self, value, message=None):
        """更新进度条的值和信息"""

        def _update():
            if message:
                self.label_translated.config(text=message)
            self.progress_bar['value'] = value

        self.root.after(0, _update)

    def hide_progress(self):
        """隐藏进度条"""

        def _hide():
            self.progress_bar.stop()
            self.progress_frame.pack_forget()
            self._adjust_height()

        self.root.after(0, _hide)

    def update_text(self, original, translated):
        def _update():
            self.label_original.config(text=original)
            self.label_translated.config(text=translated)
            self._adjust_height()

        self.root.after(0, _update)

    def _adjust_height(self):
        """动态调整窗口高度"""
        self.root.update_idletasks()
        h_orig = self.label_original.winfo_reqheight()
        h_trans = self.label_translated.winfo_reqheight()
        h_progress = self.progress_frame.winfo_reqheight() if self.progress_frame.winfo_ismapped() else 0
        new_height = h_orig + h_trans + h_progress + 40
        self.root.geometry(f'{self.config["window_width"]}x{new_height}+{self.root.winfo_x()}+{self.root.winfo_y()}')

    def run(self):
        self.root.mainloop()


# --- 4. 核心翻译逻辑 (已升级：支持下载和进度反馈) ---
class Translator:
    def __init__(self, initial_config):
        self.config = initial_config
        self.window = SubtitleWindow(self.config, self.apply_settings)
        self.stt_model = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.previous_text = ""
        self.models_ready = False

        # 在后台线程中初始化所有模型，避免阻塞UI启动
        threading.Thread(target=self.initialize_all_models, daemon=True).start()

        self.setup_audio()

    def setup_audio(self):
        try:
            devices = sd.query_devices()
            self.device_id = None
            for i, device in enumerate(devices):
                if self.config["audio_device_name_substring"] in device['name']:
                    self.device_id = i
                    print(f"找到音频设备: {device['name']} (ID: {self.device_id})")
                    break
            if self.device_id is None:
                print(f"警告: 找不到包含 '{self.config['audio_device_name_substring']}' 的音频设备。将使用默认输入设备。")
        except Exception as e:
            print(f"音频设备错误: {e}")

    def initialize_all_models(self):
        """按顺序初始化 Whisper 和 Ollama 模型"""
        self.load_whisper_model()
        self.ensure_ollama_model(self.config['ollama_model'])
        self.models_ready = True
        self.window.update_text("", "所有模型准备就绪，等待音频输入...")

    def load_whisper_model(self):
        print(f"正在加载 Whisper 模型: {self.config['stt_model_size']}...")
        # Whisper 下载很难获取精确进度，使用滚动进度条
        self.window.show_progress(f"正在加载/下载 Whisper 模型: {self.config['stt_model_size']}...",
                                  mode='indeterminate')
        try:
            self.stt_model = WhisperModel(self.config['stt_model_size'], device=self.config['device'],
                                          compute_type=self.config['compute_type'])
            print("Whisper 模型加载完成。")
        except Exception as e:
            print(f"加载 Whisper 模型失败: {e}")
            self.window.update_text("错误", f"加载 Whisper 模型失败: {e}")
        finally:
            self.window.hide_progress()

    def ensure_ollama_model(self, model_name):
        """检查 Ollama 模型是否存在，不存在则下载并显示精确进度"""
        print(f"正在检查 Ollama 模型: {model_name}...")
        self.window.show_progress(f"正在准备 Ollama 模型: {model_name}...", mode='determinate')
        try:
            # 使用 ollama.pull(stream=True) 来获取下载进度
            for progress in ollama.pull(model_name, stream=True):
                status = progress.get('status', '')
                if 'completed' in progress and 'total' in progress:
                    completed = progress['completed']
                    total = progress['total']
                    percent = (completed / total) * 100
                    # 显示当前层的下载进度
                    self.window.update_progress(percent, f"下载中 {model_name}: {status} - {percent:.1f}%")
                else:
                    # 显示状态 (如 "pulling manifest", "verifying sha256 digest")
                    self.window.update_progress(0, f"Ollama: {status}")

            print(f"Ollama 模型 {model_name} 准备就绪。")

        except Exception as e:
            print(f"准备 Ollama 模型失败: {e}")
            self.window.update_text("错误", f"准备 Ollama 模型失败: {e}")
        finally:
            self.window.hide_progress()

    def apply_settings(self, new_config):
        print("正在应用新设置...")
        stt_model_changed = new_config['stt_model_size'] != self.config['stt_model_size']
        ollama_model_changed = new_config['ollama_model'] != self.config['ollama_model']

        self.config = new_config
        self.window.config = new_config

        # 如果模型发生变化，在后台线程中重新加载
        if stt_model_changed or ollama_model_changed:
            def reload():
                self.models_ready = False
                if stt_model_changed:
                    self.load_whisper_model()
                if ollama_model_changed:
                    self.ensure_ollama_model(self.config['ollama_model'])
                self.models_ready = True
                self.window.update_text("", "设置已更新，模型准备就绪")

            threading.Thread(target=reload, daemon=True).start()
        else:
            messagebox.showinfo("设置", "设置已更新！")

    def audio_callback(self, indata, frames, time, status):
        if status: print(f"音频回调状态: {status}")
        self.audio_buffer = np.append(self.audio_buffer, indata.flatten())

    def process_audio(self):
        while True:
            sd.sleep(int(self.config["interval_seconds"] * 1000))
            if not self.models_ready: continue  # 等待模型加载完成

            if len(self.audio_buffer) < self.config["sample_rate"] * 0.5:
                continue

            processing_buffer = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)

            if self.stt_model is None: continue

            segments, _ = self.stt_model.transcribe(
                processing_buffer, beam_size=5, language=self.config["source_language_code"],
                vad_filter=True, initial_prompt=self.previous_text
            )
            original_text = " ".join([seg.text for seg in segments]).strip()

            if original_text and original_text.lower() != self.previous_text.lower():
                print(f"\n原文: {original_text}")
                self.window.update_text(original_text, "正在翻译...")
                self.previous_text = original_text

                try:
                    prompt = self.config["ollama_prompt_template"].format(
                        source_language=self.config['source_language_name'],
                        target_language=self.config['target_language'],
                        text=original_text
                    )
                    stream = ollama.generate(model=self.config['ollama_model'], prompt=prompt, stream=True)
                    translated_text = ""
                    for chunk in stream:
                        if 'response' in chunk:
                            translated_text += chunk['response']
                            self.window.update_text(original_text, translated_text)
                    print(f"译文: {translated_text}")
                except Exception as e:
                    error_message = f"Ollama 调用失败: {e}"
                    print(error_message)
                    self.window.update_text(original_text, error_message)
            elif not original_text and self.previous_text:
                self.previous_text = ""
                self.window.update_text("", "")

    def start(self):
        processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        processing_thread.start()
        # 使用配置的采样率
        with sd.InputStream(device=self.device_id, channels=1, samplerate=self.config["sample_rate"],
                            callback=self.audio_callback):
            print("--- 开始监听音频 ---")
            self.window.run()


if __name__ == "__main__":
    translator = Translator(CONFIG)
    translator.start()