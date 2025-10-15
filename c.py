import tkinter as tk
from tkinter import messagebox
import threading
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import ollama

# --- 1. 默认配置区域 ---
# 使用一个字典来集中管理所有可配置的参数
CONFIG = {
    # Whisper 模型配置
    "stt_model_size": "distil-large-v3.5",
    "device": "cuda",
    "compute_type": "int8",
    # 语言配置
    "source_language_code": "en",
    "source_language_name": "英语",
    "target_language": "中文",
    # Ollama 模型配置
    "ollama_model": 'gemma2:9b-instruct-q4_K_M',
    # Ollama 提示词模板
    # 使用 {source_language}, {target_language}, 和 {text} 作为占位符
    "ollama_prompt_template": "Translate the following {source_language} text to {target_language}. "
                              "Only output the translated content, without any explanation or original text. "
                              "Text: '{text}'",
    # 音频捕获配置
    "audio_device_name_substring": "CABLE Output",  # 使用子字符串匹配设备
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


# --- 2. 设置窗口实现 ---
class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, current_config, apply_callback):
        super().__init__(parent)
        self.title("设置")
        self.geometry("600x450")
        self.configure(bg='#2b2b2b')
        self.parent = parent
        self.config = current_config.copy()  # 创建一个配置的副本
        self.apply_callback = apply_callback
        self.entries = {}

        # 设置项
        settings = {
            "stt_model_size": "Whisper 模型:",
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
            else:
                entry = tk.Entry(row, width=40, fg='white', bg='#3c3f41', insertbackground='white')
                entry.insert(0, self.config.get(key, ''))

            entry.pack(side='left', fill='x', expand=True)
            self.entries[key] = entry

        # 保存按钮
        save_button = tk.Button(self, text="保存并应用", command=self.save_and_apply, bg='#007acc', fg='white',
                                relief='flat')
        save_button.pack(pady=20)

        # 居中显示
        self.transient(self.parent)
        self.grab_set()
        self.parent.wait_window(self)

    def save_and_apply(self):
        new_config = self.config.copy()
        for key, entry in self.entries.items():
            if isinstance(entry, tk.Text):
                new_config[key] = entry.get('1.0', 'end-1c').strip()
            else:
                new_config[key] = entry.get().strip()

        if self.apply_callback:
            self.apply_callback(new_config)

        self.destroy()


# --- 3. 字幕窗口实现 (已升级) ---
class SubtitleWindow:
    def __init__(self, initial_config, apply_settings_callback):
        self.config = initial_config
        self.apply_settings_callback = apply_settings_callback

        self.root = tk.Tk()
        self.root.title("实时翻译字幕")
        # 初始高度设为较小值，后续会动态调整
        initial_height = 150
        self.root.geometry(
            f'{self.config["window_width"]}x{initial_height}+{(self.root.winfo_screenwidth() - self.config["window_width"]) // 2}+100')
        self.root.configure(bg=self.config['background_color'])
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", self.config['background_color'])

        font_settings = (self.config['font_family'], self.config['font_size'], self.config['font_weight'])

        self.label_original = tk.Label(self.root, text="...", bg=self.config['background_color'],
                                       fg=self.config['original_text_color'], font=font_settings,
                                       wraplength=self.config["window_width"] - 40, justify='center')
        self.label_original.pack(pady=10, fill='x')

        self.label_translated = tk.Label(self.root, text="等待音频输入...", bg=self.config['background_color'],
                                         fg=self.config['translated_text_color'], font=font_settings,
                                         wraplength=self.config["window_width"] - 40, justify='center')
        self.label_translated.pack(pady=10, fill='x')

        # 添加设置按钮
        self.settings_button = tk.Label(self.root, text="⚙️", bg=self.config['background_color'], fg="white",
                                        font=('Arial', 16))
        self.settings_button.place(relx=1.0, rely=0.0, x=-5, y=5, anchor='ne')
        self.settings_button.bind("<Button-1>", self.open_settings)

        # 绑定窗口拖动事件
        self.root.bind("<ButtonPress-1>", self.start_move)
        self.root.bind("<ButtonRelease-1>", self.stop_move)
        self.root.bind("<B1-Motion>", self.do_move)

    def open_settings(self, event):
        SettingsWindow(self.root, self.config, self.apply_settings_callback)

    def start_move(self, event): self.x, self.y = event.x, event.y

    def stop_move(self, event): self.x, self.y = None, None

    def do_move(self, event):
        deltax, deltay = event.x - self.x, event.y - self.y
        x, y = self.root.winfo_x() + deltax, self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def update_text(self, original, translated):
        # 使用 after 确保在主线程中更新UI
        def _update():
            self.label_original.config(text=original)
            self.label_translated.config(text=translated)

            # --- 动态高度调整 ---
            self.root.update_idletasks()  # 强制UI更新以获取正确的尺寸
            h_orig = self.label_original.winfo_reqheight()
            h_trans = self.label_translated.winfo_reqheight()
            new_height = h_orig + h_trans + 40  # 增加一些垂直内边距

            current_x = self.root.winfo_x()
            current_y = self.root.winfo_y()
            # 更新窗口几何属性，只改变高度
            self.root.geometry(f'{self.config["window_width"]}x{new_height}+{current_x}+{current_y}')

        self.root.after(0, _update)

    def run(self):
        self.root.mainloop()


# --- 4. 核心翻译逻辑 (已升级) ---
class Translator:
    def __init__(self, initial_config):
        self.config = initial_config
        self.window = SubtitleWindow(self.config, self.apply_settings)
        self.stt_model = None
        self.load_whisper_model()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.previous_text = ""

        try:
            devices = sd.query_devices()
            self.device_id = None
            for i, device in enumerate(devices):
                if self.config["audio_device_name_substring"] in device['name']:
                    self.device_id = i
                    print(f"找到音频设备: {device['name']} (ID: {self.device_id})")
                    break
            if self.device_id is None:
                raise ValueError(f"找不到包含 '{self.config['audio_device_name_substring']}' 的音频设备。")
        except Exception as e:
            print(f"音频设备错误: {e}")
            print("可用设备列表:")
            print(sd.query_devices())
            exit()

    def load_whisper_model(self):
        print(f"正在加载 Whisper 模型: {self.config['stt_model_size']}...")
        try:
            self.stt_model = WhisperModel(self.config['stt_model_size'], device=self.config['device'],
                                          compute_type=self.config['compute_type'])
            print("Whisper 模型加载完成。")
            self.window.update_text("", "Whisper 模型已就绪")
        except Exception as e:
            print(f"加载 Whisper 模型失败: {e}")
            self.window.update_text("错误", f"加载 Whisper 模型失败: {e}")

    def apply_settings(self, new_config):
        print("正在应用新设置...")
        stt_model_changed = new_config['stt_model_size'] != self.config['stt_model_size']
        self.config = new_config
        self.window.config = new_config  # 确保窗口也拥有最新配置

        if stt_model_changed:
            self.window.update_text("请稍候", f"正在重新加载模型: {self.config['stt_model_size']}...")
            # 在新线程中加载模型以避免UI冻结
            threading.Thread(target=self.load_whisper_model, daemon=True).start()
        else:
            messagebox.showinfo("设置", "设置已更新！")

    def audio_callback(self, indata, frames, time, status):
        if status: print(f"音频回调状态: {status}")
        self.audio_buffer = np.append(self.audio_buffer, indata.flatten())

    def process_audio(self):
        while True:
            sd.sleep(int(self.config["interval_seconds"] * 1000))
            if len(self.audio_buffer) < self.config["sample_rate"] * 0.5:
                continue

            processing_buffer = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)

            segments, _ = self.stt_model.transcribe(
                processing_buffer, beam_size=5, language=self.config["source_language_code"],
                vad_filter=True, initial_prompt=self.previous_text
            )
            original_text = " ".join([seg.text for seg in segments]).strip()

            if original_text and original_text.lower() != self.previous_text.lower():
                print(f"\n原文 ({self.config['source_language_name']}): {original_text}")
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
                    print(f"译文 ({self.config['target_language']}): {translated_text}")
                except Exception as e:
                    error_message = f"Ollama 调用失败: {e}"
                    print(error_message)
                    self.window.update_text(original_text, error_message)
            elif not original_text and self.previous_text:
                # 如果没有识别到新文本，则清空字幕
                self.previous_text = ""
                self.window.update_text("", "")

    def start(self):
        processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        processing_thread.start()
        with sd.InputStream(device=self.device_id, channels=1, samplerate=self.config["sample_rate"],
                            callback=self.audio_callback):
            print("--- 开始监听音频 ---")
            self.window.run()


if __name__ == "__main__":
    translator = Translator(CONFIG)
    translator.start()