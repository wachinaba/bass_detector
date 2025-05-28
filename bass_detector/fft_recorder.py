import sounddevice as sd
import numpy as np
import time
from queue import Queue, Empty  # コールバックからのデータ受け渡し用
from typing import Any, List, Dict  # 型アノテーション用
import matplotlib.pyplot as plt
from collections import deque
import datetime

# オーディオ設定
SAMPLE_RATE = 44100  # サンプリングレート (Hz)
CHUNK_SIZE = 4096  # FFTの窓サイズ（周波数解像度用）
HOP_SIZE = 128  # フレームシフト（時間解像度用）
OVERLAP = 1 - HOP_SIZE / CHUNK_SIZE  # オーバーラップ率
FFT_BINS = CHUNK_SIZE // 2  # FFTの結果の有効なビンの数（ナイキスト周波数まで）

# バッファ管理用の設定
BUFFER_SIZE = CHUNK_SIZE * 2  # オーバーラップ処理用のバッファサイズ

# バスドラム検出の設定
BASS_MIN_FREQ = 20  # バスドラムの最小周波数 (Hz)
BASS_MAX_FREQ = 100  # バスドラムの最大周波数 (Hz)
BASS_THRESHOLD = 0.5  # バスドラム検出の閾値（最大振幅に対する比率）
BASS_COOLDOWN = 0.05  # 連続検出を防ぐためのクールダウン時間 (秒)

# データ記録用の設定
MAX_RECORD_TIME = 300  # 最大記録時間（秒）
RECORD_INTERVAL = 0.01  # 記録間隔（秒）

# グローバル変数
audio_buffer: Queue[Any] = Queue()
last_bass_detection_time = 0.0
audio_data_buffer = np.zeros(BUFFER_SIZE)  # オーバーラップ処理用のバッファ
buffer_position = 0  # バッファ内の現在位置


def print_fft_info(fft_magnitudes, frequencies):
    """
    FFTの結果からピーク周波数とパワー（振幅の2乗）を表示する関数 (内容は前と同じ)
    """
    if len(fft_magnitudes) == 0:
        # print("FFT結果が空です。") # 連続処理中なのでコメントアウトしてもよい
        return

    peak_index = np.argmax(fft_magnitudes)
    peak_frequency = frequencies[peak_index]
    peak_magnitude = fft_magnitudes[peak_index]

    display_bins = min(len(fft_magnitudes), 40)  # 表示するビンの数を調整
    # ゼロ除算を避けるため、分母に微小値を追加
    max_magnitude_for_scale = np.max(fft_magnitudes[:display_bins])
    if max_magnitude_for_scale == 0:
        max_magnitude_for_scale = 1.0

    scaled_magnitudes = (
        fft_magnitudes[:display_bins] / max_magnitude_for_scale * 15
    ).astype(
        int
    )  # 15段階で正規化

    # 画面クリア（簡易的、ターミナルによる）
    # print("\033[H\033[J", end="") # Linux/macOSの場合
    # import os
    # os.system('cls' if os.name == 'nt' else 'clear') # Windows/Linux/macOS
    """
    print("-" * 40)
    print(f"ピーク周波数: {peak_frequency:7.1f} Hz | パワー: {peak_magnitude:8.1f}")
    # print("簡易スペクトル (周波数とパワーの目安):")
    # for i in range(display_bins):
    #     freq_label = f"{frequencies[i]:5.0f}Hz"
    #     bar = "#" * scaled_magnitudes[i]
    #     print(f"{freq_label} | {bar:<15} ({fft_magnitudes[i]:.1f})")
    print("-" * 40)
    """


def process_audio_chunk(chunk: np.ndarray) -> List[tuple[np.ndarray, np.ndarray]]:
    """
    オーディオチャンクをオーバーラップFFTで処理する関数

    Returns:
        List[tuple[np.ndarray, np.ndarray]]: (周波数, 振幅)のタプルのリスト
    """
    global audio_data_buffer, buffer_position

    # バッファにデータを追加
    audio_data_buffer[buffer_position : buffer_position + len(chunk)] = chunk
    buffer_position += len(chunk)

    results = []

    # バッファに十分なデータがある場合、FFTを実行
    while buffer_position >= CHUNK_SIZE:
        # 現在のフレームを取得
        frame = audio_data_buffer[:CHUNK_SIZE].copy()

        # ハン窓を適用
        window = np.hanning(CHUNK_SIZE)
        frame = frame * window

        # FFTを実行
        fft_result = np.fft.fft(frame)
        fft_magnitudes = np.abs(fft_result[:FFT_BINS])
        frequencies = np.fft.fftfreq(CHUNK_SIZE, d=1 / SAMPLE_RATE)[:FFT_BINS]

        results.append((frequencies, fft_magnitudes))

        # バッファをシフト
        audio_data_buffer = np.roll(audio_data_buffer, -HOP_SIZE)
        buffer_position -= HOP_SIZE

    return results


def audio_callback(indata, frames, time_info, status):
    """
    オーディオストリームからデータを受け取るコールバック関数
    """
    global audio_buffer
    if status:
        print(f"Status: {status}")
    audio_buffer.put(indata[:, 0].copy())


def select_input_device():
    """利用可能な入力デバイスをリストし、ユーザーに選択させる関数"""
    print("\n利用可能な入力デバイス:")
    devices = sd.query_devices()
    input_devices_info = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices_info.append(
                {
                    "id": i,
                    "name": dev["name"],
                    "hostapi_name": sd.query_hostapis(dev["hostapi"])["name"],
                }
            )
            print(
                f"  ID {i}: {dev['name']} (API: {sd.query_hostapis(dev['hostapi'])['name']}, 入力Ch: {dev['max_input_channels']})"
            )

    if not input_devices_info:
        print("利用可能な入力デバイスが見つかりませんでした。")
        return None

    while True:
        try:
            choice_str = input(
                "使用するデバイスのID番号を選択してください (ループバック録音デバイスまたはマイク): "
            )
            if choice_str.lower() == "exit":
                return None
            device_id = int(choice_str)
            # 選択されたIDが実際にリストにあるか確認
            if any(d["id"] == device_id for d in input_devices_info):
                selected_device_name = next(
                    d["name"] for d in devices if d["index"] == device_id
                )
                print(f"選択されたデバイス: ID {device_id} - {selected_device_name}")
                return device_id
            else:
                print("無効なIDです。リストから選択してください。")
        except ValueError:
            print("無効な入力です。数値でIDを入力するか、'exit'と入力してください。")


def detect_bass_drum(fft_magnitudes: np.ndarray, frequencies: np.ndarray) -> bool:
    """
    バスドラムの周波数帯域でピークを検出する関数

    Args:
        fft_magnitudes: FFTの振幅スペクトル
        frequencies: 周波数軸の配列

    Returns:
        bool: バスドラムが検出された場合はTrue
    """
    global last_bass_detection_time

    # 現在時刻を取得
    current_time = time.time()

    # クールダウン期間中は検出しない
    if current_time - last_bass_detection_time < BASS_COOLDOWN:
        return False

    # バスドラムの周波数帯域のインデックスを取得
    bass_indices = np.where(
        (frequencies >= BASS_MIN_FREQ) & (frequencies <= BASS_MAX_FREQ)
    )[0]

    if len(bass_indices) == 0:
        return False

    # バスドラム帯域の最大振幅を取得
    bass_max_magnitude = np.max(fft_magnitudes[bass_indices])

    # 全周波数帯域の最大振幅を取得
    total_max_magnitude = np.max(fft_magnitudes)

    # バスドラム帯域の振幅が閾値を超えているかチェック
    if bass_max_magnitude > total_max_magnitude * BASS_THRESHOLD:
        last_bass_detection_time = current_time
        return True

    return False


# データ記録用の構造
class FFTRecorder:
    def __init__(self, max_time: float, interval: float):
        self.max_time = max_time
        self.interval = interval
        self.timestamps: List[float] = []
        self.bass_powers: Dict[float, List[float]] = {}  # 周波数ごとのパワー記録
        self.start_time = time.time()
        self.last_record_time = 0.0

    def should_record(self) -> bool:
        current_time = time.time()
        if current_time - self.start_time > self.max_time:
            return False
        if current_time - self.last_record_time < self.interval:
            return False
        return True

    def record(self, frequencies: np.ndarray, fft_magnitudes: np.ndarray):
        if not self.should_record():
            return

        current_time = time.time()
        self.timestamps.append(current_time - self.start_time)

        # バスドラム帯域の周波数ごとのパワーを記録
        bass_indices = np.where(
            (frequencies >= BASS_MIN_FREQ) & (frequencies <= BASS_MAX_FREQ)
        )[0]
        for idx in bass_indices:
            freq = frequencies[idx]
            if freq not in self.bass_powers:
                self.bass_powers[freq] = []
            self.bass_powers[freq].append(fft_magnitudes[idx])

        self.last_record_time = current_time

    def plot(self):
        if not self.timestamps or not self.bass_powers:
            print("記録データがありません。")
            return

        plt.figure(figsize=(12, 6))

        # 各周波数のパワーをプロット
        for freq, powers in self.bass_powers.items():
            if len(powers) == len(
                self.timestamps
            ):  # データ長が一致する場合のみプロット
                plt.plot(self.timestamps, powers, label=f"{freq:.1f}Hz", alpha=0.7)

        plt.title("バスドラム帯域の周波数パワー推移")
        plt.xlabel("時間 (秒)")
        plt.ylabel("パワー")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # 現在時刻をファイル名に使用
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"bass_power_plot_{timestamp}.png", bbox_inches="tight", dpi=300)
        plt.close()


def main_sounddevice():
    print("Python-SoundDevice リアルタイムFFTアナライザー")
    print("----------------------------------------------")
    print("ヒント: ループバック録音（PCから出る音を拾う）を行うには、")
    print(
        "- Windows: 「ステレオミキサー」や「Wave Out Mix」などのデバイスを選択。有効化が必要な場合あり。"
    )
    print(
        "- macOS: BlackHoleやSoundflowerなどの仮想オーディオデバイスを設定し、それを選択。"
    )
    print("- Linux: PulseAudioのモニターソース（例: 'Monitor of ...'）を選択。")

    selected_device_id = select_input_device()
    if selected_device_id is None:
        print("デバイスが選択されませんでした。終了します。")
        return

    print(
        f"\n--- デバイスID {selected_device_id} からの音声をリアルタイムFFT解析します ---"
    )
    print(
        f"サンプリングレート: {SAMPLE_RATE} Hz, チャンクサイズ: {CHUNK_SIZE} サンプル"
    )
    print(f"周波数解像度: {SAMPLE_RATE/CHUNK_SIZE:.1f} Hz/ビン")
    print(f"時間解像度: {HOP_SIZE/SAMPLE_RATE*1000:.1f} ms/フレーム")
    print(f"オーバーラップ率: {OVERLAP*100:.1f}%")
    print("Ctrl+C で停止します。")

    try:
        stream = sd.InputStream(
            device=selected_device_id,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,  # コールバックの呼び出し頻度を上げる
            callback=audio_callback,
        )

        with stream:
            print("\nストリームを開始しました。音声の解析を開始します...")
            print("バスドラム検出モード: 有効")
            print(
                f"検出範囲: {BASS_MIN_FREQ}-{BASS_MAX_FREQ}Hz, 閾値: {BASS_THRESHOLD}"
            )
            print(f"記録時間: 最大{MAX_RECORD_TIME}秒, 記録間隔: {RECORD_INTERVAL}秒")

            bass_count = 0
            recorder = FFTRecorder(MAX_RECORD_TIME, RECORD_INTERVAL)

            while True:
                try:
                    mono_data = audio_buffer.get(block=True, timeout=0.1)

                    if mono_data is None or len(mono_data) < HOP_SIZE:
                        continue

                    # オーバーラップFFTで処理
                    fft_results = process_audio_chunk(mono_data)

                    for frequencies, fft_magnitudes in fft_results:
                        # データを記録
                        recorder.record(frequencies, fft_magnitudes)

                        # バスドラム検出
                        if detect_bass_drum(fft_magnitudes, frequencies):
                            print(f"\n🎵 バスドラム検出！ 🎵 : {bass_count}")
                            bass_count += 1

                        print_fft_info(fft_magnitudes, frequencies)

                except Empty:
                    continue
                except KeyboardInterrupt:
                    print("\nプログラムを停止します。データをプロットします...")
                    recorder.plot()
                    print("プロットを保存しました。")
                    return
                except Exception as e_loop:
                    print(f"ループ内でエラー: {e_loop}")
                    time.sleep(0.1)  # 少し待ってから継続試行

    except sd.PortAudioError as e:
        print(f"PortAudioエラー: {e}")
        print(
            "選択されたオーディオデバイスに問題があるか、設定が間違っている可能性があります。"
        )
        print(
            "サンプリングレートやチャンネル数がデバイスでサポートされているか確認してください。"
        )
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
    finally:
        print("クリーンアップ中...")
        # キューに残っているデータを処理 (必要であれば)
        while not audio_buffer.empty():
            try:
                audio_buffer.get_nowait()
            except Empty:
                break
        print("終了しました。")


if __name__ == "__main__":
    main_sounddevice()
