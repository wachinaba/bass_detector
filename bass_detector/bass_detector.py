import sounddevice as sd
import numpy as np
import time
from queue import Queue, Empty  # コールバックからのデータ受け渡し用
from typing import Any, List, Dict, Optional  # 型アノテーション用
import matplotlib.pyplot as plt
from collections import deque
import datetime
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys

# オーディオ設定
SAMPLE_RATE = 44100  # サンプリングレート (Hz)
CHUNK_SIZE = 4096  # FFTの窓サイズ（周波数解像度用）
HOP_SIZE = 128  # フレームシフト（時間解像度用）
OVERLAP = 1 - HOP_SIZE / CHUNK_SIZE  # オーバーラップ率
FFT_BINS = CHUNK_SIZE // 2  # FFTの結果の有効なビンの数（ナイキスト周波数まで）

# バッファ管理用の設定
BUFFER_SIZE = CHUNK_SIZE * 2  # オーバーラップ処理用のバッファサイズ

# バスドラム検出の設定
TARGET_FREQ = 53.8  # 検出対象の周波数 (Hz)
FREQ_TOLERANCE = 2.0  # 周波数の許容範囲 (±Hz)
POWER_THRESHOLD = 250.0  # パワー検出の閾値
PEAK_COOLDOWN = 0.05  # 連続検出を防ぐためのクールダウン時間 (秒)
PEAK_HISTORY_SIZE = 5  # ピーク履歴の保持数
PEAK_RISE_THRESHOLD = 0.3  # 立ち上がり判定の閾値（直前の平均に対する比率）

# データ記録用の設定
MAX_RECORD_TIME = 300  # 最大記録時間（秒）
RECORD_INTERVAL = 0.05  # 記録間隔（秒）

# プロット用の設定
PLOT_HISTORY_SECONDS = 3.0  # プロットの履歴時間（秒）
PLOT_UPDATE_INTERVAL = 50  # プロット更新間隔 (ms)
PLOT_BIN_COUNT = 6  # プロットするビンの数

# グローバル変数
audio_buffer: Queue[Any] = Queue()
last_peak_detection_time = 0.0
audio_data_buffer = np.zeros(BUFFER_SIZE)  # オーバーラップ処理用のバッファ
buffer_position = 0  # バッファ内の現在位置
peak_history = deque(maxlen=PEAK_HISTORY_SIZE)  # ピーク履歴


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
    53.8Hz付近のパワーが閾値を超えたら検出する関数

    Args:
        fft_magnitudes: FFTの振幅スペクトル
        frequencies: 周波数軸の配列

    Returns:
        bool: バスドラムが検出された場合はTrue
    """
    global last_peak_detection_time

    # 現在時刻を取得
    current_time = time.time()

    # クールダウン期間中は検出しない
    if current_time - last_peak_detection_time < PEAK_COOLDOWN:
        return False

    # 目標周波数付近のインデックスを取得
    target_indices = np.where(
        (frequencies >= TARGET_FREQ - FREQ_TOLERANCE)
        & (frequencies <= TARGET_FREQ + FREQ_TOLERANCE)
    )[0]

    if len(target_indices) == 0:
        return False

    # 目標周波数帯域の最大パワーを取得
    target_max_power = np.max(fft_magnitudes[target_indices])

    # パワーが閾値を超えているかチェック
    if target_max_power > POWER_THRESHOLD:
        last_peak_detection_time = current_time
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
            (frequencies >= TARGET_FREQ - FREQ_TOLERANCE)
            & (frequencies <= TARGET_FREQ + FREQ_TOLERANCE)
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


class RealTimePlotter:
    def __init__(self):
        # アプリケーションの作成
        self.app = pg.mkQApp("Bass Detector")

        # メインウィンドウの設定
        self.win = pg.GraphicsLayoutWidget(show=True, title="バスドラム検出モニター")
        self.win.resize(1000, 600)
        self.win.setWindowTitle("バスドラム検出モニター")

        # プロットの設定
        self.plot = self.win.addPlot(row=0, col=0)
        self.plot.setLabel("left", "パワー")
        self.plot.setLabel("bottom", "時間", "秒")
        self.plot.showGrid(x=True, y=True)

        # データ保持用のリスト（各ビンごと）
        self.times: List[float] = []
        self.powers: List[List[float]] = [[] for _ in range(PLOT_BIN_COUNT)]

        # プロットアイテムの作成（各ビンごと）
        self.curves = []
        colors = ["b", "g", "c", "m", "y", "w"]  # 各ビンの色
        for i in range(PLOT_BIN_COUNT):
            curve = self.plot.plot(pen=pg.mkPen(colors[i], width=2), name=f"ビン{i+1}")
            self.curves.append(curve)

        # 閾値線
        self.threshold_line = pg.InfiniteLine(
            pos=POWER_THRESHOLD,
            angle=0,
            pen=pg.mkPen("r", width=2, style=QtCore.Qt.DashLine),
            name="閾値",
        )
        self.plot.addItem(self.threshold_line)

        # 検出マーカー用のプロットアイテム
        self.detection_plot = self.plot.plot(
            pen=None, symbol="o", symbolSize=10, symbolBrush="r", name="検出"
        )

        # 凡例の追加
        self.plot.addLegend()

        # タイマーの設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(PLOT_UPDATE_INTERVAL)

        # 検出時のマーカー用の配列
        self.detection_times: List[float] = []
        self.detection_powers: List[float] = []

        # プロットの表示範囲を設定
        self.plot.setYRange(0, POWER_THRESHOLD * 1.5)

        # 現在のFFTデータ
        self.current_fft_data: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.start_time = time.time()

        # デバッグ用カウンター
        self.update_count = 0
        self.last_print_time = time.time()

    def update(self):
        """プロットの更新"""
        if self.current_fft_data is None:
            return

        frequencies, fft_magnitudes = self.current_fft_data
        self.update_count += 1

        # 1秒ごとにデバッグ情報を表示
        current_time = time.time()
        if current_time - self.last_print_time >= 1.0:
            print(f"\nプロット更新回数: {self.update_count}/秒")
            print(f"FFTデータ形状: {frequencies.shape}, {fft_magnitudes.shape}")
            print(f"現在の時間: {current_time - self.start_time:.1f}秒")
            print(f"データポイント数: {len(self.times)}")
            self.update_count = 0
            self.last_print_time = current_time

        # 現在の時間を取得
        plot_time = current_time - self.start_time

        # 低い周波数帯域のビンを取得（53.8Hz付近）
        target_indices = np.where(
            (frequencies >= TARGET_FREQ - FREQ_TOLERANCE)
            & (frequencies <= TARGET_FREQ + FREQ_TOLERANCE)
        )[0]

        if len(target_indices) >= PLOT_BIN_COUNT:
            # 時間を追加
            self.times.append(plot_time)

            # 各ビンのパワーを追加
            for i in range(PLOT_BIN_COUNT):
                idx = target_indices[i]
                power = float(fft_magnitudes[idx])  # 明示的にfloatに変換
                self.powers[i].append(power)

                # デバッグ情報（最初の更新時のみ）
                if len(self.times) == 1:
                    print(
                        f"ビン{i+1} - 周波数: {frequencies[idx]:.1f}Hz, パワー: {power:.1f}"
                    )

            # 3秒より古いデータを削除
            cutoff_time = plot_time - PLOT_HISTORY_SECONDS
            while self.times and self.times[0] < cutoff_time:
                self.times.pop(0)
                for powers in self.powers:
                    powers.pop(0)

            # プロットの表示範囲を更新
            if self.times:
                self.plot.setXRange(self.times[0], self.times[-1])

            # 各ビンのプロットを更新
            for i, curve in enumerate(self.curves):
                if len(self.times) == len(self.powers[i]):
                    curve.setData(self.times, self.powers[i])

            # 検出判定（最初のビンのパワーで判定）
            if self.powers[0] and self.powers[0][-1] > POWER_THRESHOLD:
                self.detection_times.append(plot_time)
                self.detection_powers.append(self.powers[0][-1])

            # 3秒より古い検出マーカーを削除
            while self.detection_times and self.detection_times[0] < cutoff_time:
                self.detection_times.pop(0)
                self.detection_powers.pop(0)

            # マーカーをプロット
            if self.detection_times:
                self.detection_plot.setData(self.detection_times, self.detection_powers)
            else:
                self.detection_plot.setData([], [])

    def start(self, start_time: float):
        """プロットの開始"""
        self.start_time = start_time
        self.current_fft_data = None
        self.update_count = 0
        self.last_print_time = start_time
        # データをクリア
        self.times.clear()
        for powers in self.powers:
            powers.clear()
        self.detection_times.clear()
        self.detection_powers.clear()
        # プロットをクリア
        for curve in self.curves:
            curve.setData([], [])
        self.detection_plot.setData([], [])
        print("\nプロッターを初期化しました")

    def update_fft_data(self, frequencies: np.ndarray, fft_magnitudes: np.ndarray):
        """FFTデータを更新"""
        # データの型を確認して変換
        frequencies = np.asarray(frequencies, dtype=np.float64)
        fft_magnitudes = np.asarray(fft_magnitudes, dtype=np.float64)

        # データの有効性をチェック
        if not np.all(np.isfinite(frequencies)) or not np.all(
            np.isfinite(fft_magnitudes)
        ):
            print("警告: 無効なFFTデータが検出されました")
            return

        self.current_fft_data = (frequencies, fft_magnitudes)


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
        # プロッターの初期化
        plotter = RealTimePlotter()
        plotter.start(time.time())

        stream = sd.InputStream(
            device=selected_device_id,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            callback=audio_callback,
        )

        with stream:
            print("\nストリームを開始しました。音声の解析を開始します...")
            print("バスドラム検出モード: 有効")
            print(f"検出周波数: {TARGET_FREQ}Hz (±{FREQ_TOLERANCE}Hz)")
            print(f"パワー閾値: {POWER_THRESHOLD}")
            print(f"記録時間: 最大{MAX_RECORD_TIME}秒, 記録間隔: {RECORD_INTERVAL}秒")
            print(f"プロットするビン数: {PLOT_BIN_COUNT}")
            print(f"プロット更新間隔: {PLOT_UPDATE_INTERVAL}ms")

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
                        # FFTデータを更新
                        plotter.update_fft_data(frequencies, fft_magnitudes)

                        # データを記録
                        recorder.record(frequencies, fft_magnitudes)

                        # バスドラム検出
                        if detect_bass_drum(fft_magnitudes, frequencies):
                            print(f"\n🎵 バスドラム検出！ 🎵 : {bass_count}")
                            bass_count += 1

                    # Qtイベントループの処理
                    QtWidgets.QApplication.processEvents()

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
