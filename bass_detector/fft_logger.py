import numpy as np
import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
import pyqtgraph as pg
import sys
from collections import deque
import threading
import queue
import time
from enum import Enum
from typing import Callable


class ActivationFunction(Enum):
    LINEAR = "Linear"
    SIGMOID = "Sigmoid"
    RELU = "ReLU"
    TANH = "Tanh"
    STEP = "Step"


class Peak:
    def __init__(self, bin_idx, intensity, timestamp):
        self.bin_idx = bin_idx
        self.intensity = intensity
        self.timestamp = timestamp


class FFTLogger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bass Detector - FFT Logger")

        # 音声入力の設定
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.overlap = 0.90
        self.hop_size = int(self.chunk_size * (1 - self.overlap))
        self.device_id = None  # デフォルトデバイス

        # バッファの設定（3秒分）
        self.buffer_size = int(3 * self.sample_rate)
        self.audio_buffer = deque(maxlen=self.buffer_size)

        # FFTの設定
        self.fft_size = 2048
        self.freq_bins = np.fft.rfftfreq(self.fft_size, 1 / self.sample_rate)
        self.low_freq_bins = 10  # 低周波域のビン数
        self.visible_bins = set(range(self.low_freq_bins))  # 表示するビンのセット

        # 特徴点検出の設定
        self.threshold = 100.0  # 閾値
        self.sensitivity = 0.5  # 感度（0-1）
        self.marker_history = deque(maxlen=50)  # マーカーの履歴

        # ピーク管理の設定
        self.peak_history = deque(maxlen=100)  # ピークの履歴
        self.peak_decay_rate = 0.5  # ピークの減衰率
        self.peak_intensity_scale = 0.5  # ピーク強度のスケール
        self.current_peak_sum = 0.0  # 現在のピーク合計値
        self.max_peak_sum = 1000.0  # ピーク合計の最大値（明るさの正規化用）

        # アクティベーション関数の設定
        self.activation_function = ActivationFunction.SIGMOID
        self.activation_params = {
            "sigmoid_steepness": 0.01,  # シグモイドの急峻さ
            "relu_threshold": 100.0,  # ReLUの閾値
            "step_threshold": 500.0,  # ステップ関数の閾値
            "tanh_scale": 0.005,  # Tanhのスケール
        }

        # 時間-周波数データのバッファ
        self.time_buffer_size = 100  # 表示する時間フレーム数
        self.time_data = deque(maxlen=self.time_buffer_size)
        self.freq_data = np.zeros((self.low_freq_bins, self.time_buffer_size))
        self.prev_freq_data = np.zeros(self.low_freq_bins)  # 前フレームのデータ

        # 色の設定
        self.colors = [
            (255, 0, 0),  # 赤
            (255, 128, 0),  # オレンジ
            (255, 255, 0),  # 黄
            (128, 255, 0),  # 黄緑
            (0, 255, 0),  # 緑
            (0, 255, 128),  # 水色
            (0, 255, 255),  # シアン
            (0, 128, 255),  # 青緑
            (0, 0, 255),  # 青
            (128, 0, 255),  # 紫
        ]

        # プロットの設定
        self.setup_ui()

        # 音声入力用のキュー
        self.audio_queue = queue.Queue()

        # スレッドの開始
        self.running = True
        self.audio_thread = None  # デバイス選択後に初期化

        # タイマーの設定
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(30)  # 30msごとに更新

        # 背景色更新用のタイマー
        self.bg_timer = QTimer()
        self.bg_timer.timeout.connect(self.update_background)
        self.bg_timer.start(16)  # 約60FPS

    def get_activation_function(self) -> Callable[[float], float]:
        """現在選択されているアクティベーション関数を返す"""
        if self.activation_function == ActivationFunction.LINEAR:
            return lambda x: min(1.0, max(0.0, x / self.max_peak_sum))
        elif self.activation_function == ActivationFunction.SIGMOID:
            k = self.activation_params["sigmoid_steepness"]
            return lambda x: 1 / (1 + np.exp(-k * (x - self.max_peak_sum / 2)))
        elif self.activation_function == ActivationFunction.RELU:
            threshold = self.activation_params["relu_threshold"]
            max_val = self.max_peak_sum - threshold
            if max_val <= 0:
                max_val = 1.0  # ゼロ除算を防ぐ
            return lambda x: min(1.0, max(0.0, (x - threshold) / max_val))
        elif self.activation_function == ActivationFunction.TANH:
            scale = self.activation_params["tanh_scale"]
            return lambda x: (np.tanh(scale * (x - self.max_peak_sum / 2)) + 1) / 2
        elif self.activation_function == ActivationFunction.STEP:
            threshold = self.activation_params["step_threshold"]
            return lambda x: 1.0 if x > threshold else 0.0
        else:
            return lambda x: min(1.0, max(0.0, x / self.max_peak_sum))

    def setup_ui(self):
        # メインウィジェットの設定
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # デバイス選択用のウィジェット
        device_layout = QHBoxLayout()
        device_label = QLabel("入力デバイス:")
        self.device_combo = QComboBox()
        self.update_device_list()
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        layout.addLayout(device_layout)

        # 特徴点検出の設定用ウィジェット
        feature_layout = QHBoxLayout()
        threshold_label = QLabel("閾値:")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1000)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)

        sensitivity_label = QLabel("感度:")
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0, 1)
        self.sensitivity_spin.setSingleStep(0.1)
        self.sensitivity_spin.setValue(self.sensitivity)
        self.sensitivity_spin.valueChanged.connect(self.on_sensitivity_changed)

        decay_label = QLabel("減衰率:")
        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setRange(0.1, 0.99)
        self.decay_spin.setSingleStep(0.05)
        self.decay_spin.setValue(self.peak_decay_rate)
        self.decay_spin.valueChanged.connect(self.on_decay_changed)

        feature_layout.addWidget(threshold_label)
        feature_layout.addWidget(self.threshold_spin)
        feature_layout.addWidget(sensitivity_label)
        feature_layout.addWidget(self.sensitivity_spin)
        feature_layout.addWidget(decay_label)
        feature_layout.addWidget(self.decay_spin)
        feature_layout.addStretch()
        layout.addLayout(feature_layout)

        # アクティベーション関数の設定用ウィジェット
        activation_group = QWidget()
        activation_layout = QVBoxLayout(activation_group)
        activation_layout.setContentsMargins(0, 0, 0, 0)

        # 関数選択用のウィジェット
        function_layout = QHBoxLayout()
        activation_label = QLabel("アクティベーション関数:")
        self.activation_combo = QComboBox()
        for func in ActivationFunction:
            self.activation_combo.addItem(func.value)
        self.activation_combo.setCurrentText(self.activation_function.value)
        self.activation_combo.currentTextChanged.connect(self.on_activation_changed)
        function_layout.addWidget(activation_label)
        function_layout.addWidget(self.activation_combo)
        function_layout.addStretch()
        activation_layout.addLayout(function_layout)

        # パラメータ設定用のウィジェット
        self.param_widgets = {}

        # シグモイド用のパラメータ
        sigmoid_widget = QWidget()
        sigmoid_layout = QHBoxLayout(sigmoid_widget)
        sigmoid_layout.setContentsMargins(0, 0, 0, 0)
        sigmoid_label = QLabel("シグモイド急峻さ:")
        sigmoid_spin = QDoubleSpinBox()
        sigmoid_spin.setRange(0.001, 0.1)
        sigmoid_spin.setSingleStep(0.001)
        sigmoid_spin.setValue(self.activation_params["sigmoid_steepness"])
        sigmoid_spin.valueChanged.connect(
            lambda v: self.on_param_changed("sigmoid_steepness", v)
        )
        sigmoid_layout.addWidget(sigmoid_label)
        sigmoid_layout.addWidget(sigmoid_spin)
        sigmoid_layout.addStretch()
        self.param_widgets["sigmoid_steepness"] = sigmoid_widget

        # ReLU用のパラメータ
        relu_widget = QWidget()
        relu_layout = QHBoxLayout(relu_widget)
        relu_layout.setContentsMargins(0, 0, 0, 0)
        relu_label = QLabel("ReLU閾値:")
        relu_spin = QDoubleSpinBox()
        relu_spin.setRange(0, 500)
        relu_spin.setValue(self.activation_params["relu_threshold"])
        relu_spin.valueChanged.connect(
            lambda v: self.on_param_changed("relu_threshold", v)
        )
        relu_layout.addWidget(relu_label)
        relu_layout.addWidget(relu_spin)
        relu_layout.addStretch()
        self.param_widgets["relu_threshold"] = relu_widget

        # ステップ関数用のパラメータ
        step_widget = QWidget()
        step_layout = QHBoxLayout(step_widget)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_label = QLabel("ステップ閾値:")
        step_spin = QDoubleSpinBox()
        step_spin.setRange(0, 1000)
        step_spin.setValue(self.activation_params["step_threshold"])
        step_spin.valueChanged.connect(
            lambda v: self.on_param_changed("step_threshold", v)
        )
        step_layout.addWidget(step_label)
        step_layout.addWidget(step_spin)
        step_layout.addStretch()
        self.param_widgets["step_threshold"] = step_widget

        # Tanh用のパラメータ
        tanh_widget = QWidget()
        tanh_layout = QHBoxLayout(tanh_widget)
        tanh_layout.setContentsMargins(0, 0, 0, 0)
        tanh_label = QLabel("Tanhスケール:")
        tanh_spin = QDoubleSpinBox()
        tanh_spin.setRange(0.001, 0.1)
        tanh_spin.setSingleStep(0.001)
        tanh_spin.setValue(self.activation_params["tanh_scale"])
        tanh_spin.valueChanged.connect(lambda v: self.on_param_changed("tanh_scale", v))
        tanh_layout.addWidget(tanh_label)
        tanh_layout.addWidget(tanh_spin)
        tanh_layout.addStretch()
        self.param_widgets["tanh_scale"] = tanh_widget

        # パラメータウィジェットをレイアウトに追加
        for widget in self.param_widgets.values():
            activation_layout.addWidget(widget)

        layout.addWidget(activation_group)

        # ビン選択用のチェックボックス
        bin_layout = QGridLayout()
        self.bin_checkboxes = []
        for i in range(self.low_freq_bins):
            checkbox = QCheckBox(f"{self.freq_bins[i]:.1f} Hz")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, idx=i: self.on_bin_visibility_changed(idx, state)
            )
            self.bin_checkboxes.append(checkbox)
            bin_layout.addWidget(checkbox, i // 5, i % 5)
        layout.addLayout(bin_layout)

        # プロットウィジェットの設定
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Power")
        self.plot_widget.setLabel("bottom", "Time (frames)")
        self.plot_widget.setYRange(0, 300)
        self.plot_widget.setXRange(0, self.time_buffer_size)

        # プロットアイテムの設定
        self.plot_items = []
        self.marker_items = []
        for i in range(self.low_freq_bins):
            pen = pg.mkPen(color=self.colors[i], width=2)
            plot_item = self.plot_widget.plot(
                pen=pen, name=f"{self.freq_bins[i]:.1f} Hz"
            )
            self.plot_items.append(plot_item)

            # マーカー用のプロットアイテム
            marker_item = pg.ScatterPlotItem(
                size=10,
                symbol="o",
                pen=pg.mkPen(color=self.colors[i], width=2),
                brush=pg.mkBrush(color=self.colors[i]),
            )
            self.plot_widget.addItem(marker_item)
            self.marker_items.append(marker_item)

        # 凡例の追加
        self.plot_widget.addLegend()

        layout.addWidget(self.plot_widget)

        # 背景色の初期設定
        self.set_background_color(0)

        # パラメータウィジェットの表示/非表示を更新
        self.update_param_visibility()

    def on_threshold_changed(self, value):
        """閾値が変更されたときの処理"""
        self.threshold = value

    def on_sensitivity_changed(self, value):
        """感度が変更されたときの処理"""
        self.sensitivity = value

    def on_decay_changed(self, value):
        """減衰率が変更されたときの処理"""
        self.peak_decay_rate = value

    def on_activation_changed(self, value):
        """アクティベーション関数が変更されたときの処理"""
        self.activation_function = ActivationFunction(value)
        self.update_param_visibility()

    def on_param_changed(self, param_name, value):
        """パラメータが変更されたときの処理"""
        self.activation_params[param_name] = value

    def update_param_visibility(self):
        """パラメータウィジェットの表示/非表示を更新"""
        # 全てのパラメータウィジェットを非表示
        for widget in self.param_widgets.values():
            widget.setVisible(False)

        # 現在のアクティベーション関数に応じたパラメータを表示
        if self.activation_function == ActivationFunction.SIGMOID:
            self.param_widgets["sigmoid_steepness"].setVisible(True)
        elif self.activation_function == ActivationFunction.RELU:
            self.param_widgets["relu_threshold"].setVisible(True)
        elif self.activation_function == ActivationFunction.STEP:
            self.param_widgets["step_threshold"].setVisible(True)
        elif self.activation_function == ActivationFunction.TANH:
            self.param_widgets["tanh_scale"].setVisible(True)

    def set_background_color(self, intensity):
        """背景色を設定"""
        try:
            # アクティベーション関数を適用
            activation_func = self.get_activation_function()
            normalized_intensity = activation_func(intensity)

            # 強度を0-255の範囲に変換（確実に0-1の範囲に収める）
            normalized_intensity = min(1.0, max(0.0, normalized_intensity))
            brightness = int(255 * normalized_intensity)

            color = QColor(brightness, brightness, brightness)
            self.setStyleSheet(f"QMainWindow {{ background-color: {color.name()}; }}")
        except Exception as e:
            print(f"背景色の更新でエラーが発生しました: {e}")
            # エラー時は黒色に設定
            self.setStyleSheet("QMainWindow { background-color: black; }")

    def update_background(self):
        """背景色を更新"""
        current_time = time.time()
        peak_sum = 0.0

        # 各ピークの現在の強度を計算
        for peak in self.peak_history:
            time_diff = current_time - peak.timestamp
            decayed_intensity = peak.intensity * (self.peak_decay_rate**time_diff)
            if peak.bin_idx in self.visible_bins:
                peak_sum += decayed_intensity

        # ピーク合計を更新
        self.current_peak_sum = peak_sum

        # 背景色を更新
        self.set_background_color(peak_sum)

    def detect_features(self, current_data):
        """特徴点を検出"""
        features = []
        current_time = time.time()

        for i in range(self.low_freq_bins):
            if i not in self.visible_bins:
                continue

            # 急激な増加を検出
            diff = current_data[i] - self.prev_freq_data[i]
            if (
                current_data[i] > self.threshold
                and diff > self.threshold * self.sensitivity
            ):
                # ピークを記録
                peak_intensity = current_data[i] * self.peak_intensity_scale
                self.peak_history.append(Peak(i, peak_intensity, current_time))
                features.append((i, current_data[i]))

        self.prev_freq_data = current_data.copy()
        return features

    def update_markers(self, features):
        """マーカーを更新"""
        # 古いマーカーをクリア
        for marker_item in self.marker_items:
            marker_item.setData([], [])

        # 新しいマーカーを表示
        for bin_idx, value in features:
            if bin_idx in self.visible_bins:
                self.marker_items[bin_idx].setData([self.time_buffer_size - 1], [value])

    def on_bin_visibility_changed(self, bin_idx, state):
        """ビンの表示/非表示が変更されたときの処理"""
        if state == 2:  # Qt.Checked
            self.visible_bins.add(bin_idx)
        else:
            self.visible_bins.discard(bin_idx)
        self.update_plot_visibility()

    def update_plot_visibility(self):
        """プロットの表示/非表示を更新"""
        for i, (plot_item, marker_item) in enumerate(
            zip(self.plot_items, self.marker_items)
        ):
            visible = i in self.visible_bins
            plot_item.setVisible(visible)
            marker_item.setVisible(visible)

    def update_device_list(self):
        """利用可能な入力デバイスのリストを更新"""
        devices = sd.query_devices()
        input_devices = [
            (i, dev["name"])
            for i, dev in enumerate(devices)
            if dev["max_input_channels"] > 0
        ]

        self.device_combo.clear()
        for device_id, device_name in input_devices:
            self.device_combo.addItem(f"{device_name} (ID: {device_id})", device_id)

    def on_device_changed(self, index):
        """デバイスが変更されたときの処理"""
        if index < 0:
            return

        # 現在のスレッドを停止
        if self.audio_thread is not None:
            self.running = False
            self.audio_thread.join()
            self.running = True

        # 新しいデバイスIDを設定
        self.device_id = self.device_combo.currentData()

        # 新しいスレッドを開始
        self.audio_thread = threading.Thread(target=self.audio_callback)
        self.audio_thread.start()

    def audio_callback(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                device=self.device_id,
                callback=callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
            ):
                while self.running:
                    sd.sleep(100)
        except Exception as e:
            print(f"音声入力エラー: {e}")
            # エラーが発生した場合はデフォルトデバイスに戻す
            self.device_id = None
            self.device_combo.setCurrentIndex(0)

    def update_plot(self):
        try:
            while True:
                audio_data = self.audio_queue.get_nowait()
                self.audio_buffer.extend(audio_data.flatten())
        except queue.Empty:
            pass

        if len(self.audio_buffer) >= self.fft_size:
            # オーバーラップFFTの計算
            audio_array = np.array(self.audio_buffer)
            window = np.hanning(self.fft_size)

            # 最新のデータでFFTを計算
            fft_data = np.abs(np.fft.rfft(audio_array[-self.fft_size :] * window))
            # fft_data = fft_data / np.max(fft_data)  # 正規化

            # 低周波域のデータを更新
            new_data = fft_data[: self.low_freq_bins]
            self.freq_data = np.roll(self.freq_data, -1, axis=1)
            self.freq_data[:, -1] = new_data

            # 特徴点を検出
            features = self.detect_features(new_data)

            # マーカーを更新
            self.update_markers(features)

            # プロットの更新（表示されているビンのみ）
            for i in range(self.low_freq_bins):
                if i in self.visible_bins:
                    self.plot_items[i].setData(self.freq_data[i])

    def closeEvent(self, event):
        self.running = False
        if self.audio_thread is not None:
            self.audio_thread.join()
        self.plot_timer.stop()
        self.bg_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FFTLogger()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec())
