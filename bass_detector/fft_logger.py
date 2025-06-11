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
    QPushButton,
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
import serial
import serial.tools.list_ports


class ActivationFunction(Enum):
    LINEAR = "Linear"
    SIGMOID = "Sigmoid"
    RELU = "ReLU"
    TANH = "Tanh"
    STEP = "Step"
    ADAPTIVE_EXP = "Adaptive Exponential"
    ADAPTIVE_SIGMOID = "Adaptive Sigmoid"
    PEAK_PULSE = "Peak Pulse"  # 新しいアクティベーション関数


class Peak:
    def __init__(self, bin_idx, intensity, timestamp):
        self.bin_idx = bin_idx
        self.intensity = intensity
        self.timestamp = timestamp


class FFTLogger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bass Detector - FFT Logger")

        # シリアル通信の設定
        self.serial_port = None
        self.serial_output_enabled = False
        self.serial_update_interval = 10  # シリアル出力の更新間隔（ミリ秒）

        # 音声入力の設定
        self.sample_rate = 44100
        self.chunk_size = 512
        self.overlap = 0.90
        self.hop_size = int(self.chunk_size * (1 - self.overlap))
        self.device_id = 8  # デフォルトデバイスIDを8に設定

        # バッファの設定（3秒分）
        self.buffer_size = int(3 * self.sample_rate)
        self.audio_buffer = deque(maxlen=self.buffer_size)

        # FFTの設定
        self.fft_size = 2048
        self.freq_bins = np.fft.rfftfreq(self.fft_size, 1 / self.sample_rate)
        self.low_freq_bins = 10  # 低周波域のビン数
        # 中央の4つのビンだけを表示するように設定
        center_start = (self.low_freq_bins - 4) // 2
        self.visible_bins = set(range(center_start, center_start + 4))

        # 特徴点検出の設定
        self.threshold = 80.0  # 閾値
        self.sensitivity = 0.5  # 感度（0-1）
        self.marker_history = deque(maxlen=50)  # マーカーの履歴

        # ピーク管理の設定
        self.peak_history = deque(maxlen=100)  # ピークの履歴
        self.peak_decay_rate = 0.5  # ピークの減衰率
        self.peak_intensity_scale = 0.5  # ピーク強度のスケール
        self.current_peak_sum = 0.0  # 現在のピーク合計値
        self.max_peak_sum = 1000.0  # ピーク合計の最大値（明るさの正規化用）

        # アダプティブ関数用の設定
        self.adaptive_window_size = 2.0  # 最大値の保持期間（秒）
        self.adaptive_max_history = deque(
            maxlen=int(self.adaptive_window_size * self.sample_rate / self.chunk_size)
        )
        self.adaptive_threshold_ratio = 0.3  # 最大値に対する閾値の比率
        self.adaptive_exponent = 2.0  # 指数の強さ

        # アクティベーション関数の設定
        self.activation_function = ActivationFunction.PEAK_PULSE
        self.activation_params = {
            "sigmoid_steepness": 0.01,  # シグモイドの急峻さ
            "relu_threshold": 100.0,  # ReLUの閾値
            "step_threshold": 500.0,  # ステップ関数の閾値
            "tanh_scale": 0.005,  # Tanhのスケール
            "adaptive_threshold_ratio": 0.3,  # アダプティブ閾値の比率
            "adaptive_exponent": 2.0,  # アダプティブ指数の強さ
            "adaptive_sigmoid_center": 0.5,  # アダプティブシグモイドの中心位置
            "adaptive_sigmoid_steepness": 0.001,  # アダプティブシグモイドの急峻さ
            "adaptive_sigmoid_min_max": 100.0,  # アダプティブシグモイドの最大値の下限
            "adaptive_sigmoid_clip_threshold": 0.0,  # アダプティブシグモイドのクリップ閾値
            "peak_pulse_threshold": 80.0,  # ピーク検出の閾値
            "peak_pulse_sensitivity": 0.5,  # ピーク検出の感度（微分値の閾値）
            "peak_pulse_width": 0.1,  # パルス幅（秒）
            "peak_pulse_cooldown": 0.2,  # 不感帯時間（秒）
        }

        # ピークパルス用の状態管理
        self.peak_pulse_state = {
            "is_pulsing": False,  # パルス出力中かどうか
            "prev_value": 0.0,  # 前回の値（微分計算用）
            "last_peak_time": 0.0,  # 最後にピークを検出した時間
            "pulse_value": 0.0,  # 現在のパルス値
            "pulse_start_time": 0.0,  # 現在のパルスの開始時間
        }

        # パルス制御用のタイマー
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.end_pulse)
        self.pulse_timer.setSingleShot(True)  # ワンショットタイマーとして設定

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

        # 光るバー用の設定
        self.light_bar = None
        self.light_bar_height = 30  # バーの高さ（ピクセル）

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

        # シリアル出力用のタイマー
        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self.update_serial_output)
        self.serial_timer.start(self.serial_update_interval)

    def get_activation_function(self) -> Callable[[float], float]:
        """現在選択されているアクティベーション関数を返す"""
        if self.activation_function == ActivationFunction.PEAK_PULSE:
            current_time = time.time()
            threshold = self.activation_params["peak_pulse_threshold"]
            sensitivity = self.activation_params["peak_pulse_sensitivity"]
            cooldown = self.activation_params["peak_pulse_cooldown"]
            pulse_width = self.activation_params["peak_pulse_width"]

            def peak_pulse(x):
                # 状態の更新
                state = self.peak_pulse_state
                current_time = time.time()

                # 微分値の計算（変化率）
                diff = x - state["prev_value"]
                state["prev_value"] = x

                # パルス出力中の処理
                if state["is_pulsing"]:
                    # パルス幅の経過時間をチェック
                    elapsed = current_time - state["pulse_start_time"]
                    if elapsed >= pulse_width:
                        state["is_pulsing"] = False
                        state["pulse_value"] = 0.0
                        print(f"パルス終了（時間経過）: 経過時間={elapsed:.3f}秒")
                        return 0.0
                    return state["pulse_value"]

                # 不感帯時間のチェック
                if current_time - state["last_peak_time"] < cooldown:
                    if state["pulse_value"] > 0:
                        print(
                            f"不感帯時間中: {current_time - state['last_peak_time']:.3f}秒"
                        )
                    state["pulse_value"] = 0.0
                    return 0.0

                # ピーク検出（閾値と微分値の両方をチェック）
                if x > threshold and diff > threshold * sensitivity:
                    state["last_peak_time"] = current_time
                    state["pulse_start_time"] = current_time
                    state["is_pulsing"] = True
                    state["pulse_value"] = 1.0
                    print(
                        f"パルス開始: 幅={pulse_width:.3f}秒, 時刻={current_time:.3f}"
                    )
                    return 1.0

                state["pulse_value"] = 0.0
                return 0.0

            return peak_pulse
        elif self.activation_function == ActivationFunction.LINEAR:
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
        elif self.activation_function == ActivationFunction.ADAPTIVE_EXP:
            # 現在の最大値を取得
            current_max = (
                max(self.adaptive_max_history)
                if self.adaptive_max_history
                else self.max_peak_sum
            )
            threshold = current_max * self.activation_params["adaptive_threshold_ratio"]
            exponent = self.activation_params["adaptive_exponent"]

            def adaptive_exp(x):
                if x < threshold:
                    return 0.0
                # 閾値以上の値を0-1の範囲に正規化して指数関数を適用
                normalized = (x - threshold) / (current_max - threshold)
                return min(1.0, normalized**exponent)

            return adaptive_exp
        elif self.activation_function == ActivationFunction.ADAPTIVE_SIGMOID:
            # 現在の最大値を取得（下限値を考慮）
            current_max = max(
                max(self.adaptive_max_history) if self.adaptive_max_history else 0.0,
                self.activation_params["adaptive_sigmoid_min_max"],
            )
            center_ratio = self.activation_params["adaptive_sigmoid_center"]
            steepness = self.activation_params["adaptive_sigmoid_steepness"]
            clip_threshold = self.activation_params["adaptive_sigmoid_clip_threshold"]
            center = current_max * center_ratio

            def adaptive_sigmoid(x):
                # シグモイド関数を適用（中心位置と急峻さを動的に調整）
                sigmoid_output = 1 / (1 + np.exp(-steepness * (x - center)))
                # シグモイド出力がクリップ閾値以下の場合は0を返す
                return 0.0 if sigmoid_output <= clip_threshold else sigmoid_output

            return adaptive_sigmoid
        else:
            return lambda x: min(1.0, max(0.0, x / self.max_peak_sum))

    def setup_ui(self):
        # メインウィジェットの設定
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # シリアル通信設定用のウィジェット
        serial_layout = QHBoxLayout()
        serial_label = QLabel("シリアルポート:")
        self.serial_combo = QComboBox()
        self.update_serial_ports()
        self.serial_combo.currentIndexChanged.connect(self.on_serial_port_changed)

        self.serial_button = QPushButton("接続")
        self.serial_button.setCheckable(True)
        self.serial_button.clicked.connect(self.toggle_serial_output)

        serial_layout.addWidget(serial_label)
        serial_layout.addWidget(self.serial_combo)
        serial_layout.addWidget(self.serial_button)
        serial_layout.addStretch()
        layout.addLayout(serial_layout)

        # デバイス選択用のウィジェット
        device_layout = QHBoxLayout()
        device_label = QLabel("入力デバイス:")
        self.device_combo = QComboBox()
        self.update_device_list()
        # デフォルトデバイス（ID=8）を選択
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == self.device_id:
                self.device_combo.setCurrentIndex(i)
                break
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

        # アダプティブ・エキスポネンシャル用のパラメータ
        adaptive_widget = QWidget()
        adaptive_layout = QHBoxLayout(adaptive_widget)
        adaptive_layout.setContentsMargins(0, 0, 0, 0)

        # 閾値比率の設定
        threshold_ratio_label = QLabel("閾値比率:")
        threshold_ratio_spin = QDoubleSpinBox()
        threshold_ratio_spin.setRange(0.1, 0.9)
        threshold_ratio_spin.setSingleStep(0.1)
        threshold_ratio_spin.setValue(
            self.activation_params["adaptive_threshold_ratio"]
        )
        threshold_ratio_spin.valueChanged.connect(
            lambda v: self.on_param_changed("adaptive_threshold_ratio", v)
        )

        # 指数の強さの設定
        exponent_label = QLabel("指数の強さ:")
        exponent_spin = QDoubleSpinBox()
        exponent_spin.setRange(0.5, 5.0)
        exponent_spin.setSingleStep(0.1)
        exponent_spin.setValue(self.activation_params["adaptive_exponent"])
        exponent_spin.valueChanged.connect(
            lambda v: self.on_param_changed("adaptive_exponent", v)
        )

        adaptive_layout.addWidget(threshold_ratio_label)
        adaptive_layout.addWidget(threshold_ratio_spin)
        adaptive_layout.addWidget(exponent_label)
        adaptive_layout.addWidget(exponent_spin)
        adaptive_layout.addStretch()

        self.param_widgets["adaptive_threshold_ratio"] = adaptive_widget
        self.param_widgets["adaptive_exponent"] = adaptive_widget

        # アダプティブシグモイド用のパラメータ
        adaptive_sigmoid_widget = QWidget()
        adaptive_sigmoid_layout = QVBoxLayout(adaptive_sigmoid_widget)
        adaptive_sigmoid_layout.setContentsMargins(0, 0, 0, 0)

        # 上部のパラメータ（水平レイアウト）
        top_params = QHBoxLayout()

        # 中心位置の設定
        center_label = QLabel("中心位置:")
        center_spin = QDoubleSpinBox()
        center_spin.setRange(0.1, 0.9)
        center_spin.setSingleStep(0.1)
        center_spin.setValue(self.activation_params["adaptive_sigmoid_center"])
        center_spin.valueChanged.connect(
            lambda v: self.on_param_changed("adaptive_sigmoid_center", v)
        )

        # 急峻さの設定
        steepness_label = QLabel("急峻さ:")
        steepness_spin = QDoubleSpinBox()
        steepness_spin.setRange(0.0001, 1.0)
        steepness_spin.setSingleStep(0.0001)
        steepness_spin.setDecimals(4)
        steepness_spin.setValue(self.activation_params["adaptive_sigmoid_steepness"])
        steepness_spin.valueChanged.connect(
            lambda v: self.on_param_changed("adaptive_sigmoid_steepness", v)
        )

        top_params.addWidget(center_label)
        top_params.addWidget(center_spin)
        top_params.addWidget(steepness_label)
        top_params.addWidget(steepness_spin)
        top_params.addStretch()

        # 下部のパラメータ（水平レイアウト）
        bottom_params = QHBoxLayout()

        # 最大値の下限設定
        min_max_label = QLabel("最大値の下限:")
        min_max_spin = QDoubleSpinBox()
        min_max_spin.setRange(0.0, 1000.0)
        min_max_spin.setSingleStep(10.0)
        min_max_spin.setValue(self.activation_params["adaptive_sigmoid_min_max"])
        min_max_spin.valueChanged.connect(
            lambda v: self.on_param_changed("adaptive_sigmoid_min_max", v)
        )

        # クリップ閾値の設定
        clip_label = QLabel("クリップ閾値:")
        clip_spin = QDoubleSpinBox()
        clip_spin.setRange(0.0, 1000.0)
        clip_spin.setSingleStep(10.0)
        clip_spin.setValue(self.activation_params["adaptive_sigmoid_clip_threshold"])
        clip_spin.valueChanged.connect(
            lambda v: self.on_param_changed("adaptive_sigmoid_clip_threshold", v)
        )

        bottom_params.addWidget(min_max_label)
        bottom_params.addWidget(min_max_spin)
        bottom_params.addWidget(clip_label)
        bottom_params.addWidget(clip_spin)
        bottom_params.addStretch()

        # レイアウトの組み立て
        adaptive_sigmoid_layout.addLayout(top_params)
        adaptive_sigmoid_layout.addLayout(bottom_params)

        self.param_widgets["adaptive_sigmoid_center"] = adaptive_sigmoid_widget
        self.param_widgets["adaptive_sigmoid_steepness"] = adaptive_sigmoid_widget
        self.param_widgets["adaptive_sigmoid_min_max"] = adaptive_sigmoid_widget
        self.param_widgets["adaptive_sigmoid_clip_threshold"] = adaptive_sigmoid_widget

        # パラメータウィジェットをレイアウトに追加
        for widget in self.param_widgets.values():
            activation_layout.addWidget(widget)

        layout.addWidget(activation_group)

        # ピークパルス用のパラメータ
        peak_pulse_widget = QWidget()
        peak_pulse_layout = QVBoxLayout(peak_pulse_widget)
        peak_pulse_layout.setContentsMargins(0, 0, 0, 0)

        # 上部のパラメータ（水平レイアウト）
        top_params = QHBoxLayout()

        # ピーク検出閾値の設定
        threshold_label = QLabel("ピーク閾値:")
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.0, 1000.0)
        threshold_spin.setSingleStep(10.0)
        threshold_spin.setValue(self.activation_params["peak_pulse_threshold"])
        threshold_spin.valueChanged.connect(
            lambda v: self.on_param_changed("peak_pulse_threshold", v)
        )

        # 感度の設定
        sensitivity_label = QLabel("感度:")
        sensitivity_spin = QDoubleSpinBox()
        sensitivity_spin.setRange(0.0, 1.0)
        sensitivity_spin.setSingleStep(0.1)
        sensitivity_spin.setValue(self.activation_params["peak_pulse_sensitivity"])
        sensitivity_spin.valueChanged.connect(
            lambda v: self.on_param_changed("peak_pulse_sensitivity", v)
        )

        top_params.addWidget(threshold_label)
        top_params.addWidget(threshold_spin)
        top_params.addWidget(sensitivity_label)
        top_params.addWidget(sensitivity_spin)
        top_params.addStretch()

        # 下部のパラメータ（水平レイアウト）
        bottom_params = QHBoxLayout()

        # パルス幅の設定
        width_label = QLabel("パルス幅(秒):")
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.01, 1.0)
        width_spin.setSingleStep(0.01)
        width_spin.setValue(self.activation_params["peak_pulse_width"])
        width_spin.valueChanged.connect(
            lambda v: self.on_param_changed("peak_pulse_width", v)
        )

        # 不感帯時間の設定
        cooldown_label = QLabel("不感帯時間(秒):")
        cooldown_spin = QDoubleSpinBox()
        cooldown_spin.setRange(0.1, 5.0)
        cooldown_spin.setSingleStep(0.1)
        cooldown_spin.setValue(self.activation_params["peak_pulse_cooldown"])
        cooldown_spin.valueChanged.connect(
            lambda v: self.on_param_changed("peak_pulse_cooldown", v)
        )

        bottom_params.addWidget(width_label)
        bottom_params.addWidget(width_spin)
        bottom_params.addWidget(cooldown_label)
        bottom_params.addWidget(cooldown_spin)
        bottom_params.addStretch()

        # レイアウトの組み立て
        peak_pulse_layout.addLayout(top_params)
        peak_pulse_layout.addLayout(bottom_params)

        # パラメータウィジェットを登録
        self.param_widgets["peak_pulse_threshold"] = peak_pulse_widget
        self.param_widgets["peak_pulse_sensitivity"] = peak_pulse_widget
        self.param_widgets["peak_pulse_width"] = peak_pulse_widget
        self.param_widgets["peak_pulse_cooldown"] = peak_pulse_widget

        # パラメータウィジェットをレイアウトに追加
        for widget in self.param_widgets.values():
            activation_layout.addWidget(widget)

        layout.addWidget(activation_group)

        # ビン選択用のチェックボックス
        bin_layout = QGridLayout()
        self.bin_checkboxes = []
        for i in range(self.low_freq_bins):
            checkbox = QCheckBox(f"{self.freq_bins[i]:.1f} Hz")
            checkbox.setChecked(
                i in self.visible_bins
            )  # 初期状態をvisible_binsに合わせる
            checkbox.stateChanged.connect(
                lambda state, idx=i: self.on_bin_visibility_changed(idx, state)
            )
            self.bin_checkboxes.append(checkbox)
            bin_layout.addWidget(checkbox, i // 5, i % 5)
        layout.addLayout(bin_layout)

        # プロットウィジェットの設定
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Power")
        self.plot_widget.setLabel("bottom", "Time (frames)")
        self.plot_widget.setYRange(0, 300)
        self.plot_widget.setXRange(0, self.time_buffer_size)
        plot_layout.addWidget(self.plot_widget)

        # 光るバーの設定
        self.light_bar = QWidget()
        self.light_bar.setFixedHeight(self.light_bar_height)
        self.light_bar.setStyleSheet("background-color: black;")
        plot_layout.addWidget(self.light_bar)

        layout.addWidget(plot_container)

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
        print(f"パラメータ変更: {param_name} = {value}")
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
        elif self.activation_function == ActivationFunction.ADAPTIVE_EXP:
            self.param_widgets["adaptive_threshold_ratio"].setVisible(True)
            self.param_widgets["adaptive_exponent"].setVisible(True)
        elif self.activation_function == ActivationFunction.ADAPTIVE_SIGMOID:
            self.param_widgets["adaptive_sigmoid_center"].setVisible(True)
            self.param_widgets["adaptive_sigmoid_steepness"].setVisible(True)
            self.param_widgets["adaptive_sigmoid_min_max"].setVisible(True)
            self.param_widgets["adaptive_sigmoid_clip_threshold"].setVisible(True)
        elif self.activation_function == ActivationFunction.PEAK_PULSE:
            self.param_widgets["peak_pulse_threshold"].setVisible(True)
            self.param_widgets["peak_pulse_sensitivity"].setVisible(True)
            self.param_widgets["peak_pulse_width"].setVisible(True)
            self.param_widgets["peak_pulse_cooldown"].setVisible(True)

    def set_background_color(self, intensity):
        """光るバーの色を設定"""
        try:
            # アクティベーション関数を適用
            activation_func = self.get_activation_function()
            normalized_intensity = activation_func(intensity)

            # 強度を0-255の範囲に変換（確実に0-1の範囲に収める）
            normalized_intensity = min(1.0, max(0.0, normalized_intensity))
            brightness = int(255 * normalized_intensity)

            # 光るバーの色を更新
            if self.light_bar is not None:
                color = QColor(brightness, brightness, brightness)
                self.light_bar.setStyleSheet(f"background-color: {color.name()};")
        except Exception as e:
            print(f"光るバーの更新でエラーが発生しました: {e}")
            # エラー時は黒色に設定
            if self.light_bar is not None:
                self.light_bar.setStyleSheet("background-color: black;")

    def update_background(self):
        """光るバーを更新"""
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

        # 光るバーを更新
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
            # デバイスID=8が見つからない場合は、最初のデバイスを使用
            if device_id == self.device_id:
                self.device_id = device_id
            elif self.device_combo.count() == 1:
                self.device_id = device_id

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

            # 低周波域のデータを更新
            new_data = fft_data[: self.low_freq_bins]
            self.freq_data = np.roll(self.freq_data, -1, axis=1)
            self.freq_data[:, -1] = new_data

            # アダプティブ・エキスポネンシャル用の最大値を更新
            current_sum = np.sum(new_data)
            self.adaptive_max_history.append(current_sum)

            # 特徴点を検出
            features = self.detect_features(new_data)

            # マーカーを更新
            self.update_markers(features)

            # プロットの更新（表示されているビンのみ）
            for i in range(self.low_freq_bins):
                if i in self.visible_bins:
                    self.plot_items[i].setData(self.freq_data[i])

    def update_serial_ports(self):
        """利用可能なシリアルポートのリストを更新"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.serial_combo.clear()
        for port in ports:
            self.serial_combo.addItem(port)

    def on_serial_port_changed(self, index):
        """シリアルポートが変更されたときの処理"""
        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None
        self.serial_button.setChecked(False)
        self.serial_output_enabled = False

    def toggle_serial_output(self, checked):
        """シリアル出力の有効/無効を切り替え"""
        if checked:
            try:
                port = self.serial_combo.currentText()
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                self.serial_output_enabled = True
                self.serial_button.setText("切断")
            except Exception as e:
                print(f"シリアルポートの接続に失敗しました: {e}")
                self.serial_button.setChecked(False)
                self.serial_output_enabled = False
        else:
            if self.serial_port is not None:
                self.serial_port.close()
            self.serial_output_enabled = False
            self.serial_button.setText("接続")

    def update_serial_output(self):
        """シリアルポートに現在のピーク値を出力"""
        if not self.serial_output_enabled or self.serial_port is None:
            return

        try:
            # 現在のピーク合計値を0-255の範囲に正規化
            activation_func = self.get_activation_function()
            normalized_intensity = activation_func(self.current_peak_sum)
            brightness = min(255, int(255 * normalized_intensity))

            # シリアルポートに出力
            self.serial_port.write(bytes([brightness]))
        except Exception as e:
            print(f"シリアル出力でエラーが発生しました: {e}")
            self.serial_output_enabled = False
            self.serial_button.setChecked(False)
            if self.serial_port is not None:
                self.serial_port.close()
                self.serial_port = None

    def end_pulse(self):
        """パルスを終了する（タイマーによる終了）"""
        current_time = time.time()
        elapsed = current_time - self.peak_pulse_state["pulse_start_time"]
        print(f"パルス終了（タイマー）: 経過時間={elapsed:.3f}秒")
        self.peak_pulse_state["is_pulsing"] = False
        self.peak_pulse_state["pulse_value"] = 0.0

    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        self.running = False
        if self.audio_thread is not None:
            self.audio_thread.join()
        self.plot_timer.stop()
        self.bg_timer.stop()
        self.serial_timer.stop()
        self.pulse_timer.stop()  # パルスタイマーも停止
        if self.serial_port is not None:
            self.serial_port.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FFTLogger()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec())
