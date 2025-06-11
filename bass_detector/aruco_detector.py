#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUcoマーカー姿勢推定アプリケーション
PySide6とOpenCVを使用してArUcoマーカーの検出と姿勢推定を行います
"""

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
    QTextEdit,
    QSizePolicy,
)
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from typing import Optional, Tuple, List


class ArUcoDetector(QMainWindow):
    """ArUcoマーカーの検出と姿勢推定を行うメインアプリケーション"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArUco マーカー姿勢推定アプリ")
        self.setGeometry(100, 100, 1200, 800)

        # OpenCV関連の初期化
        self.cap = None
        self.camera_index = 0
        self.is_camera_open = False

        # ArUco関連の初期化
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # カメラキャリブレーション用パラメータ（デフォルト値）
        self.camera_matrix = np.array(
            [[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.zeros((4, 1))
        self.marker_length = 0.05  # マーカーサイズ（メートル）

        # タイマー設定
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # UI初期化
        self.init_ui()

    def init_ui(self):
        """UIの初期化"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)

        # 左側：ビデオ表示エリア
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # ビデオ表示ラベル
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "border: 2px solid gray; background-color: black;"
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("カメラ映像がここに表示されます")
        left_layout.addWidget(self.video_label)

        # カメラ制御ボタン
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("カメラ開始")
        self.start_button.clicked.connect(self.toggle_camera)
        self.stop_button = QPushButton("カメラ停止")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)

        main_layout.addWidget(left_widget, 2)

        # 右側：設定と情報パネル
        right_widget = QWidget()
        right_widget.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_widget)

        # カメラ設定グループ
        camera_group = QGroupBox("カメラ設定")
        camera_layout = QGridLayout(camera_group)

        camera_layout.addWidget(QLabel("カメラID:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2", "3"])
        self.camera_combo.currentTextChanged.connect(self.change_camera)
        camera_layout.addWidget(self.camera_combo, 0, 1)

        right_layout.addWidget(camera_group)

        # ArUco設定グループ
        aruco_group = QGroupBox("ArUco設定")
        aruco_layout = QGridLayout(aruco_group)

        aruco_layout.addWidget(QLabel("辞書タイプ:"), 0, 0)
        self.dict_combo = QComboBox()
        self.dict_combo.addItems(
            [
                "DICT_4X4_50",
                "DICT_4X4_100",
                "DICT_4X4_250",
                "DICT_5X5_50",
                "DICT_6X6_50",
                "DICT_6X6_250",
            ]
        )
        self.dict_combo.setCurrentText("DICT_6X6_250")
        self.dict_combo.currentTextChanged.connect(self.change_aruco_dict)
        aruco_layout.addWidget(self.dict_combo, 0, 1)

        aruco_layout.addWidget(QLabel("マーカーサイズ(m):"), 1, 0)
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(0.001, 1.0)
        self.marker_size_spin.setValue(0.05)
        self.marker_size_spin.setSingleStep(0.01)
        self.marker_size_spin.setDecimals(3)
        self.marker_size_spin.valueChanged.connect(self.change_marker_size)
        aruco_layout.addWidget(self.marker_size_spin, 1, 1)

        right_layout.addWidget(aruco_group)

        # カメラキャリブレーション設定グループ
        calib_group = QGroupBox("カメラキャリブレーション")
        calib_layout = QGridLayout(calib_group)

        # 焦点距離
        calib_layout.addWidget(QLabel("焦点距離 fx:"), 0, 0)
        self.fx_spin = QDoubleSpinBox()
        self.fx_spin.setRange(100, 2000)
        self.fx_spin.setValue(800)
        self.fx_spin.valueChanged.connect(self.update_camera_matrix)
        calib_layout.addWidget(self.fx_spin, 0, 1)

        calib_layout.addWidget(QLabel("焦点距離 fy:"), 1, 0)
        self.fy_spin = QDoubleSpinBox()
        self.fy_spin.setRange(100, 2000)
        self.fy_spin.setValue(800)
        self.fy_spin.valueChanged.connect(self.update_camera_matrix)
        calib_layout.addWidget(self.fy_spin, 1, 1)

        # 主点
        calib_layout.addWidget(QLabel("主点 cx:"), 2, 0)
        self.cx_spin = QDoubleSpinBox()
        self.cx_spin.setRange(0, 1000)
        self.cx_spin.setValue(320)
        self.cx_spin.valueChanged.connect(self.update_camera_matrix)
        calib_layout.addWidget(self.cx_spin, 2, 1)

        calib_layout.addWidget(QLabel("主点 cy:"), 3, 0)
        self.cy_spin = QDoubleSpinBox()
        self.cy_spin.setRange(0, 1000)
        self.cy_spin.setValue(240)
        self.cy_spin.valueChanged.connect(self.update_camera_matrix)
        calib_layout.addWidget(self.cy_spin, 3, 1)

        right_layout.addWidget(calib_group)

        # 検出情報表示エリア
        info_group = QGroupBox("検出情報")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(200)
        self.info_text.setFont(QFont("Courier", 9))
        info_layout.addWidget(self.info_text)

        right_layout.addWidget(info_group)

        # スペーサー
        right_layout.addStretch()

        main_layout.addWidget(right_widget, 1)

    def toggle_camera(self):
        """カメラの開始/停止を切り替え"""
        if not self.is_camera_open:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """カメラを開始"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.info_text.append("カメラを開けませんでした")
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.is_camera_open = True
            self.start_button.setText("カメラ停止")
            self.stop_button.setEnabled(True)
            self.timer.start(33)  # 約30 FPS
            self.info_text.append("カメラを開始しました")

        except Exception as e:
            self.info_text.append(f"カメラ開始エラー: {str(e)}")

    def stop_camera(self):
        """カメラを停止"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.is_camera_open = False
        self.start_button.setText("カメラ開始")
        self.stop_button.setEnabled(False)
        self.video_label.setText("カメラ映像がここに表示されます")
        self.info_text.append("カメラを停止しました")

    def change_camera(self, camera_id_str: str):
        """カメラIDを変更"""
        self.camera_index = int(camera_id_str)
        if self.is_camera_open:
            self.stop_camera()
            self.start_camera()

    def change_aruco_dict(self, dict_name: str):
        """ArUco辞書を変更"""
        dict_mapping = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        if dict_name in dict_mapping:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_mapping[dict_name])
            self.info_text.append(f"ArUco辞書を{dict_name}に変更しました")

    def change_marker_size(self, size: float):
        """マーカーサイズを変更"""
        self.marker_length = size

    def update_camera_matrix(self):
        """カメラマトリックスを更新"""
        fx = self.fx_spin.value()
        fy = self.fy_spin.value()
        cx = self.cx_spin.value()
        cy = self.cy_spin.value()

        self.camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

    def update_frame(self):
        """フレームの更新"""
        if not self.cap or not self.is_camera_open:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # ArUcoマーカーの検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        # マーカーが検出された場合
        if ids is not None:
            # マーカーの境界を描画
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 姿勢推定 - OpenCV4では solvePnP を使用
            # マーカーのワールド座標を定義（マーカー中心が原点）
            half_size = self.marker_length / 2
            marker_points_3d = np.array(
                [
                    [-half_size, half_size, 0],
                    [half_size, half_size, 0],
                    [half_size, -half_size, 0],
                    [-half_size, -half_size, 0],
                ],
                dtype=np.float32,
            )

            rvecs = []
            tvecs = []

            # 各マーカーの姿勢を計算
            for i in range(len(ids)):
                marker_corners = corners[i].reshape((-1, 2))

                # solvePnP で姿勢推定
                success, rvec, tvec = cv2.solvePnP(
                    marker_points_3d,
                    marker_corners,
                    self.camera_matrix,
                    self.dist_coeffs,
                )

                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)

                    # 軸を描画
                    cv2.drawFrameAxes(
                        frame,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                        self.marker_length * 0.5,
                    )

            # 検出情報を更新
            if rvecs and tvecs:
                self.update_detection_info(ids, rvecs, tvecs)
        else:
            # マーカーが検出されない場合
            self.info_text.clear()
            self.info_text.append("マーカーが検出されていません")

        # フレームをQtのラベルに表示
        self.display_frame(frame)

    def update_detection_info(self, ids: np.ndarray, rvecs: list, tvecs: list):
        """検出情報を更新"""
        self.info_text.clear()
        self.info_text.append(f"検出されたマーカー数: {len(ids)}")
        self.info_text.append("-" * 40)

        for i in range(len(ids)):
            marker_id = ids[i][0]
            rvec = rvecs[i]
            tvec = tvecs[i]

            # 回転ベクトルを回転行列に変換
            rmat, _ = cv2.Rodrigues(rvec)

            # オイラー角を計算（ZYX順序）
            sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rmat[2, 1], rmat[2, 2])
                y = np.arctan2(-rmat[2, 0], sy)
                z = np.arctan2(rmat[1, 0], rmat[0, 0])
            else:
                x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                y = np.arctan2(-rmat[2, 0], sy)
                z = 0

            # 度数に変換
            roll = np.degrees(x)
            pitch = np.degrees(y)
            yaw = np.degrees(z)

            # カメラの光軸に対するマーカーZ軸の傾き角を計算
            # マーカーのZ軸ベクトルは回転行列の第3列
            marker_z_axis = rmat[:, 2]  # [x, y, z]成分
            # カメラの光軸ベクトル（カメラ座標系でのZ軸）
            camera_z_axis = np.array([0, 0, 1])

            # 内積を計算してコサインを求める
            # cos(θ) = a·b / (|a||b|) だが、両方とも単位ベクトルなので cos(θ) = a·b
            dot_product = np.dot(marker_z_axis, camera_z_axis)
            # 値を[-1, 1]の範囲にクリップ（数値誤差対策）
            dot_product = np.clip(dot_product, -1.0, 1.0)
            # 傾き角を計算（度数）
            # abs()を使用してマーカーの表裏に関係なく傾き角を取得
            # 0°: 完全に平行（マーカーがカメラを正面に向いている）
            # 90°: 完全に垂直（マーカーの端面がカメラを向いている）
            tilt_angle = np.degrees(np.arccos(np.abs(dot_product)))

            # X軸、Y軸に対する傾き角を計算（正負を含む）
            # X軸方向の傾き：左右の傾き（正：右傾き、負：左傾き）
            # Y軸方向の傾き：上下の傾き（正：上傾き、負：下傾き）
            #
            # カメラ座標系：
            # - X軸: 右方向が正
            # - Y軸: 下方向が正（画像座標系）
            # - Z軸: カメラから奥方向が正
            #
            # arctan2(x, z) でX軸周りの傾きを計算
            # arctan2(y, z) でY軸周りの傾きを計算
            if abs(marker_z_axis[2]) > 1e-6:  # ゼロ除算を防ぐ
                x_tilt_rad = np.arctan2(marker_z_axis[0], marker_z_axis[2])
                y_tilt_rad = np.arctan2(marker_z_axis[1], marker_z_axis[2])

                x_tilt_deg = np.degrees(x_tilt_rad)
                y_tilt_deg = np.degrees(y_tilt_rad)
            else:
                # マーカーがほぼ垂直の場合
                x_tilt_deg = 90.0 if marker_z_axis[0] > 0 else -90.0
                y_tilt_deg = 90.0 if marker_z_axis[1] > 0 else -90.0

            # 傾き方向の説明文を生成
            def get_tilt_description(angle, axis):
                if abs(angle) < 5:
                    return "まっすぐ"
                elif axis == "x":
                    return (
                        f"右に{angle:.1f}°" if angle > 0 else f"左に{abs(angle):.1f}°"
                    )
                else:  # axis == 'y'
                    return (
                        f"上に{angle:.1f}°" if angle > 0 else f"下に{abs(angle):.1f}°"
                    )

            x_description = get_tilt_description(x_tilt_deg, "x")
            y_description = get_tilt_description(y_tilt_deg, "y")

            # 傾き角に基づく状態の判定
            if tilt_angle < 15:
                tilt_status = "正面向き"
            elif tilt_angle < 30:
                tilt_status = "やや傾斜"
            elif tilt_angle < 60:
                tilt_status = "大きく傾斜"
            elif tilt_angle < 75:
                tilt_status = "ほぼ垂直"
            else:
                tilt_status = "垂直（端面）"

            self.info_text.append(f"マーカーID: {marker_id}")
            self.info_text.append(f"位置 (m):")
            self.info_text.append(f"  X: {tvec[0][0]:.4f}")
            self.info_text.append(f"  Y: {tvec[1][0]:.4f}")
            self.info_text.append(f"  Z: {tvec[2][0]:.4f}")
            self.info_text.append(f"回転 (度):")
            self.info_text.append(f"  Roll:  {roll:.2f}")
            self.info_text.append(f"  Pitch: {pitch:.2f}")
            self.info_text.append(f"  Yaw:   {yaw:.2f}")
            self.info_text.append(f"傾き角:")
            self.info_text.append(f"  カメラ光軸に対する傾き: {tilt_angle:.2f}°")
            self.info_text.append(
                f"  X軸方向の傾き: {x_tilt_deg:+.1f}° ({x_description})"
            )
            self.info_text.append(
                f"  Y軸方向の傾き: {y_tilt_deg:+.1f}° ({y_description})"
            )
            self.info_text.append(f"状態: {tilt_status}")
            self.info_text.append("-" * 40)

    def display_frame(self, frame: np.ndarray):
        """フレームをラベルに表示"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # ラベルサイズに合わせてスケーリング
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        self.stop_camera()
        event.accept()


def main():
    """メイン関数"""
    app = QApplication(sys.argv)

    # 日本語フォントの設定
    app.setStyle("Fusion")

    window = ArUcoDetector()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
