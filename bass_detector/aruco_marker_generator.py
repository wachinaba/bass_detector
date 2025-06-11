#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUcoマーカー生成スクリプト
テスト用のArUcoマーカーを生成します
"""

import cv2
import numpy as np
import os
from pathlib import Path


def generate_aruco_marker(
    marker_id: int,
    marker_size: int = 200,
    dict_type=cv2.aruco.DICT_6X6_250,
    save_path: str | None = None,
) -> np.ndarray:
    """
    ArUcoマーカーを生成する

    Args:
        marker_id: マーカーのID
        marker_size: マーカーのサイズ（ピクセル）
        dict_type: ArUco辞書のタイプ
        save_path: 保存先のパス（指定しない場合は保存しない）

    Returns:
        生成されたマーカー画像
    """
    # ArUco辞書を取得
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

    # マーカーを生成
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # 保存先が指定されている場合は保存
    if save_path:
        cv2.imwrite(save_path, marker_img)
        print(f"マーカー ID {marker_id} を {save_path} に保存しました")

    return marker_img


def generate_multiple_markers(
    marker_ids: list,
    marker_size: int = 200,
    dict_type=cv2.aruco.DICT_6X6_250,
    output_dir: str = "aruco_markers",
):
    """
    複数のArUcoマーカーを生成する

    Args:
        marker_ids: マーカーIDのリスト
        marker_size: マーカーのサイズ（ピクセル）
        dict_type: ArUco辞書のタイプ
        output_dir: 出力ディレクトリ
    """
    # 出力ディレクトリを作成
    Path(output_dir).mkdir(exist_ok=True)

    for marker_id in marker_ids:
        save_path = os.path.join(output_dir, f"aruco_marker_{marker_id}.png")
        generate_aruco_marker(marker_id, marker_size, dict_type, save_path)


def create_marker_board(
    marker_ids: list,
    markers_per_row: int = 3,
    marker_size: int = 150,
    margin: int = 50,
    dict_type=cv2.aruco.DICT_6X6_250,
    save_path: str = "aruco_board.png",
) -> np.ndarray:
    """
    複数のマーカーを配置したボードを作成する

    Args:
        marker_ids: マーカーIDのリスト
        markers_per_row: 1行あたりのマーカー数
        marker_size: マーカーのサイズ（ピクセル）
        margin: マーカー間の余白（ピクセル）
        dict_type: ArUco辞書のタイプ
        save_path: 保存先のパス

    Returns:
        ボード画像
    """
    # 行数を計算
    rows = (len(marker_ids) + markers_per_row - 1) // markers_per_row

    # ボード画像のサイズを計算
    board_width = markers_per_row * marker_size + (markers_per_row + 1) * margin
    board_height = rows * marker_size + (rows + 1) * margin

    # 白い背景のボード画像を作成
    board_img = np.ones((board_height, board_width), dtype=np.uint8) * 255

    # ArUco辞書を取得
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

    # マーカーを配置
    for i, marker_id in enumerate(marker_ids):
        row = i // markers_per_row
        col = i % markers_per_row

        # マーカーの位置を計算
        x = col * (marker_size + margin) + margin
        y = row * (marker_size + margin) + margin

        # マーカーを生成
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

        # ボードにマーカーを配置
        board_img[y : y + marker_size, x : x + marker_size] = marker_img

    # ボードを保存
    cv2.imwrite(save_path, board_img)
    print(f"マーカーボードを {save_path} に保存しました")

    return board_img


def generate_single_marker(dictionary, marker_id, marker_size, output_path):
    """単一のマーカーを生成して保存"""
    # マーカー画像を生成
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, marker_image)

    # 画像を保存
    cv2.imwrite(output_path, marker_image)
    print(f"マーカーID {marker_id} を保存しました: {output_path}")


def main():
    """メイン関数"""
    print("ArUcoマーカー生成スクリプト")
    print("=" * 40)

    # 辞書タイプの選択肢
    dict_options = {
        "1": cv2.aruco.DICT_4X4_50,
        "2": cv2.aruco.DICT_4X4_100,
        "3": cv2.aruco.DICT_4X4_250,
        "4": cv2.aruco.DICT_5X5_50,
        "5": cv2.aruco.DICT_6X6_50,
        "6": cv2.aruco.DICT_6X6_250,
    }

    print("ArUco辞書タイプを選択してください:")
    print("1. DICT_4X4_50")
    print("2. DICT_4X4_100")
    print("3. DICT_4X4_250")
    print("4. DICT_5X5_50")
    print("5. DICT_6X6_50")
    print("6. DICT_6X6_250 (推奨)")

    try:
        choice = input("選択 (1-6): ").strip()
        if choice not in dict_options:
            choice = "6"  # デフォルト

        dict_type = dict_options[choice]

        # 生成モードの選択
        print("\n生成モードを選択してください:")
        print("1. 単一マーカー")
        print("2. 複数マーカー")
        print("3. マーカーボード")

        mode = input("選択 (1-3): ").strip()

        if mode == "1":
            # 単一マーカー生成
            marker_id = int(input("マーカーID (0-249): "))
            marker_size = int(input("マーカーサイズ (ピクセル) [200]: ") or "200")

            save_path = f"aruco_marker_{marker_id}.png"
            generate_aruco_marker(marker_id, marker_size, dict_type, save_path)

        elif mode == "2":
            # 複数マーカー生成
            ids_input = input("マーカーID (カンマ区切り、例: 0,1,2,3): ")
            marker_ids = [int(id.strip()) for id in ids_input.split(",")]
            marker_size = int(input("マーカーサイズ (ピクセル) [200]: ") or "200")

            generate_multiple_markers(marker_ids, marker_size, dict_type)

        elif mode == "3":
            # マーカーボード生成
            ids_input = input("マーカーID (カンマ区切り、例: 0,1,2,3,4,5): ")
            marker_ids = [int(id.strip()) for id in ids_input.split(",")]
            markers_per_row = int(input("1行あたりのマーカー数 [3]: ") or "3")
            marker_size = int(input("マーカーサイズ (ピクセル) [150]: ") or "150")

            create_marker_board(
                marker_ids, markers_per_row, marker_size, dict_type=dict_type
            )

        print("\nマーカーの生成が完了しました！")

    except ValueError as e:
        print(f"エラー: 無効な入力です - {e}")
    except KeyboardInterrupt:
        print("\n処理を中断しました。")


if __name__ == "__main__":
    main()
