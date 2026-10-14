"""このモジュールは、泡動画解析用のクラス Bubblytics を提供します。

Examples:
    Bubblytics クラスの使用例を以下に示します。

    from bubblytics import Bubblytics
    bt = Bubblytics('input_file.avi')
    bt.set_parameters(...)
    bt.set_detection_parameters(...)
    bt.execute(f'output/{bt.get_filename()}')
"""
import os
import csv
from typing import Optional
import cv2
from numpy.typing import NDArray
import numpy as np

class Bubblytics:
    """泡動画像解析用のクラスです。

    Methods:
        set_parameters: パラメーターを設定するメソッドです。
        execute: 動画像解析を実行するメソッドです。

    Functions:
        get_filename: 入力ファイル名を取得する関数です。
    """
    def __init__(self, input_filepath: str):
        """
        クラスの初期化を行い、動画ファイルを読み込みます。

        Parameters:
            input_filepath (str): 読み込む動画のファイルパスです。
        """
        self.input_filepath = input_filepath
        self.video_capture = cv2.VideoCapture(input_filepath)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.video_capture.isOpened():
            print(f"Failed to open {input_filepath}.")

        # パラメーターのデフォルト値を設定します。
        self.background_frames = 5
        self.minimum_mesh_area = 2300
        self.mesh_binarize_threshold = -1
        self.minimum_white_ratio = 0.2
        self.mesh_expand_width = 5
        self.background_diff_threshold = 40
        self.maximum_hole_area = 500
        self.frame_skip = 0
        self.show_progress = True

        # 円検出用パラメーターのデフォルト値を設定します。
        self.gaussian_kernel_half = 4
        self.gaussian_sigma = 3.0
        self.hough_dp = 1.2
        self.hough_min_dist = 10.0
        self.hough_param1 = 40.0
        self.hough_param2 = 20.0
        self.hough_min_radius = 5
        self.hough_max_radius = 50
        self.area_ratio = 0.9

    def __del__(self) -> None:
        """デストラクタでリソースを解放します。"""
        if self.video_capture:
            self.video_capture.release()
        if self.video_writer:
            self.video_writer.release()


    def set_parameters(
            self,
            background_frames : int = 5,
            minimum_mesh_area : int = 2300,
            mesh_binarize_threshold : int = -1,
            minimum_white_ratio : float = 0.5,
            mesh_expand_width : int = 5,
            background_diff_threshold : int = 40,
            maximum_hole_area : int = 500,
            frame_skip : int = 0,
            show_progress : bool = False) -> None:
        """動画像解析用のパラメーターを設定するメソッドです。

        Parameters:
            background_frames (int): 背景差分法に用いられる背景画像の作成に用いられるフレーム数です。
            minimum_mesh_area (int): メッシュ内側 (白領域) の最小面積を設定します。メッシュ構造抽出時、誤検出された白領域を塗りつぶすために用いられます。
            mesh_binarize_threshold (int): 背景画像を二値化してメッシュ領域を取得するための閾値です。負値の場合は自動算出されます。
            minimum_white_ratio (float): メッシュの白領域が全体の面積に占める割合の最小値です。閾値の自動算出時に用いられます。
            mesh_expand_width (int): メッシュ領域を広げるピクセル数を設定します。メッシュ領域と重複する泡は検出されません。
            background_diff_threshold (int): 背景差分に対する閾値を設定します。閾値を超えた領域は泡候補領域とみなされます。
            maximum_hole_area (int): 泡内側の黒領域の塗りつぶしに用いられる閾値です。この値未満の面積の領域が塗りつぶされます。
            frame_skip (int): 動画解析時にスキップするフレーム数を指定します。デフォルトは 0 で、すべてのフレームを対象に解析が行われます。
            show_progress (bool): 処理中の進捗を表示するかどうかを設定します。
        """
        self.background_frames = background_frames
        self.minimum_mesh_area = minimum_mesh_area
        self.mesh_binarize_threshold = mesh_binarize_threshold
        self.minimum_white_ratio = minimum_white_ratio
        self.mesh_expand_width = mesh_expand_width
        self.background_diff_threshold = background_diff_threshold
        self.maximum_hole_area = maximum_hole_area
        self.frame_skip = frame_skip
        self.show_progress = show_progress


    def set_detection_parameters(
            self,
            gaussian_kernel_half : int = 4,
            gaussian_sigma : float = 3.0,
            hough_dp : float = 1.2,
            hough_min_dist : float = 10.0,
            hough_param1 : float = 40.0,
            hough_param2 : float = 20.0,
            hough_min_radius : int = 5,
            hough_max_radius : int = 50,
            area_ratio : float = 0.9) -> None:
        """円検出用のパラメーターを設定するメソッドです。

        ハフ変換による円検出については下記の OpenCV 公式ドキュメントをご参照ください。
        https://docs.opencv.org/4.9.0/d4/d70/tutorial_hough_circle.html

        Parameters:
            gaussian_kernel_half (int): ガウシアン フィルターのカーネル サイズの半分の大きさを設定します。
            gaussian_sigma (float): ガウシアン フィルターのシグマの値を設定します。
            hough_dp (float): ハフ変換に用いられる dp の値を設定します。
            hough_min_dist (float): ハフ変換に用いられる円の中心間の最小距離を設定します。
            hough_param1 (float): ハフ変換に用いられる閾値 param1 の値を設定します。
            hough_param2 (float): ハフ変換に用いられる閾値 param2 の値を設定します。
            hough_min_radius (int): ハフ変換に用いられる最小半径を設定します。
            hough_max_radius (int): ハフ変換に用いられる最大半径を設定します。
            area_ratio (float): 検出結果のフィルタリングに用いられる面積比を設定します。
        """
        self.gaussian_kernel_half = gaussian_kernel_half
        self.gaussian_sigma = gaussian_sigma
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.hough_min_radius = hough_min_radius
        self.hough_max_radius = hough_max_radius
        self.area_ratio = area_ratio

    def execute(self,
                output_dir : str,
                background_image_path : Optional[str] = None) -> None:
        """動画像処理を実行するメソッドです。

        Parameters:
            output_dir (str): 出力ディレクトリのパスです。
            background_image (NDArray[np.uint8]): 背景画像です。None の場合は動画先頭のフレームから自動生成されます。
        """
        # 出力ディレクトリを作成します。
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 出力ファイル パスを作成します。
        output_movie_path = os.path.join(output_dir, f'{self.get_filename()}_out.mp4')

        # 入力動画の設定を取得し、出力動画に反映させます。
        if hasattr(cv2, 'VideoWriter_fourcc'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise RuntimeError('cv2.VideoWriter_fourcc を参照できません。')
        fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 動画出力用のオブジェクトを初期化します。
        self.video_writer = cv2.VideoWriter(output_movie_path, fourcc, fps, (2 * width, height))
        if not self.video_writer:
            raise RuntimeError("動画出力用オブジェクトの初期化に失敗しました。")

        # フレーム数を保存しておく変数です。
        current_frame = 0

        # 背景差分法に用いる背景画像を生成します。
        if background_image_path is None:
            background_image = self.__extract_background_image(
                self.video_capture, self.background_frames)
            current_frame += self.background_frames
        else:
            background_image = self.__imread(background_image_path)
        background_image = self.__convert_to_grayscale(background_image)

        # 背景画像を出力します。
        cv2.imwrite(os.path.join(output_dir, 'background.png'), background_image)

        # 閾値の自動算出が設定されているかどうか確認します。
        auto_threshold_mode = self.mesh_binarize_threshold < 0

        # 指定された閾値を用いて背景画像からメッシュを抽出します。
        mesh_mask = self.__extract_mesh_structure(
            background_image,
            threshold=self.mesh_binarize_threshold,
            area_threshold=self.minimum_mesh_area)

        # メッシュ領域の面積を算出します。
        self.whole_area : int = mesh_mask.size
        self.mesh_black_area : int = np.sum(mesh_mask == 0)
        self.mesh_white_area : int = self.whole_area - self.mesh_black_area

        if auto_threshold_mode:
            # 閾値を自動算出して背景画像からメッシュを抽出します。
            # 閾値は白色領域の割合が一定以上になるように設定されます。
            while self.mesh_white_area / self.whole_area  < self.minimum_white_ratio:
                # 背景画像からメッシュを抽出します。
                mesh_mask = self.__extract_mesh_structure(
                    background_image,
                    threshold=self.mesh_binarize_threshold - 1,
                    area_threshold=self.minimum_mesh_area)

                # メッシュ領域の面積を更新します。
                self.whole_area = mesh_mask.size
                self.mesh_black_area = np.sum(mesh_mask == 0)
                self.mesh_white_area = self.whole_area - self.mesh_black_area

        # CSV 形式で出力します。
        with open(
            os.path.join(output_dir, 'mesh_data.csv'),
            'w', newline='', encoding='utf-8') as file:
            # CSV 出力オブジェクトを作成します。
            csv_writer = csv.writer(file)
            # ヘッダを出力します。
            csv_writer.writerow(['whole_area (px)', 'black_area (px)', 'white_area (px)', 'binarize_threshold'])
            # 値を出力します。
            csv_writer.writerow([self.whole_area, self.mesh_black_area, self.mesh_white_area, self.mesh_binarize_threshold])

        # メッシュ画像を出力します。
        cv2.imwrite(os.path.join(output_dir, 'mesh.png'), mesh_mask)

        # メッシュ領域を拡大しておきます。
        mesh_mask = self.__expand_black_regions(mesh_mask, self.mesh_expand_width)

        # CSV ファイル出力を開始します。
        with open(
            os.path.join(output_dir, f'{self.get_filename()}.csv'),
            'w', newline='', encoding='utf-8') as file:

            # CSV 出力オブジェクトを作成します。
            csv_writer = csv.writer(file)

            # ヘッダを出力します。
            csv_writer.writerow(['frame', 'center_x (px)', 'center_y (px)', 'radius (px)'])

            # すべてのフレームを処理するループです。
            while True:
                # 1 フレーム読み込みます
                ret, rgb_frame = self.video_capture.read()
                if not ret:
                    break

                # 進捗を出力します。
                if self.show_progress and current_frame % 20 == 0:
                    print(f'Progress: {current_frame}/{self.total_frames} frames.')

                # 処理対象のフレームを判定します。
                if current_frame % (self.frame_skip + 1) == 0:

                    # フレームをグレースケール化します。
                    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

                    # 背景との差分を計算します。
                    diff = cv2.absdiff(background_image, gray_frame)

                    # 差分を二値化します。
                    _, thresh = cv2.threshold(
                        diff, self.background_diff_threshold, 255, cv2.THRESH_BINARY)

                    # 穴埋め処理を行います。
                    thresh = self.__fill_small_black_areas(thresh.astype(np.uint8), self.maximum_hole_area)

                    # 検出処理の高速化のために二値画像とメッシュとの AND をとります。
                    thresh = cv2.bitwise_and(thresh, mesh_mask).astype(np.uint8)

                    # 検出処理を行います。
                    circles = self.__detect_circles(thresh, mesh_mask)

                    # 検出結果を出力します。
                    output_image = rgb_frame.copy()
                    for circle in circles:
                        # CSV 出力します。
                        csv_writer.writerow([current_frame, *circle])
                        # 検出された円を描画します。
                        round_circle = np.round(circle).astype(np.uint32)
                        cv2.circle(output_image,
                                (round_circle[0], round_circle[1]),
                                round_circle[2], (255, 0, 255), 2)

                    # 出力フレームを作成し、出力動画に書き込みます。
                    concatenated_frame = cv2.hconcat([rgb_frame, output_image])
                    self.video_writer.write(concatenated_frame)

                # フレーム数をインクリメントします。
                current_frame += 1


    def get_filename(self, with_extension : bool = False) -> str:
        """入力ファイル名を取得するメソッドです。

        Parameters:
            with_extension (bool): 拡張子つきのファイル名を取得するかどうか設定します。

        Returns:
            str: 入力ファイル名を返します。
        """
        filename_with_ext = os.path.basename(self.input_filepath)
        if with_extension:
            return filename_with_ext
        filename, _ = os.path.splitext(filename_with_ext)
        return filename

    def __imread(self, filepath: str) -> NDArray[np.uint8]:
        """マルチバイト文字をパスに含む画像ファイルを読み込む関数です。

        Parameters:
            filepath (str): 読み込む画像ファイルのパスを渡します。

        Returns:
            NDArray[np.uint8]: 読み込まれた画像データを返します。
        """
        # ファイルをバイトモードで読み込みます。
        with open(filepath, 'rb') as file:
            data = file.read()

        # バイトデータから画像データをデコードします。
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.uint8)

    def __convert_to_grayscale(self,image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """引数として渡された画像をグレースケール化します。

        この関数は、入力画像がカラー (RGB または RGBA) の場合にグレースケールに変換します。
        画像がすでにグレースケールであるか、2 次元の場合はそのまま返します。

        Parameters:
            image (NDArray[np.uint8]): 変換したい画像データ。np.ndarray 形式である必要があります。

        Returns:
            NDArray[np.uint8]: グレースケールに変換された画像、または元の画像データ。
        """
        # 画像が 2 次元 (グレースケール) か 3 次元 (カラー) かを確認します。
        if image.ndim == 3:
            # チャンネル数を確認します。
            channels = image.shape[2]
            # RGB または RGBA (4 チャンネル) の場合、グレースケールに変換します。
            if channels in (3, 4):
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # すでにグレースケールの場合、または 2 次元の場合はそのまま返します。
        return image


    def __extract_background_image(
            self,
            video_capture : cv2.VideoCapture,
            frame_count : int,
            gray_scale : bool = True) -> NDArray[np.uint8]:
        """ビデオから指定フレーム数の画像を平均化して背景画像を算出します。

        フレームの読み込みに失敗した場合には RuntimeError を発生させます。

        Parameters:
            video_capture (cv2.VideoCapture): cv2.VideoCapture オブジェクトを渡します。
            frame_count (int): 平均化に使用するフレーム数を指定します。
            gray_scale (bool): グレースケール化するかどうか指定します。

        Returns:
            NDArray[np.uint8]: 平均化された背景画像が返ります。
        """
        # 平均を計算するための変数を初期化します。
        average_frame = None
        for _ in range(frame_count):
            # フレームを読み込みます。
            ret, frame = video_capture.read()
            if not ret:
                raise RuntimeError("背景画像生成用フレームの読み込みに失敗しました。")
            # グレースケール化します。
            if gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # フレームを浮動小数点型に変換し、平均を計算します。
            if average_frame is None:
                average_frame = frame.astype(np.float32)
            else:
                average_frame += frame

        if average_frame is None:
            raise RuntimeError("背景画像の平均化に失敗しました。")

        # 平均化されたフレームを計算します。
        average_frame /= frame_count
        # 結果を uint8 に変換して返します。
        return average_frame.astype(np.uint8)


    def __remove_small_areas(
            self,
            binary_image: NDArray[np.uint8],
            area_threshold: int) -> NDArray[np.uint8]:
        """二値画像中の閾値以下の面積を持つ白領域を黒に変更します。

        Parameters:
            binary_image (NDArray[np.uint8]): 入力二値画像です。白領域が 255、黒領域が 0 です。
            area_threshold (int): 面積の閾値。この値未満の面積を持つ白領域は黒に変更されます。

        Returns:
            NDArray[np.uint8]: 処理後の画像を返します。
        """
        # 連結成分を見つけます。
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 各連結成分に対して面積を計算し、閾値以下のものを黒で塗りつぶします。
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_threshold:
                cv2.drawContours(binary_image, [contour], -1, [0], thickness=cv2.FILLED)

        return binary_image


    def __extract_mesh_structure(
            self,
            image : NDArray[np.uint8],
            threshold : int = -1,
            area_threshold : int = 0) -> NDArray[np.uint8]:
        """画像からメッシュ構造を二値画像として抽出します。

        Parameters:
            image (NDArray[np.uint8]): グレースケール画像を渡します。
            threshold (int): 二値化の閾値を渡します。負の値が渡された場合は大津の二値化が行われます。
            area_threshold (int): メッシュ領域判定用の面積の閾値を渡します。渡された値未満の面積の白領域はメッシュ領域とみなされます。

        Returns:
            NDArray[np.uint8]: メッシュ構造が黒色の二値画像を返します。
        """
        # 二値化します。
        if threshold < 0:
            auto_threshold, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.mesh_binarize_threshold = int(auto_threshold)
        else:
            _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            self.mesh_binarize_threshold = threshold

        if area_threshold > 0:
            # 閾値未満の面積の領域を除去して返します。
            return self.__remove_small_areas(thresh.astype(np.uint8), area_threshold)

        return thresh.astype(np.uint8)


    def __expand_black_regions(
            self,
            binary_image: NDArray[np.uint8],
            expand_width: int) -> NDArray[np.uint8]:
        """二値画像の黒領域を 1 ピクセル分拡大します。

        Parameters:
            binary_image (NDArray[np.uint8]): 入力二値画像です。白領域が 255、黒領域が 0 です。
            expand_width (int): 黒領域を拡大するピクセル数です。

        Returns:
            NDArray[np.uint8]: 処理後の画像を返します。
        """
        # エロージョンに使用するカーネルを定義します。
        kernel_size = expand_width * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # エロージョン (収縮) を適用して黒領域を拡大します。
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)

        return eroded_image.astype(np.uint8)


    def __fill_small_black_areas(
            self,
            binary_image: NDArray[np.uint8],
            area_threshold: int = -1) -> NDArray[np.uint8]:
        """二値画像において、白領域に囲まれた閾値未満の面積の黒領域を白に変更します。

        Parameters:
            binary_image (NDArray[np.uint8]): 入力二値画像 (白領域が 255、黒領域が 0) 
            area_threshold (int): 面積の閾値です。この値未満の黒領域を白に変更します。

        Returns:
            NDArray[np.uint8]: 処理後の画像を返します。
        """
        # 画像のコピーを作成します。
        processed_image = binary_image.copy()

        # 黒領域を検出します。findContours は白領域を検出するため、画像の色を反転させます。
        contours, _ = cv2.findContours(
            255 - processed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 各領域の面積を計算します。
            area = cv2.contourArea(contour)
            if area < area_threshold:
                # 面積が閾値未満の場合、領域を白色で塗りつぶします。
                cv2.drawContours(processed_image, [contour], -1, [255], thickness=cv2.FILLED)

        return processed_image


    def __detect_circles(
            self,
            binary_image: NDArray[np.uint8],
            mask_image: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
        """二値画像に対して円検出を行います。

        Parameters:
            binary_image (NDArray[np.uint8]): 泡領域が抽出された二値画像を渡します。
            mask_image (NDArray[np.uint8]): メッシュ領域が抽出された二値画像を渡します。

        Returns:
            list: 検出された円の中心と半径のリストを返します。
        """
        # 円検出のための前処理としてガウシアン フィルターを適用します。
        kernel_size = self.gaussian_kernel_half * 2 + 1
        blurred_image = cv2.GaussianBlur(
            binary_image, (kernel_size, kernel_size), self.gaussian_sigma)

        # ハフ変換を使用して円検出を行います。
        circles = cv2.HoughCircles(
            blurred_image, method=cv2.HOUGH_GRADIENT,
            dp=self.hough_dp, minDist=self.hough_min_dist,
            param1=self.hough_param1, param2=self.hough_param2,
            minRadius=self.hough_min_radius, maxRadius=self.hough_max_radius)

        # 検出された円のうち、妥当な結果を選別します。
        filtered_circles = []
        if circles is not None:
            circles = circles[0, :]
            if circles is not None:
                for circle in circles:
                    # 検出された円の中心と半径を取得します。
                    round_circle = np.round(circle).astype(np.uint32)
                    center = (round_circle[0], round_circle[1])
                    radius = round_circle[2]
                    circle_mask = np.zeros_like(binary_image)
                    cv2.circle(circle_mask, center, radius, [255], -1)
                    # 検出円領域とメッシュ領域が重複していない円のみ選別します。
                    circle_masked = cv2.bitwise_and(circle_mask, mask_image)
                    if np.array_equal(circle_mask, circle_masked):
                        masked_circle = cv2.bitwise_and(circle_mask, binary_image)
                        # 検出円領域内に二値画像の白領域が一定の面積比以上存在している円のみ選別します。
                        area = np.sum(masked_circle == 255)
                        if area > radius * radius * np.pi * self.area_ratio:
                            filtered_circles.append(circle)
        return filtered_circles
