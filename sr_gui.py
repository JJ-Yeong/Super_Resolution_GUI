import os
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

import debugpy
import ffmpeg
import cv2
import glob
import mimetypes
import numpy as np
import os
import os.path as osp
import shutil
import torch
from torchvision.transforms import ToPILImage, FiveCrop, Pad, Resize, InterpolationMode
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from easydict import EasyDict

from realesrgan import RealESRGANer

# 메인화면 추가
# 추론 끝나고 complete 누르면 메인화면 나오게끔


MAIN_PATH = "qtd/mainwindow.ui"
PBAR_PATH = "qtd/progressbar.ui"

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def get_ui(ui_path):
    form = resource_path(ui_path)
    return uic.loadUiType(form)[0]

form_class = get_ui(MAIN_PATH)
form_class_pbar = get_ui(PBAR_PATH)


class ProgressDialog(QDialog, form_class_pbar):
    stop_signal = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setModal(True)
        self.pbar: QProgressBar
        self.label_file_name: QLabel
        self.btn_cancel: QPushButton
        self.btn_open_savedir: QPushButton

        self.stop = False
        self.filename = ""
        self.dot = 1

    def progress(self, val):
        self.text1 = f"{self.filename}\n"
        self.text2 = f"Inference..{'.' * self.dot}"
        self.label_file_name.setText(self.text1 + self.text2)
        self.dot = self.dot + 1 if self.dot < 3 else 1
        self.pbar.setValue(val)

    def fnc_stop_progress(self):
        self.stop = True
        self.stop_signal.emit(0)
        self.close()

    def fnc_open_savedir(self):
        os.startfile(self.output_path)

    def closeEvent(self, e):
        # sys.exit()
        self.stop = True
        self.stop_signal.emit(0)
        self.close()


class MainWindow(QMainWindow, QWidget, form_class):
    # sr_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.label_input_file: QLabel 
        self.label_output_dir: QLabel
        self.comboBox_upfactor: QComboBox
        # self.comboBox_tilesize: QComboBox
        self.check_tile: QCheckBox

        # self.comboBox_tilesize.setEnabled(False)
        # self.input_file_path = ""
        # self.output_dir_path = ""
        self.pdialog = ProgressDialog()
        self.pdialog.stop_signal.connect(self.fnc_add_progress)
        self.input_file_path = "C:/Users/ZZY/Desktop/0_cityeyelab/code/Super_Resolution_GUI/inputs/video/project7_resize0.25.mp4"
        self.output_dir_path = "C:/Users/ZZY/Desktop/0_cityeyelab/code/Super_Resolution_GUI/outputs"

    def fnc_select_input(self):
        self.input_file_path, _ = QFileDialog.getOpenFileName(self, 'Please select input file')
        self.label_input_file.setText(self.input_file_path)
        self.pdialog.filename = osp.basename(self.input_file_path)
    
    def fnc_select_output(self):
        self.output_dir_path = QFileDialog.getExistingDirectory(self, "Please select output directory")
        self.label_output_dir.setText(self.output_dir_path)

    # def fnc_tile_checkbox(self):
    #     if self.check_tile.isChecked():
    #         self.comboBox_tilesize.setEnabled(True)
    #     else:
    #         self.comboBox_tilesize.setEnabled(False)

    def fnc_start_upscale(self):
        self.upscale_factor = int(self.comboBox_upfactor.currentText())
        # self.tile_size = int(self.comboBox_tilesize.currentText()) if self.comboBox_tilesize.isEnabled() else 0
        self.tile_size = 0
        self.pdialog.show()

        self.th = Inference(
            input_path = self.input_file_path,
            output_path = self.output_dir_path,
            upscale_factor = self.upscale_factor,
            tile_size = self.tile_size
        )

        #TODO max_length
        # self.max_length = 5000
        self.max_length = self.th.pbar_max
        self.th.progress_signal.connect(self.fnc_add_progress)

        # self.sr_signal.connect(self.th.add_sec)
        self.th.start()
        self.th.working = True
    
    def fnc_add_progress(self, pbar):
        progress_step = int(pbar / self.max_length * 100)
        # print(progress_step)
        self.pdialog.progress(progress_step)
        if progress_step >= 100:
            self.close_pdialog(f"Complete!!\n>> {self.th.output_name1}\n>> {self.th.output_name2}")
            self.pdialog.output_path = resource_path(self.output_dir_path)
            self.pdialog.btn_open_savedir.setEnabled(True)
            # self.pdialog.btn_cancel.setEnabled(True)
            self.close_cv2()
        elif pbar == 0:
            self.close_pdialog("Aborted!!")

    def close_pdialog(self, pdialog_text: str):
        self.th.stop()
        self.pdialog.text2 = pdialog_text
        self.pdialog.label_file_name.setText(self.pdialog.text1 + self.pdialog.text2)
        if pdialog_text.startswith("Complete"):
            self.pdialog.btn_cancel.setText("Complete")

    def close_cv2(self):
        if self.th.cap_org.isOpened():
            self.th.video_writer.release()
            self.th.cap_org.release()
        self.th.reader.close()
        self.th.writer.close()


class Inference(QThread):
    progress_signal = pyqtSignal(int)
    def __init__(self, input_path, output_path, upscale_factor, tile_size):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path if output_path else "results"
        self.upscale_factor = upscale_factor
        self.tile_size = tile_size

        self.working = True

        INPUT_PATH      = resource_path(self.input_path)
        OUTPUT_PATH     = resource_path(self.output_path)
        UPSCALE_FACTOR  = self.upscale_factor
        TILE_SIZE       = self.tile_size
        # OUTPUT_FPS      = gui_args["output_fps"]

        # INPUT_PATH = "inputs/video/project7_resize0.25.mp4"
        # UPSCALE_FACTOR = 4
        # TILE_SIZE = 0
        OUTPUT_FPS = None

        # gui에 argparser는 필요 없으므로 args의 default값을 딕셔너리로 재구성
        args = EasyDict({
            "model_name": f"RealESRGAN_x{UPSCALE_FACTOR}plus.pth",   # Model name
            "input": INPUT_PATH,                # Input video, image or folder
            "output": OUTPUT_PATH,                # Output folder
            "outscale": UPSCALE_FACTOR,         # The final upsampling scale of the image
            "suffix": f"out_x{UPSCALE_FACTOR}", # Suffix of the restored video
            "tile": TILE_SIZE,                  # Tile size, 0 for no tile during testing (minimum: 32)
            "tile_pad": 10,                     # Tile padding
            "pre_pad": 0,                       # Pre padding size at each border
            "fp32": False,                      # Use fp32 precision during inference. Default: fp16 (half precision).
            "fps": OUTPUT_FPS,                  # FPS of the output video
            "ffmpeg_bin": "ffmpeg",             # The path to ffmpeg
            "num_process_per_gpu": 1,
            "alpha_upsampler": "realesrgan",    # The upsampler for the alpha channels. Options: realesrgan | bicubic
            "ext": "auto"                       # Image extension. Options: auto | jpg | png, auto means using the same extension as inputs
        })

        args.input = args.input.rstrip('/').rstrip('\\')
        os.makedirs(args.output, exist_ok=True)

        if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
            is_video = True
        else:
            is_video = False

        if is_video and args.input.endswith('.flv'):
            mp4_path = args.input.replace('.flv', '.mp4')
            os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
            args.input = mp4_path

        ext = ".mp4"
        args.video_name = osp.splitext(os.path.basename(args.input))[0]
        video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}{ext}')
        unique = 1
        while osp.exists(video_save_path):
            video_save_path = osp.join(args.output, f"{args.video_name}_{args.suffix}_{unique}{ext}")
            unique += 1

        num_gpus = torch.cuda.device_count()
        num_process = num_gpus * args.num_process_per_gpu
        # if num_process == 1:

        self.device = None
        total_workers = 1
        worker_idx = 0
        # ---------------------- determine models according to model names ---------------------- #
        args.model_name = args.model_name.split('.pth')[0]
        if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        # ---------------------- determine model paths ---------------------- #
        model_path = resource_path(osp.join('weights', args.model_name + '.pth'))
        if not osp.isfile(model_path):
            ROOT_DIR = osp.dirname(osp.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=osp.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        # use dni to control the denoise strength
        dni_weight = None

        # restorer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=not args.fp32,
            device=self.device,
        )


        reader = Reader(args, total_workers, worker_idx)
        audio = reader.get_audio()
        height, width = reader.get_resolution()
        fps = reader.get_fps()
        writer = Writer(args, audio, height, width, video_save_path, fps)


        cap_org = cv2.VideoCapture(args.input)
        fps_org = cap_org.get(cv2.CAP_PROP_FPS)
        frame_count_org = cap_org.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_width_org = int(cap_org.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_org = int(cap_org.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size_org = (frame_width_org, frame_height_org)

        final_width = int(frame_width_org * args.outscale * 2 + 10)
        final_height = int(frame_height_org) * args.outscale + 10 + int(final_width / int(10 * int(int(frame_width_org * args.outscale) // 5 + 1)) * int(int(frame_width_org * args.outscale) // 5 - 9))
        final_video_size = (final_width, final_height)

        video_save_name, ext = osp.splitext(video_save_path)
        compare_save_path = video_save_name + f"_compare" + ext 
        video_writer = cv2.VideoWriter(compare_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_org, final_video_size)


        self.output_name1 = osp.basename(video_save_path)
        self.output_name2 = osp.basename(compare_save_path)

        self.upsampler = upsampler
        self.reader = reader
        self.writer = writer
        self.cap_org = cap_org
        self.video_writer = video_writer
        self.outscale = args.outscale
        self.pbar_max = len(reader)
        self.pbar = 0

        # self.step = 1

    def run(self):
        while self.working:
            hasFrame, frame_org = self.cap_org.read()
            img = self.reader.get_frame()
            if img is None:
                break
            if not hasFrame:
                # print("영상1 끝!")
                break

            try:
                output, _ = self.upsampler.enhance(img, outscale=self.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                self.writer.write_frame(output)

            # SR 이미지
            frame_sr = ToPILImage()(np.uint8(output))
            crop_sr = FiveCrop(size=frame_sr.width // 5 - 9)(frame_sr)
            crop_sr = [np.asarray(Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr]
            frame_sr = Pad(padding=(5, 0, 0, 5))(frame_sr)

            # 원본 이미지를 SR 이미지 사이즈만큼 업스케일
            frame_org_upscaled = Resize(size=(output.shape[0], output.shape[1]), interpolation=InterpolationMode.BICUBIC)(ToPILImage()(np.uint8(frame_org)))
            crop_org_upscaled = FiveCrop(size=frame_org_upscaled.width // 5 - 9)(frame_org_upscaled)
            crop_org_upscaled = [np.asarray(Pad(padding=(0, 5, 10, 0))(img)) for img in crop_org_upscaled]
            frame_org_upscaled = Pad(padding=(0, 0, 5, 5))(frame_org_upscaled)

            # concatenate all the pictures to one single picture
            top_image = np.concatenate((np.asarray(frame_org_upscaled), np.asarray(frame_sr)), axis=1)
            bottom_image = np.concatenate(crop_org_upscaled + crop_sr, axis=1)
            bottom_image = np.asarray(
                Resize(size=(int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0]), top_image.shape[1]))(ToPILImage()(bottom_image))
            )
            final_image = np.concatenate((top_image, bottom_image))

            self.video_writer.write(final_image)

            torch.cuda.synchronize(self.device)
            self.pbar += 1
            self.progress_signal.emit(self.pbar)
    

    def stop(self):
        self.working = False
        self.quit()
        self.wait(3000)
        # if self.cap_org.isOpened():
        #     self.video_writer.release()
        #     self.cap_org.release()

        # self.reader.close()
        # self.writer.close()

    # @pyqtSlot()
    # def add_sec(self):
    #     print("add_sec....")
    #     self.step += 1

#######################################################################################################################


def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret


class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = args.input
            ffmpeg1 = ffmpeg.input(video_path)
            ffmpeg2 = ffmpeg1.output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
            ffmpeg3 = ffmpeg2.run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            self.stream_reader = (ffmpeg3)
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
