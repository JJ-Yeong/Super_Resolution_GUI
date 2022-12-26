import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from torchvision.transforms import ToPILImage, FiveCrop, Pad, Resize, InterpolationMode
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm
from easydict import EasyDict


from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# try:
import ffmpeg
# except ImportError:
#     import pip
#     pip.main(['install', '--user', 'ffmpeg-python'])
#     import ffmpeg


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


def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path


class Reader:

    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
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


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

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
        device=device,
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


    # pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    pbar_max = len(reader)
    pbar = 0
    while True:
        hasFrame, frame_org = cap_org.read()
        img = reader.get_frame()
        if img is None:
            break
        if not hasFrame:
            print("영상1 끝!")
            break

        try:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            writer.write_frame(output)

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

        video_writer.write(final_image)

        torch.cuda.synchronize(device)
        pbar += 1
        # pbar.update(1)

    if cap_org.isOpened():
        video_writer.release()
        cap_org.release()

    cv2.destroyAllWindows()
    reader.close()
    writer.close()

    print(f"Saved!!! >>> {video_save_path}")
    print(f"Saved!!! >>> {compare_save_path}")


def run(args):
    ext = ".mp4"
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}{ext}')
    unique = 1
    while osp.exists(video_save_path):
        video_save_path = osp.join(args.output, f"{args.video_name}_{args.suffix}_{unique}{ext}")
        unique += 1

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {tmp_frames_folder}/frame%08d.png')
        args.input = tmp_frames_folder

    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        inference_video(args, video_save_path)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/{args.video_name}_vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', f'{args.output}/{args.video_name}_vidlist.txt', '-c',
        'copy', f'{video_save_path}'
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'))
    if osp.exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos')):
        shutil.rmtree(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'))
    os.remove(f'{args.output}/{args.video_name}_vidlist.txt')


def super_resolution(gui_args):

    INPUT_PATH      = gui_args["input_path"]
    UPSCALE_FACTOR  = gui_args["upscale_factor"]
    TILE_SIZE       = gui_args["tile_size"]
    # OUTPUT_FPS      = gui_args["output_fps"]

    # INPUT_PATH = "inputs/video/project7_resize0.25.mp4"
    # UPSCALE_FACTOR = 4
    # TILE_SIZE = 0
    OUTPUT_FPS = None

    # gui에 argparser는 필요 없으므로 args의 default값을 딕셔너리로 재구성
    args = EasyDict({
        "model_name": f"RealESRGAN_x{UPSCALE_FACTOR}plus.pth",   # Model name
        "input": INPUT_PATH,                # Input video, image or folder
        "output": "results",                # Output folder
        "outscale": UPSCALE_FACTOR,         # The final upsampling scale of the image
        "suffix": f"out_x{UPSCALE_FACTOR}", # Suffix of the restored video
        "tile": TILE_SIZE,                  # Tile size, 0 for no tile during testing (minimum: 32)
        "tile_pad": 10,                     # Tile padding
        "pre_pad": 0,                       # Pre padding size at each border
        "fp32": False,                      # Use fp32 precision during inference. Default: fp16 (half precision).
        "fps": OUTPUT_FPS,                  # FPS of the output video
        "ffmpeg_bin": "ffmpeg",             # The path to ffmpeg
        "extract_frame_first": False,
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

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    run(args)

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        shutil.rmtree(tmp_frames_folder)


if __name__ == '__main__':
    super_resolution()
