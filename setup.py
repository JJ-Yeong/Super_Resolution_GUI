from cx_Freeze import setup, Executable
import sys
# python setup.py build
# .exe로 빌드 완료한 후에 반드시 이하의 폴더 3개가 .exe파일이 속한 경로에 존재해야함
# inputs/
# results/
# _QTD/


# 1
buildOptions = dict(
    packages = [
        "sys", 
        "os", 
        "glob", 
        "numpy", 
        "PyQt5", 
        "ffmpeg", 
        "cv2", 
        "mimetypes", 
        "torch", 
        "torchvision.transforms", 
        "basicsr.archs.rrdbnet_arch",
        "basicsr.utils.download_util", 
        "easydict",
        "collections",
        "random",
        ],
    excludes = [],
    includes = [
        "realesrgan"
    ],
    include_files = [
        "inputs/",
        "results/",
        "_QTD/"
        ]
    )

base = None
if sys.platform == "win32":
    base = "Win32GUI"

 # 2   
exe = [
    Executable(script="sr_gui.py", base=base)
    ]
    
# exe = [Executable(
#     script="sr_gui.py",
#     icon="icon.ico")]

# 3
setup(
    name= 'sr_gui',
    version = '0.1',
    author = "JJY",
    description = "sr_gui",
    options = dict(build_exe = buildOptions),
    executables = exe
)