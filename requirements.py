# import pip
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description='install requirements')
parser.add_argument('--cuda', default=None, type=str)
args = parser.parse_args()

pkgs=f'''
cython
scikit-image
loguru
matplotlib
tabulate
tqdm
pywin32
PyAutoGUI
opencv_python
keyboard
Pillow
pymouse
numpy==1.19.5
torch==1.7.0+{"cpu" if args.cuda is None else "cu" + args.cuda} -f https://download.pytorch.org/whl/torch_stable.html
torchvision==0.8.1+{"cpu" if args.cuda is None else "cu" + args.cuda} --no-deps -f https://download.pytorch.org/whl/torch_stable.html
thop --no-deps
git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
'''

for line in pkgs.split('\n'):
    if len(line)>0:
        pip.main(['install', '--default-timeout=100', *line.split()])

