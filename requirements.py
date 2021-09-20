import pip
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
PyUserInput
tqdm
pywin32
PyAutoGUI
opencv_python
keyboard
Pillow
pymouse
numpy==1.19.5
torch==1.7.0+{"cpu" if args.cuda is None else args.cuda} -f https://download.pytorch.org/whl/torch_stable.html
--no-deps torchvision==0.8.1+{"cpu" if args.cuda is None else args.cuda} -f https://download.pytorch.org/whl/torch_stable.html
thop
git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
'''

for line in pkgs.split('\n'):
    pip.main(['install', *line.split()])