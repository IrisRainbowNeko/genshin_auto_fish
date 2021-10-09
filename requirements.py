# import pip
import subprocess
import sys
import argparse

# use python type hints to make code more readable
from typing import List, Optional


def pip_install(proxy: Optional[str], args: List[str]) -> None:
    if proxy is None:
        # pip.main(["install", f"--proxy={proxy}", *args])
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *args],
            capture_output=False,
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"--proxy={proxy}", *args],
            capture_output=False,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="install requirements")
    parser.add_argument("--cuda", default=None, type=str)
    parser.add_argument(
        "--proxy",
        default=None,
        type=str,
        help="specify http proxy, [http://127.0.0.1:1080]",
    )
    args = parser.parse_args()

    pkgs = f"""
    cython
    scikit-image
    loguru
    matplotlib
    tabulate
    tqdm
    pywin32
    PyAutoGUI
    PyYAML>=5.3.1
    opencv_python
    keyboard
    Pillow
    pymouse
    numpy==1.19.5
    torch==1.7.0+{"cpu" if args.cuda is None else "cu" + args.cuda} -f https://download.pytorch.org/whl/torch_stable.html
    torchvision==0.8.1+{"cpu" if args.cuda is None else "cu" + args.cuda} --no-deps -f https://download.pytorch.org/whl/torch_stable.html
    thop --no-deps
    git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    """

    for line in pkgs.split("\n"):
        # handle multiple space in an empty line
        line = line.strip()

        if len(line) > 0:
            # use pip's internal APIs in this way is deprecated. This will fail in a future version of pip.
            # The most reliable approach, and the one that is fully supported, is to run pip in a subprocess.
            # ref: https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
            # pip.main(['install', *line.split()])

            pip_install(args.proxy, line.split())

    print("\nsuccessfully installed requirements!")


if __name__ == "__main__":
    main()
