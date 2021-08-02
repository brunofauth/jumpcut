# Jumpcut
Make videos play faster at times when they are silent.

This was, originally, a fork of `https://github.com/carykh/jumpcutter`, but I changed so much stuff it didn't even make sense to not have this project be its own thing. Anyway, thanks to CaryKH for the original idea.

# Installation
1. Clone this repo:
    - `git clone https://github.com/brunofauth/jumpcut`
1. Switch to the cloned directory:
    - `cd jumpcut`
1. Create a virtual environment to host this script's dependencies:
    - `python3 -m venv .venv`
1. Activate the virtual environment:
    - bash: `source .venv/bin/activate`
    - fish: `source .venv/bin.activate.fish`
    - powershell: `.venv/Scripts/activate.ps1`
1. Install this script's dependencies:
    - `pip install -r requirements.txt`
1. Install [ffmpeg](https://www.ffmpeg.org/)
    - Arch/Manjaro: `sudo pacman -S ffmpeg`
    - Others: see link above.

# Usage
1. Switch to the cloned directory:
    - `cd jumpcut`
1. Activate the virtual environment which hosts those dependencies:
    - bash: `source .venv/bin/activate`
    - fish: `source .venv/bin.activate.fish`
    - powershell: `.venv/Scripts/activate.ps1`
1. Run the script:
    - `python3.9 jumpcut.py --help`

# Compatibility
* I made this script with python 3.9 in mind, but it probably works with 3.8 as well. I'm not so sure about 3.7 and lower, though.
* I tested this script on Manjaro and on Arch Linux, but it should work on windows too.

# Overview
```
USAGE
  jumpcut.py [OPTIONS...] src dst
  
DESCRIPTION
  Make a video play at a different speed when it's silent.
  
POSITIONAL ARGUMENTS
  src                   path to the source/input video file.
  dst                   Path to the destination/output file.
  
OPTIONS:
  -h, --help                   Show this help message and exit
  -s, --speed SPEED            The speed that silent frames should be played at
  -t, --threshold THRESHOLD    How loud, from 0 (0%) to 1 (100%), a
                                 frame needs to be to not be considered silent
  -m, --margin MARGIN          How many silent frames to keep adjascent to a
                                 non-silent video segment, for context
  -q, --quality QUALITY        Quality of frames to be extracted from input
                                 video (1=highest, 31=lowest, 3=default)
  -d, --directory DIRECTORY    Extract video data to this folder instead of
                                 doing it to a temporary one
  -f, --force                  Do not re-use the previously extracted files
                                 present in '--directory'.
```

