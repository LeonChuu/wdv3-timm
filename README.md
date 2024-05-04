# wdv3-timm

small example thing showing how to use `timm` to run the WD Tagger V3 models.

Also Generates a json file formatted for [TagStudio](https://github.com/TagStudioDev/TagStudio)

## How To Use

1. clone the repository and enter the directory:
```sh
git clone https://github.com/neggles/wdv3-timm.git
cd wd3-timm
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.10 -m venv .venv
# Activate it
source .venv/bin/activate
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install PyTorch manually (e.g. if you are not using an nVidia GPU)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install requirements
python -m pip install -r requirements.txt
```

3. Run the example script:
```sh
python wdv3_timm.py  path/to/image.png
```
You can pass a directory as well. The script will create a .TagStudio subdirectory if it doesn't exist, and output over the `ts_library.json` file.

Example output from `python wdv3_timm.py a_picture_of_nene.png`:
```sh
--------
Caption: 1girl, solo, long_hair, purple_eyes, green_hair, shaded_face, upper_body, chibi, own_hands_together, open_mouth, hair_ornament, low-tied_sidelocks, simple_background, triangle_mouth, sidelocks, white_background, blush, bow, kusanagi_nene
--------
Tags: 1girl, solo, long hair, purple eyes, green hair, shaded face, upper body, chibi, own hands together, open mouth, hair ornament, low-tied sidelocks, simple background, triangle mouth, sidelocks, white background, blush, bow, kusanagi nene
--------
Ratings:
  general: 0.973
  sensitive: 0.029
  questionable: 0.000
  explicit: 0.000
  general: 0.973
--------
Character tags (threshold=0.75):
  kusanagi_nene: 0.996
--------
General tags (threshold=0.35):
  1girl: 0.997
  solo: 0.968
  long_hair: 0.905
  purple_eyes: 0.880
  green_hair: 0.836
  shaded_face: 0.679
  upper_body: 0.616
  chibi: 0.610
  own_hands_together: 0.600
  open_mouth: 0.589
  hair_ornament: 0.566
  low-tied_sidelocks: 0.541
  simple_background: 0.523
  triangle_mouth: 0.510
  sidelocks: 0.479
  white_background: 0.431
  blush: 0.419
  bow: 0.357
Done!
```

