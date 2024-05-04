import json
import os
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOTAGGER_TAG = "autotagger"
default_color="Green"
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels

def get_filepaths(dir: Path) -> List[dict]:
    result = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if(filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                result.append({'fullpath': Path(os.path.join(dirpath, filename)), 'dirpath': dirpath, 'filename': filename})
    return result

@dataclass
class ScriptOptions:
    image_file_or_dir: Path = field(positional=True)
    model: str = field(default="vit")
    output_json_file: Path = field(default="./output.json")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)

def add_autotagger_tag(tag_list: list):
    autotagger_dict = {
      "id": tag_list[len(tag_list) -1]['id'] + 1,
      "name": AUTOTAGGER_TAG,
      "aliases": [
        "Autotagged",
        "Autotagger",
        "Autotag"
      ],
      "color": "Black"
    }
    tag_list.append(autotagger_dict)
    return autotagger_dict

def read_json_data(filepath: Path) -> tuple[TextIOWrapper,dict]:
    if(not filepath.exists()):
        json_file = open(filepath,"w+")
        json_data = {
            "ts-version": "9.1.0",
            "tags": [
                {
                "id": 0,
                "name": "Archived",
                "aliases": [
                    "Archive"
                ],
                "color": "Red"
                },
                {
                "id": 1,
                "name": "Favorite",
                "aliases": [
                    "Favorited",
                    "Favorites"
                ],
                "color": "Yellow"
                },
            ],
            "collations": [],
            "fields": [],
            "macros": [],
            "entries": []
        }
    else:
        json_file = open(filepath,"r+")
        json_data=json.load(json_file)
        backup_filepath=str(filepath) + '_backup.json'
        with open(backup_filepath,'w+') as f:
            json.dump(json_data,f)
        print('backup saved to ' + backup_filepath)


    return (json_file, json_data)

def get_autotagged_tag(tag_list: list[dict], tag: str, autotagged_id: int):
    tagged_list = [x for x in tag_list if (x["name"] == tag) and autotagged_id in x.get("subtag_ids",[])]
    if(len(tagged_list) == 0):
        new_tag = {
            "id": tag_list[len(tag_list) - 1]["id"] + 1,
            "name": tag,
            "aliases": [
                tag
            ],
            "color": default_color,
            "subtag_ids":[autotagged_id]
        }
        tag_list.append(new_tag)
        return new_tag
    else:
        return tagged_list[len(tagged_list) -1]

def update_image_tag(tag_list: list, autotagging_tags: list, key: str, value: float, autotagger_tag: dict):
    output_list = []
    tag = autotagging_tags.get(key)
    if( tag is None):
        tag = get_autotagged_tag(tag_list, key, autotagger_tag['id'])
        autotagging_tags[key] = tag
    output_list.append(tag['id'])
    print(f"  {key}: {value:.3f}")
    return output_list

def main(opts: ScriptOptions):

    repo_id = MODEL_REPO_MAP.get(opts.model)
    if not opts.image_file_or_dir.exists():
        raise FileNotFoundError(f"Image file not found: {opts.image_file_or_dir}")
    if(opts.image_file_or_dir.is_dir()):
        images = get_filepaths(opts.image_file_or_dir.resolve())
        tagstudio_dir = opts.image_file_or_dir / '.TagStudio'
        if(not (tagstudio_dir).exists()):
            os.mkdir(tagstudio_dir)
        output_file = tagstudio_dir / 'ts_library.json'

    else:
        output_file = opts.output_json_file
        abspath=os.path.abspath(opts.image_file_or_dir)
        images = [{'fullpath': Path(abspath).resolve(),
                    'dirpath': os.path.dirname(abspath),
                    'filename': os.path.basename(abspath)}]

    json_file, json_data = read_json_data(output_file)
    tag_list = sorted(json_data['tags'], key=lambda x: x['id'])

    autotagger_list = [x for x in tag_list if x["name"] == AUTOTAGGER_TAG]
    if(len(autotagger_list) == 0):
        autotagger_tag = add_autotagger_tag(tag_list)
        autotagged_before = False
    else:
        autotagger_tag = autotagger_list[len(autotagger_list) - 1]
        autotagged_before = True

    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    model: nn.Module = timm.create_model("hf-hub:" + repo_id, pretrained=True)
    state_dict = timm.models.load_state_dict_from_hf(repo_id)
    model.load_state_dict(state_dict)

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Creating data transform...")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    autotagging_tags={x['name']: x for x in tag_list if autotagger_tag['id'] in x.get("subtag_ids",[])}

    #TODO not reprocess same file twice
    img_id=0
    print("Loading images and preprocessing...")
    entries_dict = {x.get('path','.') + x['filename']  : x for x in json_data['entries']}
    for image_path in images:
        is_new_entry=False
        print(image_path)
        # get image
        img_input: Image.Image = Image.open(image_path['fullpath'])
        # ensure image is RGB
        img_input = pil_ensure_rgb(img_input)
        # pad to square with white background
        img_input = pil_pad_square(img_input)
        # run the model's input transform to convert to tensor and rescale
        inputs: Tensor = transform(img_input).unsqueeze(0)
        # NCHW image RGB to BGR
        inputs = inputs[:, [2, 1, 0]]

        print("Running inference...")
        with torch.inference_mode():
            # move model to GPU, if available
            if torch_device.type != "cpu":
                model = model.to(torch_device)
                inputs = inputs.to(torch_device)
            # run the model
            outputs = model.forward(inputs)
            # apply the final activation function (timm doesn't support doing this internally)
            outputs = F.sigmoid(outputs)
            # move inputs, outputs, and model back to to cpu if we were on GPU
            if torch_device.type != "cpu":
                inputs = inputs.to("cpu")
                outputs = outputs.to("cpu")
                model = model.to("cpu")

        print("Processing results...")
        caption, taglist, ratings, character, general = get_tags(
            probs=outputs.squeeze(0),
            labels=labels,
            gen_threshold=opts.gen_threshold,
            char_threshold=opts.char_threshold,
        )
        relative_path = os.path.relpath(image_path['dirpath'], opts.image_file_or_dir)
        image_dict = entries_dict.get(relative_path + image_path['filename'])
        if(image_dict is None):
            image_dict = {"id": img_id,"filename": image_path['filename']}
            is_new_entry = True
        if(relative_path != '.'):
            image_dict['path'] = relative_path
            #check and test this whole block
        if(image_dict.get('fields') is not None):
            current_image_tag_list= [x['7'] for x in image_dict['fields'] if isinstance(x,dict) and x.get('7') is not None]
            if(len(current_image_tag_list) != 0):
                current_image_tag_list = current_image_tag_list[0]
            else:
                image_dict['fields'].append({"7": current_image_tag_list})
        else:
            current_image_tag_list = []
            image_dict['fields'] = [{"7": current_image_tag_list}]
        print("--------")
        print(f"Caption: {caption}")
        print("--------")
        print(f"Tags: {taglist}")

        print("--------")
        print("Ratings:")
        max_rating =('k',0)
        for k, v in ratings.items():
            if(v > max_rating[1]):
                max_rating=(k,v)
            print(f"  {k}: {v:.3f}")
        current_image_tag_list.extend(update_image_tag(tag_list, autotagging_tags, max_rating[0],max_rating[1], autotagger_tag))

        print("--------")
        print(f"Character tags (threshold={opts.char_threshold}):")
        for k, v in character.items():
            current_image_tag_list.extend(update_image_tag(tag_list, autotagging_tags, k, v, autotagger_tag))

        print("--------")
        print(f"General tags (threshold={opts.gen_threshold}):")
        for k, v in general.items():
            current_image_tag_list.extend(update_image_tag(tag_list, autotagging_tags, k, v, autotagger_tag))


        image_tag_list_reference = current_image_tag_list
        current_image_tag_list = sorted(set(current_image_tag_list))
        image_tag_list_reference.clear()
        image_tag_list_reference.extend(current_image_tag_list)
        #is this right?
        # image_dict["fields"] = [{"7": current_image_tag_list}]
        if(is_new_entry):
            json_data["entries"].append(image_dict)
        img_id=img_id+1
        print("Done!")
    json_data["tags"] = tag_list
    json_file.seek(0)
    json_file.truncate()
    json.dump(json_data, json_file)


if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
