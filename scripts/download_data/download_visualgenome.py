#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Downloads and prepares the VisualGenome images dataset.

All images are downloaded and regions are extracted. Regions with names that correspond with those in DCASE_VG_SELECTED
are kept and truncated from the original (full resolution) image.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from audiblelight.utils import get_project_root, sanitise_directory, sanitise_filepath
from scripts.download_data import utils

DEFAULT_PATH = str(get_project_root() / "resources/images")
DEFAULT_CLEANUP = True

# These classes are contained in audiblelight/class_mappings.py::DCASE2023Task3
DCASE_VG_SELECTED = {
    "telephone": ["telephone", "phone", "cellphone"],
    "waterTap": ["tap", "faucet"],
    "doorCupboard": [
        "door",
        "arched doorway",
        "open door",
        "doorway",
        "cupboard",
        "closet",
        "cabinet",
        "locker",
    ],
    "bell": [
        "bell",
        "chime",
        "alarm",
    ],
    "music": [
        "musician",
        "loudspeaker",
        "speaker",
    ],
    "musicInstrument": ["guitar", "instrument"],
    "femaleSpeech": ["woman", "girl", "lady", "female"],
    "maleSpeech": ["guy", "man", "gentleman", "male", "boy"],
    "laughter": ["mouth", "face", "head"],
    "clapping": ["hand", "two hands", "glove", "mitts", "mittens"],
    "footsteps": ["foot", "footstep", "shoe"],
    "domesticSounds": [
        "dish",
        "pot",
        "teapot",
        "pan",
        "knife",
        "fork",
        "spoon",
        "food",
        "microwave",
        "blender",
        "kettle",
        "tap",
        "faucet",
        "sink",
        "kitchen sink",
        "tub",
        "dryer",
        "toilet",
        "zipper",
    ],
    "knock": [
        "door",
        "arched doorway",
        "open door",
        "doorway",
        "cupboard",
        "closet",
        "cabinet",
        "locker",
    ],
}

VG_OBJECT_JSON = (
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"
)
VG_IMAGE_ZIPS = [
    ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "VG_100K"),
    ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "VG_100K_2"),
]

# If an image region contains fewer than this many pixels, we'll reject it
#  This is to ensure we don't have loads of low-res, blurry images
VG_MIN_PIXELS = 64 * 64


def format_aliases() -> dict[str, str]:
    """
    Maps visualgenome object names to a stable alias, e.g. person, persons, person is are all mapped to "person"
    """
    # Download the text file and split into a list of strings
    alias_txt = (
        open(
            get_project_root() / "scripts/download_data/visualgenome_object_alias.txt",
            "r",
        )
        .read()
        .splitlines()
    )

    alias_mapper = {}
    for obj_alias in alias_txt:
        split_alias = obj_alias.split(",")
        first_alias = split_alias[0]
        for al in split_alias:
            alias_mapper[al] = first_alias

    return alias_mapper


def format_desired_objects() -> dict[str, str]:
    """
    Creates a mapping between visualgenome class names and class names from DCASE2023
    """
    desired_objs_inv = defaultdict(list)
    for category, labels in DCASE_VG_SELECTED.items():
        for label in labels:
            desired_objs_inv[label].append(category)
    return dict(desired_objs_inv)


def extract_valid_regions(
    images: list[dict], alias_mapper: dict[str, str], obj_mapper: dict[str, str]
) -> list[dict]:
    """
    Extract valid bounding boxes for all visualgenome images where name is in desired classes
    """

    img_keep_objs = []

    # Iterate over all the images: one dict per image
    for img in tqdm(images, desc="Extracting valid regions from images..."):
        if "image_id" not in img:
            logger.warning("Could not get image ID!")
            continue

        image_id = img["image_id"]

        # Images can contain multiple objects
        for ob_inner in img["objects"]:

            # argh, why is this a list?
            assert len(ob_inner["names"]) == 1
            name = ob_inner["names"][0]

            # Convert names to desired alias
            if name in alias_mapper:
                name = alias_mapper[name]

            # If we don't want this object, skip over
            if name not in obj_mapper:
                continue

            # It is possible for some VG names to map to multiple DCASE classes
            #  so we need to append the region for each class
            for cls in obj_mapper[name]:
                obj_res = {
                    "cls": cls,
                    "vg_name": name,
                    "vg_obj_id": ob_inner["object_id"],
                    "vg_image_id": image_id,
                    "bbox": {
                        "x": ob_inner["x"],
                        "y": ob_inner["y"],
                        "w": ob_inner["w"],
                        "h": ob_inner["h"],
                    },
                }
                img_keep_objs.append(obj_res)

    return img_keep_objs


def handle_visualgenome_download(target_folder, src_url, src_folder) -> None:
    """
    Handle visual genome download and extraction into a target directory
    """
    url_path = target_folder / src_url.split("/")[-1]

    # Do the download if the zip doesn't exist
    if url_path.exists():
        logger.info(f"Skipping download for '{url_path.stem}', exists already!")
    else:
        utils.download_file(src_url, url_path)

    # Do the extract
    #  VG image zips contain two separate folder: VG_100K and VG_100K_2
    #  We want to extract all into the parent (temporary) directory)
    interim_folder_full = target_folder / src_folder
    logger.info(f"Extracting zip to {interim_folder_full}")
    utils.extract_zip(url_path, target_folder)

    # Move the files into the parent directory and delete the interim folder
    utils.move_all_files_in_folder(interim_folder_full, target_folder)
    shutil.rmtree(interim_folder_full)


def combine_extracted_regions(
    regions: list[dict], all_image_ids: list[int]
) -> dict[str, list[dict]]:
    res = {}
    for found_region in regions:
        img_id = found_region["vg_image_id"]
        if img_id in all_image_ids:
            if img_id not in res:
                res[img_id] = [found_region]
            else:
                res[img_id].append(found_region)
    return res


def load_image_as_array(img_path: Path) -> np.ndarray:
    # Load up the image as an array
    img_loaded = Image.open(sanitise_filepath(img_path))
    return np.asarray(img_loaded)


def extract_bounding_boxes(
    img_array: np.ndarray, img_regions: list[dict]
) -> list[tuple]:
    extracted_bboxes = []

    # Iterate through the regions found for this image
    for img_region in img_regions:
        # Get the bounding box coordinates
        bbox = img_region["bbox"]
        x1, y1, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        x2 = x1 + w
        y2 = y1 + h

        # Extract the region from the full image
        region_extract = img_array[y1:y2, x1:x2]

        # Skip over cases where the bounding box is too small, to prevent blurry/low-res images
        if np.prod(region_extract.shape[:-1]) < VG_MIN_PIXELS:
            continue

        # Append the extracted bounding box and class name
        class_name = img_region["cls"]
        extracted_bboxes.append((region_extract, class_name))

    return extracted_bboxes


def main(
    path: str = DEFAULT_PATH,
    cleanup: bool = DEFAULT_CLEANUP,
):
    logger.info("---- VISUALGENOME download script ----")
    logger.info(f"Images will be downloaded to: {path}")

    # Coerce to a Path object and create if missing
    path = sanitise_directory(path, create_if_missing=True)

    # Download the object JSON file: this contains the regions + names for each
    logger.info("Downloading object mappings...")
    objects_json_zip = path / "objects.json.zip"
    if not (objects_json_zip.exists()):
        utils.download_file(VG_OBJECT_JSON, path / "objects.json.zip")
    if not (path / "objects.json").exists():
        utils.extract_zip(path / "objects.json.zip", path)

    # Now we're ready to load up the JSON object
    imgs = json.load(open(path / "objects.json", "r"))

    # Format alias and DCASE object mappings
    logger.info("Creating object -> region mappings...")
    alias_mapping = format_aliases()
    obj_mapping = format_desired_objects()

    # Extract regions that correspond with desired objects
    regions = extract_valid_regions(imgs, alias_mapping, obj_mapping)
    logger.info(f"Found {len(regions)} valid regions!")

    # Create a temporary directory to download all images to
    image_tmp_path = sanitise_directory(path / "images_tmp", create_if_missing=True)

    # Download all images to this directory
    logger.info("Downloading visual genome images (this may take a while)")
    for url, interim_folder_path in VG_IMAGE_ZIPS:
        handle_visualgenome_download(image_tmp_path, url, interim_folder_path)

    # Grab the IDs of all images we have available
    all_image_ids = [int(pt.stem) for pt in image_tmp_path.glob("*.jpg")]

    # Combine to a dictionary with form {id1: [region1, region2, ...], id2: [region1, ...], id3: [], ...}
    combined_regions = combine_extracted_regions(regions, all_image_ids)

    # Iterate over all regions and extract the bounding boxes
    for img_id, img_regions in combined_regions.items():
        # Load up the image
        img_path = (image_tmp_path / str(img_id)).with_suffix(".jpg")
        img_loaded = load_image_as_array(img_path)

        # Grab all the valid bounding boxes
        img_bboxes = extract_bounding_boxes(img_loaded, img_regions)

        # Save all the bboxes
        for img_bbox, img_class_name in img_bboxes:
            # Silently increment the file suffix until a valid path is found
            #  This is needed in case we have e.g., multiple doorCupboard objects in one image
            img_class_dir = sanitise_directory(
                path / img_class_name, create_if_missing=True
            )
            img_fname = utils.increment_filename(
                str(img_class_dir / (str(img_id) + "_%s.jpg"))
            )

            # Save the bbox to the path
            Image.fromarray(img_bbox).save(img_fname)

    logger.info("Saved bounding boxes!")

    # Cleanup by removing all temporary images
    if cleanup:
        logger.info("Cleaning up...")
        shutil.rmtree(image_tmp_path)
        os.remove(path / "objects.json")
        os.remove(path / "objects.json.zip")

    logger.success("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Path to store and process the dataset, defaults to {DEFAULT_PATH}",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=f"Whether to cleanup after download, defaults to {DEFAULT_CLEANUP}",
        default=DEFAULT_CLEANUP,
    )
    args = vars(parser.parse_args())
    main(**args)
