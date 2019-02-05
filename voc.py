import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
from collections import defaultdict

def _parse_voc_object(object_node: ET.Element):
    return {
        "name": object_node.findtext("name"),
        "xmin": int(object_node.findtext("bndbox/xmin")),
        "xmax": int(object_node.findtext("bndbox/xmax")),
        "ymin": int(object_node.findtext("bndbox/ymin")),
        "ymax": int(object_node.findtext("bndbox/ymax")),
    }

def _parse_voc_annotation(annotation_node: ET.Element, image_directory, labels):
    instance = {
        "filename": os.path.join(image_directory, annotation_node.findtext("filename")),
        "width": int(annotation_node.findtext("size/width")),
        "height": int(annotation_node.findtext("size/height")),
        "object": [_parse_voc_object(node) for node in annotation_node.findall("object")]
    }

    # Filter objects by label
    if labels:
        instance["object"] = [obj for obj in instance["object"] if obj["name"] in labels]

    return instance

def parse_voc_annotation_file(filename, image_directory, labels=None):
    tree = ET.parse(filename)

    if tree.getroot() == "annotation":
        # Single file with single annotation
        inst = [_parse_voc_annotation(tree, image_directory, labels)]
        instances = [inst]
    else:
        # File with multiple annotations
        instances = [_parse_voc_annotation(node, image_directory, labels) for node in tree.findall("annotation")]
    
    # Filter instances without objects
    instances = [inst for inst in instances if inst["object"]]

    # Count labels
    label_counts = defaultdict(int)
    for inst in instances:
        for obj in inst["object"]:
            label_counts[obj["name"]] += 1

    return instances, label_counts


def parse_voc_annotation(ann_dir, image_directory, cache_name, labels=None):
    if cache_name and os.path.exists(cache_name):
        with open(cache_name, "rb") as handle:
            cache = pickle.load(handle)
        all_insts, label_counts = cache["all_insts"], cache["label_counts"]
    else:
        all_insts = []
        label_counts = defaultdict(int)
        
        for ann in sorted(os.listdir(ann_dir)):
            filename = os.path.join(ann_dir, ann)

            # Skip folders etc.
            if not os.path.isfile(filename):
                continue

            try:
                file_instances, file_label_counts = parse_voc_annotation_file(filename, image_directory, labels)
            except Exception as e:
                print("Failed to parse annotation file %s: %s" % (filename, e))
                continue

            # Add together instances and label counts
            all_insts += file_instances
            for k, v in file_label_counts.items():
                label_counts[k] += v

        # Save cache
        if cache_name:
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            with open(cache_name, "wb") as handle:
                cache = {"all_insts": all_insts, "label_counts": label_counts}
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
    return all_insts, label_counts
