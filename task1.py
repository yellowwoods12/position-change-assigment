import os, sys
import argparse
import copy
import numpy as np
import torch
from PIL import Image
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append("Inpaint-Anything/")

from huggingface_hub import hf_hub_download
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_args(parser):
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt with object class",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=25,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, default = "sam_vit_h_4b8939.pth" , 
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--box_threshold", type=int, default=0.3,
        help="Box threshold for groundingDINO.",
    )
    parser.add_argument(
        "--text_threshold", type=int, default=0.25,
        help="Text threshold for groundingDINO.",
    )

    

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def show_mask(mask, image, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        #red colored mask
        color = np.array([255, 0, 0, 0.1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        
    
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    sam = build_sam(checkpoint=args.sam_ckpt)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    TEXT_PROMPT = args.prompt

    local_image_path = args.image
    image_source, image = load_image(local_image_path)
    
    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=args.box_threshold, 
        text_threshold=args.text_threshold,
        device=DEVICE
    )
    
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    
    sam_predictor.set_image(image_source)

    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    masks = masks.cpu().numpy()
    masks = masks.astype(np.uint8) * 255
    orig_masks = masks.copy()
    masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks[0]]
    annotated_frame_with_mask = show_mask(orig_masks, annotated_frame)

    segmented_image = Image.fromarray(annotated_frame_with_mask)

    img_stem = Path(args.image).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    masked_img_p = str(out_dir / img_stem) + "_masked_object.png"

    segmented_image.save(masked_img_p)
    
    