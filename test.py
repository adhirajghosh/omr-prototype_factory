import math
import warnings
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image as PImage
from PIL.Image import Image
from PIL.ImageChops import invert
from PIL.ImageDraw import Draw
from PIL.ImageOps import grayscale
from shapely.affinity import rotate
from shapely.geometry import Polygon
from skimage.morphology import binary_erosion

from optical_flow import optical_flow_merging
from render import Render

warnings.filterwarnings("ignore", category=DeprecationWarning)

# https://www.geogebra.org/calculator
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + ggbApplet.getValue('w').toFixed() + ", " + ggbApplet.getValue('h').toFixed() + ", " + ggbApplet.getValue('α').toPrecision(4)
# or
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + ggbApplet.getValue('w').toFixed() + ", " + ggbApplet.getValue('h').toFixed() + ", " + -ggbApplet.getValue('β').toPrecision(4)
SAMPLES = [
    ('images/sample.png', [
        {
            'proposal': np.array([155, 242, 110, 41, 1.4105, "clefG"]),
            'gt': np.array([156, 240, 43, 116, 0.0, "clefG"])
        }, {
            'proposal': np.array([513, 180, 19, 16, -0.2863, "noteheadBlackOnLine"]),
            'gt': np.array([513, 180, 19, 17, 0.0, "noteheadBlackOnLine"])
        }, {
            'proposal': np.array([513, 163, 71, 14, -0.0841, "slur"]),
            'gt': np.array([512, 164, 75, 18, 0.1401, "slur"])
        }, {
            'proposal': np.array([619, 240, 30, 16, 1.8093, "rest8th"]),
            'gt': np.array([621, 240, 17, 27, 0.0, "rest8th"])
        }, {
            'proposal': np.array([946, 311, 43, 13, 2.1764, "dynamicF"]),
            'gt': np.array([948, 312, 34, 37, 0.0, "dynamicF"])
        }, {
            'proposal': np.array([985, 193, 28, 3, 0.277, "ledgerLine"]),
            'gt': np.array([987, 194, 26, 3, 0.0, "ledgerLine"])
        }, {
            'proposal': np.array([1308, 266, 98, 5, 0.0303, "beam"]),
            'gt': np.array([1310, 266, 100, 7, -0.03434, "beam"])
        }, {
            'proposal': np.array([590, 311, 551, 23, -0.0208, "dynamicCrescendoHairpin"]),
            'gt': np.array([597, 311, 586, 26, 0.0, "dynamicCrescendoHairpin"])
        }, {
            'proposal': np.array([734, 1087, 43, 11, 1.891, "flag8thDown"]),
            'gt': np.array([733, 1088, 19, 44, 0.0, "flag8thDown"])
        },
    ])
]


def pad(img: Image, pad: int) -> Image:
    width, height = img.size
    result = PImage.new(img.mode, (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    result.paste(img, (pad, pad))
    return result


def cls_to_glyph(class_name: str, width: int, height: int, angle: float, padding: int) -> Image:
    png_data = Render(
        class_name=class_name, height=height, width=width, csv_path='name_uni.csv').render(
        'Bravura.svg', save_svg=False, save_png=False)
    with BytesIO(png_data) as bio:
        img = PImage.open(bio)
        img.load()
        img = img.rotate(angle * 180.0 / math.pi, PImage.BILINEAR, expand=True, fillcolor=(0, 0, 0, 0))
        img = img.transpose(PImage.FLIP_TOP_BOTTOM)
        img = pad(img, padding)
        glyph = PImage.new("RGB", img.size, (255, 255, 255))
        glyph.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        glyph = invert(glyph)
        return grayscale(glyph)


def bbox_translate(bbox: np.ndarray) -> np.ndarray:
    if bbox[4] > np.pi / 4:
        bbox[4] = bbox[4] - np.pi / 2
        bbox[3], bbox[2] = bbox[2], bbox[3]
    return bbox


def get_glyphs(cls: str, bbox: np.ndarray, padding: int = 50) -> List[Image]:
    _, _, w, h, a = bbox
    glyph = cls_to_glyph(cls, w, h, a, padding)
    if cls in ['tie', 'slur']:
        return [glyph, glyph.transpose(PImage.FLIP_TOP_BOTTOM)]
    return [glyph]


def extract_bbox_from(glyph: Image, prop_bbox: np.ndarray, cls: str) -> np.ndarray:
    x1, y1, _, _, a = prop_bbox
    angle = 0.0
    if cls in ['tie', 'slur', 'beam']:  # Transfer angle if tie, slur etc.
        angle = a
    rectified_glyph = glyph.rotate(-angle * 180.0 / math.pi, PImage.BILINEAR, fillcolor=0)
    bbox = rectified_glyph.getbbox()
    if bbox is None:
        return prop_bbox
    bx1, by1, bx2, by2 = bbox
    cx1, cy1 = glyph.size
    cx1 /= 2
    cy1 /= 2
    w = bx2 - bx1
    h = by2 - by1
    cx2 = bx1 + w / 2
    cy2 = by1 + h / 2
    cxdiff = cx2 - cx1
    cydiff = cy2 - cy1
    x2 = x1 + cxdiff
    y2 = y1 + cydiff
    return np.array([int(x2), int(y2), int(w), int(h), angle])


def process(img: Image, bbox: np.ndarray, glyph: Image) -> Image:
    if len(bbox) == 0:
        return

    img_np = np.array(img.convert('L')) < 128
    w, h = glyph.size
    x, y, _, _, _ = bbox
    w2 = w / 2
    h2 = h / 2
    img_roi = img_np[int(y - h2):int(y + h2), int(x - w2):int(x + w2)]

    mask_np = np.array(glyph) > 0
    mask_np = np.flipud(mask_np)
    y_pad, x_pad = img_roi.shape[0] - mask_np.shape[0], img_roi.shape[1] - mask_np.shape[1]
    mask_np = np.pad(mask_np, ((int(np.ceil(y_pad / 2)), y_pad // 2), (int(np.ceil(x_pad / 2)), x_pad // 2)))
    new_mask = optical_flow_merging(img_roi, mask_np)
    return PImage.fromarray(binary_erosion(new_mask))


def bbox_to_polygon(bbox: np.ndarray) -> Polygon:
    x, y, w, h, a = bbox
    w2 = w / 2.0
    h2 = h / 2.0
    p = Polygon([
        (x - w2, y - h2),
        (x + w2, y - h2),
        (x + w2, y + h2),
        (x - w2, y + h2),
    ])
    return rotate(p, a, use_radians=True)


def calc_loss(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    a = bbox_to_polygon(bbox1)
    b = bbox_to_polygon(bbox2)
    return a.intersection(b).area / a.union(b).area


def visualize(crop: Image, prop_bbox: np.ndarray, gt_bbox: np.ndarray,
              gt_glyph: Image, new_bbox: np.ndarray, new_glyph: Image):
    img = crop.copy().convert('RGBA')
    draw = Draw(img, 'RGBA')
    draw.polygon(list(bbox_to_polygon(prop_bbox).exterior.coords)[:4], outline='#E00')
    draw.polygon(list(bbox_to_polygon(gt_bbox).exterior.coords)[:4], outline='#1F2')
    draw.polygon(list(bbox_to_polygon(new_bbox).exterior.coords)[:4], outline='#EC0')
    # x, y, _, _, _ = prop_bbox
    # glyph_img = PImage.new('RGBA', img.size, '#0000')
    # Draw(glyph_img).bitmap((x - (gt_glyph.width // 2), y - (gt_glyph.height // 2)), gt_glyph, fill='#E005')
    # img = PImage.alpha_composite(img, glyph_img)
    x, y, _, _, _ = gt_bbox
    glyph_img = PImage.new('RGBA', img.size, '#0000')
    Draw(glyph_img).bitmap((x - (gt_glyph.width // 2), y - (gt_glyph.height // 2)), gt_glyph, fill='#1F28')
    img = PImage.alpha_composite(img, glyph_img)
    x, y, _, _, _ = new_bbox
    glyph_img = PImage.new('RGBA', img.size, '#0000')
    Draw(glyph_img).bitmap((x - (new_glyph.width // 2), y - (new_glyph.height // 2)), new_glyph, fill='#EC0A')
    img = PImage.alpha_composite(img, glyph_img)
    x, y, w, h, _ = gt_bbox
    img.crop((x - w // 2 - 50, y - h // 2 - 50, x + w // 2 + 50, y + h // 2 + 50)).show()


if __name__ == '__main__':
    for image_fp, samples in SAMPLES:
        img = PImage.open(image_fp)
        for sample in samples:
            scores = []
            det_bbox = sample['proposal']
            prop_bbox: np.ndarray = sample['proposal'][:5].astype(np.float)
            prop_bbox = bbox_translate(prop_bbox)
            cls: str = sample['proposal'][5]
            gt_bbox: np.ndarray = sample['gt'][:5].astype(np.float)
            print(f"IoU [{cls}]: ", end='')
            for glyph in get_glyphs(cls, prop_bbox, 0):
                new_glyph = process(img, prop_bbox, glyph)
                derived_bbox = extract_bbox_from(glyph, prop_bbox, cls)
                new_bbox = extract_bbox_from(new_glyph, prop_bbox, cls)
                iou = calc_loss(gt_bbox, new_bbox)
                base_iou = calc_loss(gt_bbox, derived_bbox)
                visualize(img, prop_bbox, gt_bbox, glyph, new_bbox, new_glyph)
                print(f", {iou:.3} (baseline: {base_iou:.3})", end='')
            print()
