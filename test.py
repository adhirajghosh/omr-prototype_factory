import math
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image as PImage
from PIL.Image import Image
from PIL.ImageChops import invert
from PIL.ImageOps import grayscale
from shapely.affinity import rotate
from shapely.geometry import Polygon
from skimage.morphology import binary_erosion

from optical_flow import optical_flow_merging
# https://www.geogebra.org/classic
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + Math.min(ggbApplet.getValue('l1').toFixed(), ggbApplet.getValue('l2').toFixed()) + ", " + Math.max(ggbApplet.getValue('l1').toFixed(), ggbApplet.getValue('l2').toFixed()) + ", " + (ggbApplet.getValue('α')*180/Math.PI).toFixed()
from render import Render

SAMPLES = [
    ('images/lg-2267728-aug-gutenberg1939--page-2.png', [
        {
            'proposal': (1475, 2379, 19, 26, 2 / 180.0 * math.pi),
            'class': 29,  # noteheadHalfOnLine
            'gt': (1477, 2375, 16, 25, 7 / 180.0 * math.pi)
        },
        {
            'proposal': (125, 161, 45, 104, 0 / 180.0 * math.pi),
            'class': 6,  # clefG
            'gt': (127, 164, 44, 101, 5 / 180.0 * math.pi)
        }
    ]),
    ('images/lg-252689430529936624-aug-beethoven--page-3.png', [
        {
            'proposal': (506, 568, 16, 152, 0),
            'class': 123,  # tie
            'gt': (507, 569, 13, 148, 0)
        },
        {
            'proposal': (277, 525, 20, 47, 2 / 180.0 * math.pi),
            'class': 64,  # accidentalSharp
            'gt': (273, 528, 18, 48, 0)
        }
    ])
]


def pad(img: Image, pad: int) -> Image:
    width, height = img.size
    result = PImage.new(img.mode, (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    result.paste(img, (pad, pad))
    return result


def get_glyph(cls: int, bbox: Tuple[int, int, int, int, float], padding: int = 50) -> Image:
    _, _, w, h, a = bbox
    class_name = {
        6: 'clefG',
        29: 'noteheadHalfOnLine',
        64: 'accidentalSharp',
        123: 'tie'
    }[cls]
    png_data = Render(
        class_name=class_name, height=h, width=w, csv_path='name_uni.csv').render(
        'Bravura.svg', save_svg=False, save_png=False)
    with BytesIO(png_data) as bio:
        img = PImage.open(bio)
        img.load()
        img = img.rotate(a * 180.0 / math.pi, PImage.BILINEAR, expand=True, fillcolor=(0, 0, 0, 0))
        img = img.transpose(PImage.FLIP_TOP_BOTTOM)
        img = pad(img, padding)
        glyph = PImage.new("RGB", img.size, (255, 255, 255))
        glyph.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        glyph = invert(glyph)
        return grayscale(glyph)


def extract_bbox_from(glyph: Image, proposed_bbox: Tuple[int, int, int, int, float]) -> Tuple[
    int, int, int, int, float]:
    x1, y1, _, _, a = proposed_bbox
    rectified_glyph = glyph.rotate(-a * 180.0 / math.pi, PImage.BILINEAR, fillcolor=0)
    bx1, by1, bx2, by2 = rectified_glyph.getbbox()
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
    return int(x2), int(y2), int(w), int(h), a


def process(img: Image, proposed_bbox: Tuple[int, int, int, int, float], glyph: Image) -> Image:
    if len(proposed_bbox) == 0:
        return

    img_np = np.array(img.convert('L')) < 128
    w, h = glyph.size
    x, y, _, _, _ = proposed_bbox
    w2 = w / 2
    h2 = h / 2
    img_roi = img_np[int(y - h2):int(y + h2), int(x - w2):int(x + w2)]

    mask_np = np.array(glyph) > 0
    mask_np = np.flipud(mask_np)
    y_pad, x_pad = img_roi.shape[0] - mask_np.shape[0], img_roi.shape[1] - mask_np.shape[1]
    mask_np = np.pad(mask_np, ((int(np.ceil(y_pad / 2)), y_pad // 2), (int(np.ceil(x_pad / 2)), x_pad // 2)))
    new_mask = optical_flow_merging(img_roi, mask_np)
    return PImage.fromarray(binary_erosion(new_mask))


def bbox_to_polygon(bbox: Tuple[int, int, int, int, float]) -> Polygon:
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


def calc_loss(bbox1: Tuple[int, int, int, int, float], bbox2: Tuple[int, int, int, int, float]) -> float:
    a = bbox_to_polygon(bbox1)
    b = bbox_to_polygon(bbox2)
    return a.intersection(b).area / a.union(b).area


if __name__ == '__main__':
    for image_fp, samples in SAMPLES:
        img = PImage.open(image_fp)
        for sample in samples:
            glyph = get_glyph(sample['class'], sample['proposal'])
            new_glyph = process(img, sample['proposal'], glyph)
            derived_bbox = extract_bbox_from(glyph, sample['proposal'])
            new_bbox = extract_bbox_from(new_glyph, sample['proposal'])
            iou = calc_loss(sample['gt'], new_bbox)
            base_iou = calc_loss(sample['gt'], derived_bbox)
            print(f"IoU [{sample['class']}]: {iou:.3} (baseline: {base_iou:.3})")
