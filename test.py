import math
import warnings
from io import BytesIO
from typing import Tuple, List

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
# https://www.geogebra.org/classic
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + ggbApplet.getValue('w').toFixed() + ", " + ggbApplet.getValue('h').toFixed() + ", " + (ggbApplet.getValue('α')*180/Math.PI).toFixed()
from render import Render

warnings.filterwarnings("ignore", category=DeprecationWarning)

SAMPLES = [
    ('images/lg-2267728-aug-gutenberg1939--page-2.png', [
        {
            'proposal': (1461, 1238, 25, 16, 7 / 180.0 * math.pi),
            'class': 29,  # noteheadHalfOnLine
            'gt': (1469, 1236, 24, 18, 0)
        },
        {
            'proposal': (113, 155, 42, 92, 2 / 180.0 * math.pi),
            'class': 6,  # clefG
            'gt': (128, 165, 43, 101, 0)
        }
    ]),
    ('images/lg-252689430529936624-aug-beethoven--page-3.png', [
        {
            'proposal': (499, 565, 144, 14, 0),
            'class': 123,  # tie
            'gt': (507, 569, 147, 14, 0)
        },
        {
            'proposal': (271, 523, 17, 45, -8 / 180 * math.pi),
            'class': 64,  # accidentalSharp
            'gt': (273, 528, 19, 49, 0)
        }
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


def get_glyphs(cls: int, bbox: Tuple[int, int, int, int, float], padding: int = 50) -> List[Image]:
    _, _, w, h, a = bbox
    class_name = {
        6: 'clefG',
        29: 'noteheadHalfOnLine',
        64: 'accidentalSharp',
        123: 'tie'
    }[cls]
    glyph = cls_to_glyph(class_name, w, h, a, padding)
    if class_name == 'tie':
        return [glyph, glyph.transpose(PImage.FLIP_TOP_BOTTOM)]
    return [glyph]


def extract_bbox_from(glyph: Image, proposed_bbox: Tuple[int, int, int, int, float], cls: int) -> Tuple[
    int, int, int, int, float]:
    x1, y1, _, _, a = proposed_bbox
    angle = 0.0
    if cls in [123]:  # Transfer angle if tie, slur etc.
        angle = a
    rectified_glyph = glyph.rotate(-a * 180.0 / math.pi, PImage.BILINEAR, fillcolor=0)
    bbox = rectified_glyph.getbbox()
    if bbox is None:
        return 0, 0, 0, 0, 0.0
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
    return int(x2), int(y2), int(w), int(h), angle


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


def visualize(crop: Image, prop_bbox: Tuple[int, int, int, int, float], gt_bbox: Tuple[int, int, int, int, float],
              gt_glyph: Image, new_bbox: Tuple[int, int, int, int, float], new_glyph: Image):
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
    s = int(math.sqrt(w ** 2 + h ** 2))
    img.crop((x - s, y - s, x + s, y + s)).show()


if __name__ == '__main__':
    for image_fp, samples in SAMPLES:
        img = PImage.open(image_fp)
        for sample in samples:
            scores = []
            print(f"IoU [{sample['class']}]: ", end='')
            for glyph in get_glyphs(sample['class'], sample['proposal'], 0):
                new_glyph = process(img, sample['proposal'], glyph)
                derived_bbox = extract_bbox_from(glyph, sample['proposal'], sample['class'])
                new_bbox = extract_bbox_from(new_glyph, sample['proposal'], sample['class'])
                iou = calc_loss(sample['gt'], new_bbox)
                base_iou = calc_loss(sample['gt'], derived_bbox)
                visualize(img, sample['proposal'], sample['gt'], glyph, new_bbox, new_glyph)
                print(f", {iou:.3} (baseline: {base_iou:.3})", end='')
            print()
