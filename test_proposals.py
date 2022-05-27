
import warnings
import pickle
import numpy as np
from PIL import Image as PImage
from PIL.Image import Image
from PIL.ImageDraw import Draw
from test import pad, cls_to_glyph, bbox_translate, get_glyphs, extract_bbox_from, get_roi, process2, process, bbox_to_polygon

warnings.filterwarnings("ignore", category=DeprecationWarning)

# https://www.geogebra.org/calculator
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + ggbApplet.getValue('w').toFixed() + ", " + ggbApplet.getValue('h').toFixed() + ", " + ggbApplet.getValue('α').toPrecision(4)
# or
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + ggbApplet.getValue('w').toFixed() + ", " + ggbApplet.getValue('h').toFixed() + ", " + -ggbApplet.getValue('β').toPrecision(4)
# [x_ctr,y_ctr,w,h,angle]

with open('proposals.pkl', 'rb') as f:
    samples = pickle.load(f)
image_fp = 'images/sample.png'
img = PImage.open(image_fp)


def visualize(crop: Image, prop_bbox: np.ndarray,
              gt_glyph: Image, new_bbox: np.ndarray, new_glyph: Image):
    img = crop.copy().convert('RGBA')
    draw = Draw(img, 'RGBA')
    draw.polygon(list(bbox_to_polygon(prop_bbox).exterior.coords)[:4], outline='#E00')
    draw.polygon(list(bbox_to_polygon(new_bbox).exterior.coords)[:4], outline='#EC0')
    # x, y, _, _, _ = gt_bbox
    glyph_img = PImage.new('RGBA', img.size, '#0000')
    # Draw(glyph_img).bitmap((x - (gt_glyph.width // 2), y - (gt_glyph.height // 2)), gt_glyph, fill='#1F28')
    img = PImage.alpha_composite(img, glyph_img)
    x, y, _, _, _ = new_bbox
    glyph_img = PImage.new('RGBA', img.size, '#0000')
    Draw(glyph_img).bitmap((x - (new_glyph.width // 2), y - (new_glyph.height // 2)), new_glyph, fill='#EC0A')
    img = PImage.alpha_composite(img, glyph_img)
    # x, y, w, h, _ = gt_bbox
    # img.crop((x - w // 2 - 50, y - h // 2 - 50, x + w // 2 + 50, y + h // 2 + 50)).show()
    img.save

def det_process(img, samples):
    new_bboxes = []
    for sample in samples:
        det_bbox = sample['proposal']
        prop_bbox: np.ndarray = sample['proposal'][:5].astype(np.float)
        prop_bbox = bbox_translate(prop_bbox)
        cls: str = sample['proposal'][5]
        for glyph in get_glyphs(cls, prop_bbox, 0):
            new_glyph = process2(img, prop_bbox, glyph, cls)
            derived_bbox = extract_bbox_from(glyph, prop_bbox, cls)
            new_bbox = extract_bbox_from(new_glyph, prop_bbox, cls)
            new_bboxes.append(new_bbox)
    return np.array(new_bbox)

if __name__ == '__main__':

        for sample in samples:
            scores = []
            det_bbox = sample['proposal']
            prop_bbox: np.ndarray = sample['proposal'][:5].astype(np.float)
            prop_bbox = bbox_translate(prop_bbox)
            cls: str = sample['proposal'][5]
            for glyph in get_glyphs(cls, prop_bbox, 0):
                new_glyph = process2(img, prop_bbox, glyph, cls)
                derived_bbox = extract_bbox_from(glyph, prop_bbox, cls)
                new_bbox = extract_bbox_from(new_glyph, prop_bbox, cls)
                visualize(img, prop_bbox, glyph, new_bbox, new_glyph)

            print()

