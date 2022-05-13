import math
from io import BytesIO

from PIL import Image as PImage
from PIL.Image import Image

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
            'proposal': (277, 525, 20, 47, 2),
            'class': 64,  # accidentalSharp
            'gt': (273, 528, 18, 48, 0)
        }
    ])
]


def get_png(class_name: str, width: int, height: int) -> Image:
    png_data = Render(
        class_name=class_name, height=height, width=width, csv_path='name_uni.csv').render(
        'Bravura.svg', save_svg=False, save_png=False)
    with BytesIO(png_data) as bio:
        return PImage.open(bio)


GLYPHS = {
    6: get_png('clefG', 44, 101),
    29: get_png('noteheadHalfOnLine', 25, 16),
    64: get_png('accidentalSharp', 30, 50),
    123: get_png('tie', 148, 13),
}


def process(img, bbox, glyph):
    return bbox


def calc_loss(img, bbox, glyph) -> float:
    pass


if __name__ == '__main__':
    for image_fp, samples in SAMPLES:
        img = PImage.open(image_fp)
        for sample in samples:
            glyph = GLYPHS[sample['class']]
            corr = process(img, sample['proposal'], glyph)
            gt_loss = calc_loss(img, sample['gt'], glyph)
            new_loss = calc_loss(img, sample['proposal'], glyph)
