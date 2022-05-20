import math
from render import Render
from io import BytesIO
import numpy as np
from PIL import Image as PImage
import argparse

class_names = (
    'brace', 'ledgerLine', 'repeatDot', 'segno', 'coda', 'clefG', 'clefCAlto', 'clefCTenor', 'clefF',
    'clefUnpitchedPercussion', 'clef8', 'clef15', 'timeSig0', 'timeSig1', 'timeSig2', 'timeSig3', 'timeSig4',
    'timeSig5', 'timeSig6', 'timeSig7', 'timeSig8', 'timeSig9', 'timeSigCommon', 'timeSigCutCommon',
    'noteheadBlackOnLine', 'noteheadBlackOnLineSmall', 'noteheadBlackInSpace', 'noteheadBlackInSpaceSmall',
    'noteheadHalfOnLine', 'noteheadHalfOnLineSmall', 'noteheadHalfInSpace', 'noteheadHalfInSpaceSmall',
    'noteheadWholeOnLine', 'noteheadWholeOnLineSmall', 'noteheadWholeInSpace', 'noteheadWholeInSpaceSmall',
    'noteheadDoubleWholeOnLine', 'noteheadDoubleWholeOnLineSmall', 'noteheadDoubleWholeInSpace',
    'noteheadDoubleWholeInSpaceSmall', 'augmentationDot', 'stem', 'tremolo1', 'tremolo2', 'tremolo3', 'tremolo4',
    'tremolo5', 'flag8thUp', 'flag8thUpSmall', 'flag16thUp', 'flag32ndUp', 'flag64thUp', 'flag128thUp', 'flag8thDown',
    'flag8thDownSmall', 'flag16thDown', 'flag32ndDown', 'flag64thDown', 'flag128thDown', 'accidentalFlat',
    'accidentalFlatSmall', 'accidentalNatural', 'accidentalNaturalSmall', 'accidentalSharp', 'accidentalSharpSmall',
    'accidentalDoubleSharp', 'accidentalDoubleFlat', 'keyFlat', 'keyNatural', 'keySharp', 'articAccentAbove',
    'articAccentBelow', 'articStaccatoAbove', 'articStaccatoBelow', 'articTenutoAbove', 'articTenutoBelow',
    'articStaccatissimoAbove', 'articStaccatissimoBelow', 'articMarcatoAbove', 'articMarcatoBelow', 'fermataAbove',
    'fermataBelow', 'caesura', 'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th',
    'rest32nd', 'rest64th', 'rest128th', 'restHNr', 'dynamicP', 'dynamicM', 'dynamicF', 'dynamicS', 'dynamicZ',
    'dynamicR', 'graceNoteAcciaccaturaStemUp', 'graceNoteAppoggiaturaStemUp', 'graceNoteAcciaccaturaStemDown',
    'graceNoteAppoggiaturaStemDown', 'ornamentTrill', 'ornamentTurn', 'ornamentTurnInverted', 'ornamentMordent',
    'stringsDownBow', 'stringsUpBow', 'arpeggiato', 'keyboardPedalPed', 'keyboardPedalUp', 'tuplet3', 'tuplet6',
    'fingering0', 'fingering1', 'fingering2', 'fingering3', 'fingering4', 'fingering5', 'slur', 'beam', 'tie',
    'restHBar', 'dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin', 'tuplet1', 'tuplet2', 'tuplet4', 'tuplet5',
    'tuplet7', 'tuplet8', 'tuplet9', 'tupletBracket', 'staff', 'ottavaBracket'
)

def parse_args():
    parser = argparse.ArgumentParser(description='Glyph transformation for effective post processing')
    parser.add_argument('--class_id', help='Id of the DeepScore Class', type=int, default=5)
    parser.add_argument('--csv_path', help='the path where name_uni.csv is stored', type=str, default='name_uni.csv')
    parser.add_argument('--glyph_height', help='height of glyph', type=int, default=254)
    parser.add_argument('--glyph_width', help='width of glyph', type=int, default=236)
    parser.add_argument('--glyph_angle', help='angle of rotation of glyph', type=float, default=0.0)
    parser.add_argument('--svg_path', help='the path where Bravura.svg is stored', type=str, default='Bravura.svg')
    parser.add_argument('--padding_left', help='Length of padding on the left of glyph', type=int, default=254)
    parser.add_argument('--padding_right', help='Length of padding on the right of glyph', type=int, default=254)
    parser.add_argument('--padding_top', help='Length of padding on the top of glyph', type=int, default=254)
    parser.add_argument('--padding_bottom', help='Length of padding on the bottom of glyph', type=int, default=254)

    args = parser.parse_args()

    return args


# def get_transformed_glyph(class_id: int, glyph_width: int, glyph_height: int, glyph_angle: float, array_width: int,
#                           array_height: int) -> np.array:
def get_transformed_glyph(args) -> np.array:
    """
    returns a glyph according the parameters
    :param class_id: The class id (type of the glyph)
    :param glyph_width: width of the glyph
    :param glyph_height: height of the glyph
    :param glyph_angle: angle of the glyph
    :param array_width: width of the returned numpy array
    :param array_height: height of the returned numpy array
    :return: numpy array with the glyph
    """

    def add_padding(img, top, right, bottom, left):
        new_width = right + left
        new_height = top + bottom
        result = PImage.new(img.mode, (new_width, new_height), (255, 255, 255))
        result.paste(img, (left, top))
        return result

    # Assuming the angle is not formatted, if it is comment the next line
    # glyph_angle = args.glyph_angle / 180.0 * math.pi

    class_name = class_names[args.class_id + 1]  # Taken from detection service, don't know why +1 is done
    case = Render(class_name=class_name, height=args.glyph_height, width=args.glyph_width, csv_path=args.csv_path)
    png_data = case.render(args.svg_path)
    with BytesIO(png_data) as bio:
        img = PImage.open(bio)
        img.load()
        img = img.rotate(args.glyph_angle * 180.0 / math.pi, PImage.BILINEAR, expand=True, fillcolor=(0, 0, 0, 0))
        img = img.transpose(PImage.FLIP_TOP_BOTTOM)
        img = add_padding(img, args.padding_top, args.padding_right, args.padding_bottom, args.padding_left)

    return np.array(img)

def main():
    args = parse_args()

    img = get_transformed_glyph(args)
    print(img.shape)


if __name__ == '__main__':
    main()
