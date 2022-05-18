import io
import math

import cv2
import numpy as np
from PIL import Image

from optical_flow import optical_flow_merging
from skimage.morphology import binary_erosion
# https://www.geogebra.org/classic
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + Math.min(ggbApplet.getValue('l1').toFixed(), ggbApplet.getValue('l2').toFixed()) + ", " + Math.max(ggbApplet.getValue('l1').toFixed(), ggbApplet.getValue('l2').toFixed()) + ", " + (ggbApplet.getValue('α')*180/Math.PI).toFixed()
from render import Render

SAMPLES = [
    ('data/ds2_dense/images/lg-2267728-aug-gutenberg1939--page-2.png', [
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
    ('data/ds2_dense/images/lg-252689430529936624-aug-beethoven--page-3.png', [
        {
            'proposal': (506, 568, 16, 152, 0),
            'class': 123,  # tie
            'gt': (507, 569, 13, 148, 0)
        },
        {
            'proposal': (),
            'class': 64,  # accidentalSharp
            'gt': ()
        }
    ])
]

MASKS = {
    6: Render(class_name='clefG', height=101, width=44, csv_path='name_uni.csv').render('Bravura.svg', save_svg=False,
                                                                                        save_png=False),
    29: Render(class_name='noteheadHalfOnLine', height=50, width=22, csv_path='name_uni.csv').render('Bravura.svg',
                                                                                                      save_svg=False,
                                                                                                      save_png=False),
    64: Render(class_name='accidentalSharp', height=101, width=44, csv_path='name_uni.csv').render('Bravura.svg',
                                                                                                   save_svg=False,
                                                                                                   save_png=False),
    123: Render(class_name='tie', height=101, width=44, csv_path='name_uni.csv').render('Bravura.svg', save_svg=False,
                                                                                        save_png=False),
}

# TODO: Copied from MMDET
def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)


def get_best_begin_point(coordinates):
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def rotated_box_to_poly_np(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle = rrect[:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)

    if rrects.shape[1] > 5 and rrects.shape[0] > 0:
        # scores are being dropped -> but scores are needed to sort proposals later on.
        polys_and_scores = np.concatenate((polys, rrects[:, [5]]), axis=1)
    else:
        polys_and_scores = polys

    return polys_and_scores

# TODO: End -- Copied from MMDET

def get_roi(img, bbox):
    area_size = (np.sqrt(bbox[2] ** 2 + bbox[3] ** 2) + 50) / 2
    x_min, x_max = int(max(0, np.floor(bbox[0] - area_size))), int(min(img.shape[1], np.ceil(bbox[0] + area_size)))
    y_min, y_max = int(max(0, np.floor(bbox[1] - area_size))), int(min(img.shape[0], np.ceil(bbox[1] + area_size)))
    return img[y_min:y_max, x_min:x_max], (x_min, x_max), (y_min, y_max)


def mask_to_bb(mask):
    cnts, hierarchy = cv2.findContours((255 * mask).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pos, size, angle = cv2.minAreaRect(np.concatenate(cnts))
    return np.array([[pos[1], pos[0], size[1], size[0], -angle / 180 * math.pi]])


def process(img, bbox, mask):
    if len(bbox) == 0:
        return

    img_np = np.array(img.convert('L')) < 128
    img_roi, x_area, y_area = get_roi(img_np, bbox)

    mask_np = np.array(Image.open(io.BytesIO(mask)))[..., 3] > 0
    mask_np = np.flipud(mask_np)
    y_pad, x_pad = img_roi.shape[0] - mask_np.shape[0], img_roi.shape[1] - mask_np.shape[1]
    mask_np = np.pad(mask_np, ((int(np.ceil(y_pad / 2)), y_pad // 2), (int(np.ceil(x_pad / 2)), x_pad // 2)))
    new_mask = optical_flow_merging(img_roi, mask_np)
    new_mask = binary_erosion(new_mask)

    if np.average(img_roi[new_mask > 0]) <= np.average(img_roi[mask_np > 0]):
        return bbox

    else:
        new_bb = mask_to_bb(new_mask)

        # TODO: Delte Plotting (just for debugging)
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from skimage.draw import line

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 20), dpi=300)

        axes[0, 0].imshow(img_roi, cmap=cm.gray)
        axes[0, 0].imshow(mask_np, cmap=cm.winter, alpha=0.5)
        axes[0, 1].imshow(img_roi, cmap=cm.gray)
        axes[0, 1].imshow(new_mask, cmap=cm.winter, alpha=0.5)

        axes[1, 0].imshow(mask_np, cmap=cm.winter)
        axes[1, 1].imshow(new_mask, cmap=cm.winter)

        mask_np_draw = mask_np.copy().astype("uint8")
        bbox_draw_orig = np.array([bbox])
        bbox_draw_orig[0][0] -= x_area[0]
        bbox_draw_orig[0][1] -= y_area[0]
        bb_format = rotated_box_to_poly_np(bbox_draw_orig)
        for (a, b), (c, d) in [(bb_format[0, :2], bb_format[0, 2:4]), (bb_format[0, 2:4], bb_format[0, 4:6]),
                               (bb_format[0, 4:6], bb_format[0, 6:]), (bb_format[0, 6:], bb_format[0, :2])]:
            mask_np_draw[line(int(a), int(b), int(c), int(d))] = 3

        axes[2, 0].imshow(img_roi, cmap=cm.gray)
        axes[2, 0].imshow(mask_np_draw, cmap=cm.tab10, alpha=0.7)

        mask_np_draw = new_mask.copy().astype("uint8")
        bb_format = rotated_box_to_poly_np(np.array(new_bb))
        for (a, b), (c, d) in [(bb_format[0, 0:2], bb_format[0, 2:4]), (bb_format[0, 2:4], bb_format[0, 4:6]), (bb_format[0, 4:6], bb_format[0, 6:]), (bb_format[0, 6:], bb_format[0, :2])]:
            mask_np_draw[line(int(a), int(b), int(c), int(d))] = 3

        axes[2, 1].imshow(img_roi, cmap=cm.gray)
        axes[2, 1].imshow(mask_np_draw, cmap=cm.tab10, alpha=0.7)

        plt.show()

        new_bb[0][0] += x_area[0]
        new_bb[0][1] += y_area[0]
        print(new_bb)

        return new_bb


def calc_loss(img, bbox, mask) -> float:
    pass


if __name__ == '__main__':
    for image_fp, samples in SAMPLES:
        img = Image.open(image_fp)
        for sample in samples:
            mask = MASKS[sample['class']]
            corr = process(img, sample['proposal'], mask)
            gt_loss = calc_loss(img, sample['gt'], mask)
            new_loss = calc_loss(img, sample['proposal'], mask)
