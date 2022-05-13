import csv
from xml.dom import minidom
import os
from svgpathtools import parse_path
import numpy as np
from cairosvg import svg2png

class Render():
    def __init__(self, class_name, height, width, csv_path):
        super(Render, self).__init__()
        self.csv_path = csv_path
        self.class_name = class_name
        self.height = height
        self.width = width

    def csv2dict(self):
        reader = csv.reader(open(self.csv_path, 'r'))
        d = {}
        for row in reader:
            no, name, unicode = row
            d[name] = unicode
        return d

    def create_svg(self, bbox, base_path):
        ht = str(abs(bbox[3] - bbox[2]))
        wt = str(abs(bbox[1] - bbox[0]))

        try:
            index = np.argwhere(np.array(bbox) < 0)[0][0]
        except IndexError:
            index = -1

        tfr = ""
        if index == -1:
            tfr = "translate(0 0)"
        elif index < 2:
            tfr = "translate({0} 0)".format(abs(bbox[index]))
        else:
            tfr = "translate(0 {0})".format(abs(bbox[index]))

        root = minidom.Document()

        xml = root.createElement('svg')
        xml.setAttribute('width', wt)
        xml.setAttribute('height', ht)
        xml.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        xml.setAttribute('xlink', 'http://www.w3.org/1999/xlink')
        root.appendChild(xml)

        productChild = root.createElement('path')
        productChild.setAttribute('fill', '#000000')
        productChild.setAttribute('fill-rule', 'nonzero')
        productChild.setAttribute('transform', tfr)
        productChild.setAttribute('d', base_path)

        xml.appendChild(productChild)
        xml_str = root.toprettyxml(indent="\t")
        return xml_str

    def render(self, bravura_path, save_svg = True):
        uni_dict = self.csv2dict()
        file = minidom.parse(bravura_path)

        glyphs = file.getElementsByTagName('glyph')
        glyph_names = [glyph.attributes["glyph-name"].value for glyph in glyphs]
        index_uni = glyph_names.index(uni_dict[self.class_name])
        base_path = glyphs[index_uni].attributes['d'].value
        if self.class_name is 'tupletBracket':
            self.class_name = 'beam'
        if self.class_name is 'tie':
            base_path = "m 58.172768,0.3197629 c -8.621801,7.1639345 -49.2377949,7.1639345 -57.85316243,0 v 0 c 8.61536753,6.1963135 49.23136143,6.1963135 57.85316243,0 z"
        elif self.class_name is 'slur':
            base_path = "m 154,141.7 1.2,1.3 C 140,155 114.7,158 95.8,158 64.5,158 49.6,150.8 40,143.3 l 1.4,-1.9 c 6.7,7.3 33.4,11.7 55.3,11.7 25.2,0 41.9,-2.8 57.3,-11.3 z"
        else:
            base_path = glyphs[index_uni].attributes['d'].value

        path_alt = parse_path(base_path)
        bbox = path_alt.bbox()
        xml_str = self.create_svg(bbox, base_path)

        if not os.path.isdir('png_files/'):
            os.makedirs('png_files/')
        png_path = "png_files/{0}.png".format(self.class_name)
        png_data = svg2png(bytestring=xml_str.encode(), write_to=png_path, output_width=self.width, output_height=self.height)

        if save_svg:
            if not os.path.isdir('svg_files/'):
                os.makedirs('svg_files/')
            save_path_file = "svg_files/{0}.svg".format(self.class_name)
            with open(save_path_file, "w") as f:
                f.write(xml_str)