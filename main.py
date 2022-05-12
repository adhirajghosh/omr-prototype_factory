import argparse
import os
from render import Render

def parse_args():
    parser = argparse.ArgumentParser(description='Render classwise png representations')
    parser.add_argument('--class_name', help='Name of the DeepScore Class', default = 'clefG')
    parser.add_argument('--csv_path', help='the path where name_uni.csv is stored', default='name_uni.csv')
    parser.add_argument('--height', help='target height of final image', default=100)
    parser.add_argument('--width', help='target height of final image', default=100)
    parser.add_argument('--svg_path', help='the path where Bravura.svg is stored', default='Bravura.svg')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    #TODO: Add exceptions for tie, beam and slur
    case = Render(class_name=args.class_name, height = args.height, width = args.width, csv_path= args.csv_path)
    case.render(args.svg_path)

if __name__ == '__main__':
    main()