# Music Note Rendering Tool for OMR Post-Processing

## Installation
1.  Install cairosvg
    This can be installed executing the following line on linux
    ```bash
    pip install cairosvg
    ```
    For Windows, if the above installation command isn't enough, create an issue on this repo. I will append the Windows installation guidelines in this README
2. Install svgpathtools
    ```bash
    pip install cairosvg
    ```
 ## Running the Renderer
```bash
python main.py --class_name 'choose class name from DeepScores_v2 naming schema' --csv_path 'path where name_uni.csv is located' --height 'target height of rendered image' --width 'target width of rendered image' --svg_path 'path where Bravura.svg is located'
```
