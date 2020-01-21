import imageio
from PIL import Image
import os


def GET_DATE(FILE):  # gets the date of a given file
    if FILE[0] == '.':
        FILE = FILE[1:]
    if FILE[0:7] == "STEREO_":
        INFO = FILE.split("_")
        DATE = INFO[5][:-4]
    else:
        INFO = FILE.split('.')
        DATE = INFO[2].replace('-', '').replace('_', '').replace('T', '')[:10]
    return int(DATE)


near_uv_path = "./nearside_UV/"
near_mag_path = "./nearside_magnetogram/"
near_path = "./nearside_combined/"
far_uv_path = "./farside_UV/"
far_mag_path = "./generated_farside_magnetogram/"
far_path = "./farside_combined/"

path_list = []

if len(os.listdir(near_path)) == 0:
    path_list.append((near_uv_path, near_mag_path, near_path))
if len(os.listdir(far_path)) == 0:
    path_list.append((far_uv_path, far_mag_path, far_path))

for paths in path_list:
    uv = sorted(os.listdir(paths[0]), key=GET_DATE)
    mag = sorted(os.listdir(paths[1]), key=GET_DATE)

    assert len(uv) == len(mag)

    for i in range(len(uv)):
        name = uv[i][:-4]
        images = [Image.open(x) for x in [paths[0] + uv[i], paths[1] + mag[i]]]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(paths[2] + name + "_combined.png")
        print(name)

for path in (near_path, far_path):
    images = []
    for filename in sorted(os.listdir(path), key=GET_DATE):
        images.append(imageio.imread(path + filename))
    imageio.mimsave(path[2:-1] + '.gif', images)
