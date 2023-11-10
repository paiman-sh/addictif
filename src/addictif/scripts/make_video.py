import os
import argparse
from addictif.common.utils import mpi_size, mpi_print

def parse_args():
    parser = argparse.ArgumentParser(description="Make video from imshow")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to images")
    parser.add_argument("--field", type=str, required=True, help="Field to make video from")
    parser.add_argument("--direction", type=str, required=True, help="Direction x, y or z")
    return parser.parse_args()

def main():
    args = parse_args()

    if mpi_size > 1:
        mpi_print("Please run in serial.")
        exit()

    imgfolder = args.input

    fieldstr = "scan_" + args.field + "_" + args.direction

    files = [] 
    for f in os.listdir(imgfolder):
        if f[:len(fieldstr)] == fieldstr:
            files.append(f)

    if len(files) == 0:
        print("No matching files.")
        exit()

    outname = os.path.join(args.input, fieldstr + ".mp4")

    os.system(f"ffmpeg -framerate 30 -i {os.path.join(imgfolder, fieldstr)}_step%06d.png -c:v libx264 -pix_fmt yuv420p -y {outname}")

    print("Output:", outname)

if __name__ == "__main__":
    main()