import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate train files")
    parser.add_argument('--in_dir',
                        default='.',
                        help='root directory where frames are stored')
    parser.add_argument('--out_dir',
                        default='.',
                        help='where to store output file(s)')
    args = parser.parse_args()
    return args

def main(args):

    with open(os.path.join(args.out_dir, "train_files.txt"), "w") as trainfile:
        for name in sorted(glob.glob(os.path.join(args.in_dir, "Hampton*/*.png"))):
            trainfile.writelines([name, "\n"])

if __name__ == '__main__':
    args = parse_args()
    main(args)