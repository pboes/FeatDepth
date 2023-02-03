import cv2
import os
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Turn videos into frames")
    parser.add_argument('--in_dir',
                        default='.',
                        help='directory where mp4s are stored')
    parser.add_argument('--out_dir',
                        default='.',
                        help='directory where frames are stored')
    args = parser.parse_args()
    return args


def main(args):


    videoPath = os.path.join(args.in_dir, "*.mp4")
    videos = glob.glob(videoPath)

    parentDir = args.out_dir

    j = 0

    for video in sorted(videos):

        fileName = os.path.basename(video)
        fileName = os.path.splitext(fileName)[0]


        newDir = os.path.join(parentDir, fileName)
        os.mkdir(newDir)

        inputpath = os.path.abspath(video)

        outpath = os.path.abspath(newDir)

        # Opens the Video file
        cap = cv2.VideoCapture(inputpath)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(
                os.path.join(outpath, fileName + "_" + str(i).zfill(4) + ".png"), frame
            )
            i += 1

        cap.release()
        cv2.destroyAllWindows()

        j = j + 1

if __name__ == '__main__':
    args = parse_args()
    main(args)
