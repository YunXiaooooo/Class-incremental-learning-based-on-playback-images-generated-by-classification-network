import os, shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourceDir', type=str)
    parser.add_argument('--targetDir', type=str)
    args = parser.parse_args()

    targetDir = args.targetDir
    sourceDir = args.sourceDir

    classTargetDir=os.listdir(targetDir)
    if(len(classTargetDir)==0):
        print("inversion none!")
        print("copy all reserve")
        classTargetDir = os.listdir(sourceDir)

    for className in classTargetDir:
        src = sourceDir+className
        dst = targetDir+className

        if not os.path.exists(dst):
            os.makedirs(dst)

        imgs = os.listdir(src)
        for img in imgs:
            shutil.copy(src+"/"+img, dst+"/r"+img)