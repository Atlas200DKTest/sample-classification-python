# -*- coding: utf-8 -*-
import os
import argparse
import sys
import cv2 as cv
import numpy as np
import platform
import struct

def putText(srcFile, dstFile, text) :
    image = cv.imread(srcFile)
    cv.putText(image, text, (15,20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255) )
    cv.imwrite(dstFile, image);  
   
  

def showpic(data):
    cv.imshow("result", data)
    cv.waitKey(0)
    cv.destroyAllWindows()


def check_path(path):
    key = '\\'
    if platform.system() == "Linux":
        key = '/'
    if path[len(path)-1] != key:
        path += key
    return path


def Parse():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Convert Jpeg To NV12 and BGR raw data')
         
    parser.add_argument('--src  ', dest='src_name',
                        help='cmv or ped', type=str)
    parser.add_argument('--resize_h  ', dest='resize_h',
                        help='resize_h', type=str)
    parser.add_argument('--resize_w  ', dest='resize_w',
                        help='resize_w', type=str)
                        
    if len(sys.argv) == 1:
        helpInfo()
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def is_img(ext):
    ext = os.path.splitext(ext.lower())[1]
    if ext == '.jpg':
        return True
    elif ext == '.png':
        return True
    elif ext == '.jpeg':
        return True
    elif ext == '.bmp':
        return True
    else:
        return False


def mergeUV(u, v):
    if u.shape == v.shape:
        uv = np.zeros(shape=(u.shape[0], u.shape[1]*2))
        for i in range(0, u.shape[0]):
            for j in range(0, u.shape[1]):
                uv[i, 2*j] = u[i, j]
                uv[i, 2*j+1] = v[i, j]
        return uv
    else:
        print("size of Channel U is different with Channel V")


def rgb2nv12(image):
    if image.ndim == 3:
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        y = (0.299*r+0.587*g+0.114*b)
        u = (-0.169*r-0.331*g+0.5*b+128)[::2, ::2]
        v = (0.5*r-0.419*g-0.081*b+128)[::2, ::2]
        uv = mergeUV(u, v)
        yuv = np.vstack((y, uv))
        return yuv.astype(np.uint8)
    else:
        print("image is not BGR format")


def package2planar(image):
    if image.ndim == 3:
        b, g, r = cv.split(image)
        bgr = np.stack((b, g, r))
        bgr = np.reshape(bgr, (image.shape[2], image.shape[0], image.shape[1]))
        return bgr
    else:
        print("image is not BGR format")


def compare(src, dest):
    if src.shape == dest.shape:
        if (src == dest).all():
            return True
        else:
            return False
    else:
        print("These two images are not in same size")
        return False


def mkdirown(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def jpeg2yuv(src_name,resize_w, resize_h):
    src_root_path =  '../jpg'   #'./data/jpg'
    nv12_root_path = '../nv12'  #'./data/nv12'
    bgr_dest_path =  '../bgr'  #'./data/bgr'
    
    image_ori = cv.imread(src_name)         
    image_ori = cv.resize(image_ori,(resize_w, resize_h))   
    yuv = rgb2nv12(image_ori)
    
    return yuv

def saveFile(yuv, fileName) :
    with open(fileName, "wb") as fp:
        fp.write(yuv)
        fp.close()       
     



def helpInfo():
    print("Image Conversion Tool")
    print("Only support bmp/jpeg/jpg/png Format")
    print("Programmed by \033[1;31mz00418008\033[0m\n")
   
    

if __name__ == '__main__':

  
    args = Parse()
    src_name = args.src_name
    resize_w = int(args.resize_w)
    resize_h = int(args.resize_h)
    
    #command = "./transfor_nv12_bgr/SoftAipp"
    command = "./SoftAipp"
    
    Process(src_name, resize_h, resize_w, command)