#coding=utf-8

import hiai
import imageNetClasses
import os
import numpy as np
import time
import graph
import post_process
import cv2 as cv


resnet18OmFileName='./models/resnet18.om'
srcFileDir = './ImageNetRaw/'
dstFileDir = './resnet18Result/'


def Resnet18PostProcess(resultList, srcFilePath, dstFilePath, fileName):
        if resultList is not None :
			firstConfidence, firstClass = post_process.GenerateTopNClassifyResult(resultList, 1)
			firstLabel = imageNetClasses.imageNet_classes[firstClass[0]]
			dstFileName = os.path.join('%s%s' % (dstFilePath, fileName))
			srcFileName = os.path.join('%s%s' % (srcFilePath, fileName))
			image = cv.imread(srcFile)
			cv.putText(image, text, (15,20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255) )
			cv.imwrite(dstFile, image)
        else :
                print('graph inference failed ')
                return None


def main():
        myGraph = graph.Graph(resnet18OmFileName)
        myGraph.CreateGraph()
        if myGraph is None :
                print("CreateGraph failed")
                return None
        dvppInWidth = 224
        dvppInHeight = 224
        start = time.time()
        jpegHandler.mkdirown(dstFileDir)
        pathDir =  os.listdir(srcFileDir)
        for allDir in pathDir :
                child = os.path.join('%s%s' % (srcFileDir, allDir))
                if( not jpegHandler.is_img(child) ):
                        print('[info] file : ' + child + ' is not image !')
                        continue
                input_image = cv.imread(child)
                input_image = cv.resize(input_image, (dvppInWidth, dvppInHeight))
                input_image = cv.cvtColor(input_image, cv.COLOR_BGR2YUV_I420)
                resultList = myGraph.Inference(input_image)
                if resultList is None :
                        print("graph inference failed")
                        continue
                Resnet18PostProcess(resultList, srcFileDir, dstFileDir, allDir)
        end = time.time()
        print('cost time ' + str((end-start)*1000) + 'ms')
        myGraph.Destroy()
        print('-------------------end')


if __name__ == "__main__":
        main()