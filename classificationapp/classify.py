#coding=utf-8

import hiai
import imageNetClasses
import os
import numpy as np
import time
import graph
import post_process
import cv2 as cv

cur_path = os.path.dirname(__file__)
os.chdir(cur_path)
resnet18OmFileName='./models/resnet18.om'
srcFileDir = './ImageNetRaw/'
dstFileDir = './resnet18Result/'


def Resnet18PostProcess(resultList, srcFilePath, dstFilePath, fileName):
        if resultList is not None :
                firstConfidence, firstClass = post_process.GenerateTopNClassifyResult(resultList, 1)
                firstLabel = imageNetClasses.imageNet_classes[firstClass[0]]
                dstFileName = os.path.join(dstFilePath, fileName)
                srcFileName = os.path.join(srcFilePath, fileName)
                image = cv.imread(srcFileName)
                txt = firstLabel + " " + str(round(firstConfidence[0]*100,2))
                cv.putText(image, txt, (15,20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255))
                cv.imwrite(dstFileName, image)
        else :
                print('graph inference failed ')
                return None


def main():
        try:
                myGraph = graph.Graph(resnet18OmFileName)
                myGraph.CreateGraph()
        except Exception as e:
                print("Except:", e)
                return
        dvppInWidth = 224
        dvppInHeight = 224
        start = time.time()
        if not os.path.exists(dstFileDir):
                os.mkdir(dstFileDir)
        pathDir = os.listdir(srcFileDir)
        for allDir in pathDir:
                child = os.path.join(srcFileDir, allDir)
                input_image = cv.imread(child)
                input_image = cv.resize(input_image, (dvppInWidth, dvppInHeight))
                resultList = myGraph.Inference(input_image)
                if resultList is None:
                        print("graph inference failed")
                        continue
                Resnet18PostProcess(resultList, srcFileDir, dstFileDir, allDir)
        end = time.time()
        print('cost time '+str((end-start)*1000)+'ms')
        myGraph.Destroy()
        print('-------------------end')


if __name__ == "__main__":
        main()