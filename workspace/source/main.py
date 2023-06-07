import cv2
import numpy as np
import copy
import dlib
from pathlib import Path
from ThreadPool import ThreadPool
import logging
from logging import config
import time

# logging configuration
log_config = {
    "version":1,
    "root":{
        "handlers" : ["console", "file"],
        "level": "DEBUG"
    },
    "handlers":{
        "console":{
            "formatter": "std_format",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        },
        "file":{
            "formatter":"std_format",
            "class":"logging.FileHandler",
            "level":"DEBUG",
            "filename":"logOutput.log",
            "mode":"w"
        }
    },
    "formatters":{
        "std_format": {
            "format"  : "%(asctime)s : %(threadName)s : %(levelname)s : %(message)s", 
            "datefmt" : "%Y-%m-%d %H:%M:%S"
        }
    },
}

config.dictConfig(log_config)
logger = logging.getLogger(__name__)

# Globals
BASE_DIR   = Path.home()/"workspace"
DATA_DIR   = BASE_DIR/"data"
VIDEO_DIR  = DATA_DIR/"video"
MESH_DIR   = DATA_DIR/"mesh"
FRAMES_DIR = DATA_DIR/"frames"
VERTICES_DIR = DATA_DIR/"vertices"
MODEL_PATH   = "shape_predictor_68_face_landmarks.dat"
DELAUNAY_COLOR = (255,255,255)
POINTS_COLOR  = (0, 0, 255)
NUM_WORKERS = 3

class ProcessVideo:
    def __init__(self, videoPath, videoIDStr, threadPool):
        self.threadPool = threadPool
        self.videoPath = videoPath
        self.videoIDStr = videoIDStr
        self.fps = 0
        self.width = 0
        self.height = 0
        self.extractedFrames = 0

    def extractFramesFromVideo(self):
        capture = cv2.VideoCapture(self.videoPath)
        #create a new folder 
        outputFramesPath = FRAMES_DIR/self.videoIDStr
        outputFramesPath.mkdir(parents=True, exist_ok=True)
        success = 1

        self.width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps    = capture.get(cv2.CAP_PROP_FPS)
        logger.debug(f"Saving frames to folder {str(outputFramesPath)}")
        logger.debug(f"Video metadata width:{self.width} height:{self.height} fps:{self.fps}")
        while success:
            success, frame = capture.read()
            if success:
                frameNumStr =  "frame" + str(self.extractedFrames).zfill(5)
                filename = outputFramesPath/f"frame{frameNumStr}.jpg"
                cv2.imwrite(str(filename), frame)
                self.extractedFrames += 1


        capture.release()
        logger.debug(f"Extracted {self.extractedFrames} frames from video")

    def rect_contains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    def drawDeleaunay(self, image, subdiv):
        triangleList = subdiv.getTriangleList()
        size = image.shape
        r = (0, 0, size[1], size[0])

        for t in triangleList :
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            if self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3):
                cv2.line(image, pt1, pt2, DELAUNAY_COLOR, 1)
                cv2.line(image, pt2, pt3, DELAUNAY_COLOR, 1)
                cv2.line(image, pt3, pt1, DELAUNAY_COLOR, 1)
    
    def writeFaceLandmarksToFile(self, detectedLandmarks, filePath):
        with open(filePath, 'w') as f:
            for p in detectedLandmarks.parts():
                f.write("%s %s\n" %(int(p.x),int(p.y)))


    def generateFramesInRange(self, imageList, startIdx, endIdx):
        logger.info(f"Worker genarting delaunay triangles for frames ranging {startIdx} - {endIdx}")
        for i in range(startIdx, endIdx):
            imagePath = FRAMES_DIR/self.videoIDStr/imageList[i]
            image = cv2.imread(str(imagePath))
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faceLandmarkDetector = dlib.shape_predictor(MODEL_PATH)
            frontalFaceDetector = dlib.get_frontal_face_detector()
            allFaces = frontalFaceDetector(imageRGB, 0)
            allFacesLandmark = []
            points = []
            
            # genrate vertices
            logger.debug(f"Worker genarting vertices for frame {i}")
            verticesfilePath = VERTICES_DIR/self.videoIDStr/f"imageframe{str(i).zfill(5)}.txt"
            for k in range(0, len(allFaces)):
                # dlib rectangle class will detecting face so that landmark can apply inside of that area
                faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
                int(allFaces[k].right()),int(allFaces[k].bottom()))
                detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
                for p in detectedLandmarks.parts():
                    points.append((int(p.x), int(p.y)))
                # Svaing the landmark one by one to the output folder
                allFacesLandmark.append(detectedLandmarks)
        
                # Write landmarks to disk
                self.writeFaceLandmarksToFile(detectedLandmarks, verticesfilePath)
            
            # generate deleaunay mesh frames
            size = image.shape
            rect = (0, 0, size[1], size[0])
            subdiv = cv2.Subdiv2D(rect)
            for p in points :
                subdiv.insert(p)
            
            img_copy = image.copy()
            logger.debug(f"Worker genarting delaunay for frame {i}")
            self.drawDeleaunay(img_copy, subdiv)
            outputMeshPath = MESH_DIR/self.videoIDStr/f"imageframe{str(i).zfill(5)}.jpg"

            cv2.imwrite(str(outputMeshPath), img_copy)
            logger.debug(f"Worker finished  generating & writing delaunay for frame {i}")
        logger.info(f"Worker finished generating delaunay triangles for frames ranging {startIdx} - {endIdx}")

    def generateDelaunayFrames(self):
        # create folders as they might not exist
        verticesDir = VERTICES_DIR/self.videoIDStr
        verticesDir.mkdir(parents=True, exist_ok=True)
        meshDir = MESH_DIR/self.videoIDStr
        meshDir.mkdir(parents=True, exist_ok=True)

        frameDir = FRAMES_DIR/self.videoIDStr
        imageList = sorted(frameDir.glob('*.jpg'))
        start_idx = 0
        end_idx   = 0
        lambdaList = [None] * NUM_WORKERS
        s = [None] * (NUM_WORKERS - 1)
        e = [None] * (NUM_WORKERS - 1)
        logger.info(f"Distributing {self.extractedFrames} among {NUM_WORKERS}")
        for i in range(NUM_WORKERS - 1):
            end_idx +=  self.extractedFrames // NUM_WORKERS
            if end_idx != 0:
                s[i] = copy.deepcopy(start_idx)
                e[i] = copy.deepcopy(end_idx)
                lambdaList[i] = lambda threadName, threadId: self.generateFramesInRange(imageList, s[i], e[i])
                self.threadPool.add_task(lambdaList[i])
                time.sleep(20/1000)
            start_idx += self.extractedFrames // NUM_WORKERS
        end_idx = self.extractedFrames
        lambdaList[NUM_WORKERS - 1] = lambda threadName, threadId:self.generateFramesInRange(imageList, start_idx, end_idx)
        self.threadPool.add_task(lambdaList[NUM_WORKERS - 1])
        logger.info("Waiting for workers to complete the task")
        self.threadPool.waitForAllTasksCompleted()
    
    def generateOutputVideo(self):
        videofilename = "output_" + "output_testVid01" + ".mkv" #self.videoIDStr
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        videoPath = VIDEO_DIR/videofilename
        meshframesPath = str( MESH_DIR+"/"+"output_testVid01") #self.videoIDStr
        imageList = sorted(meshframesPath.glob('*.jpg'))

        logger.info(f"Writing output video to file: {str(videoPath)}")
        print(meshframesPath)
        temp = cv2.imread(str( meshframesPath/imageList[0]))
        height, width, layers = temp.shape
        videoObj = cv2.VideoWriter(str(videoPath), 0, 10.3, (width,height))
        numFrames = len(imageList)
        count = 1
        for image in imageList:
            imagePath = meshframesPath/image
            img = cv2.imread(str(imagePath))
            videoObj.write(img)
            logger.debug(f"videoWriter written {count} out of {numFrames} to the file")
            count +=1
        videoObj.release()

    def deinit(self):
        self.threadPool.shutDownPool()

def main():
    logger.info(f"Initializing Threadpool with {NUM_WORKERS} workers")
    
    threadPool = ThreadPool(NUM_WORKERS)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    numvidoes = len(list(VIDEO_DIR.glob('*.mkv')))
    videoIDStr = "testVid"+str(numvidoes + 1).zfill(2)
    videoPath = "/Users/deepikasulakhe/Downloads/dipuSelfie.mp4"
    videoProcessor = ProcessVideo(videoPath, videoIDStr, threadPool)
    
    logger.info(f"Extracting frames from vido: {str(videoPath)} assigned videoID: {videoIDStr}")
    videoProcessor.extractFramesFromVideo()
    logger.info("Completed frames extraction")
    
    logger.info("Start generating delaunay triangles")
    videoProcessor.generateDelaunayFrames()
    logger.info("Completed generating delaunay triangles")
    
    logger.info("Start writing outputvideo")
    videoProcessor.generateOutputVideo()
    logger.info("Finished writing output video")

    logger.info("Shutting down threadpool")
    videoProcessor.deinit()
    logger.info("Completed!")



if __name__ == "__main__":
    main()



