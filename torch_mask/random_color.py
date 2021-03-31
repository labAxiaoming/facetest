# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:20:52 2020

@author: xm
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:51:26 2020

@author: xm
"""
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchvision 
from torchvision import transforms as T
import random
import torch
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

#model.cuda()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img, threshold):
    #img = Image.open(img_path)
#    img = img0.resize((img0.width,img0.height)) 
#    resize(img,[400,400])
#    img = img.resize((img.width//2,img.height//2)) 

    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.cuda()
    pred = model([img])
#    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    if len(pred_score)>0:
        if len( [pred_score.index(x) for x in pred_score if x>threshold])>0:
            pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
            mask_tensor = (pred[0]['masks']>threshold)
            masks = (pred[0]['masks']>threshold).cpu().squeeze().numpy()
            
            pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
            masks = masks[:pred_t+1]
            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]
            return masks, pred_boxes, pred_class,mask_tensor
    else:
        return [],[],[],[]


def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,3)]
    print (r.size)
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask





def instance_segmentation_api(image, threshold=0.5, rect_th=3, text_size=3, text_th=3):
#    image = image.resize((image.width//2,image.height//2)) 
    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    masks, boxes, pred_cls,mask_tensor = get_prediction(image, threshold)
#    img = cv2.imread(image)
    
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(len(masks))
    for i in range(len(masks)):
        print("masks[i].size= ",masks[i].size)
        if masks[i].size == img.size//3:
            if pred_cls[i] == "person":
                rgb_mask = random_colour_masks(masks[i])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.8, 0)
#        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
#        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
#    plt.figure(figsize=(20,30))
#    plt.imshow(img)
#    plt.xticks([])
#    plt.yticks([])
#    plt.show()
    return img,masks,mask_tensor
    
    
#for i in range(10):   
#    img = Image.open("timg.jpg")
#    img,_,_= instance_segmentation_api(img)
#    cv2.imshow("img",img)
#    cv2.waitKey(10)
#
#            
#cv2.destroyAllWindows()
##    
    
    
if __name__ == "__main__":
    video_writer = cv2.VideoWriter("lj2.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25.01, (1280,720), True)

    cap = cv2.VideoCapture("lj.mp4")
    outputFile = "mask_rcnn_out_py.avi"
    while cv2.waitKey(1) < 0:
        # Get frame from the video
        hasFrame, frame = cap.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv2.waitKey(3000)
            break
        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) 
        img,masks,mask_tensor = instance_segmentation_api(image)
        video_writer.write(img)
        cv2.imshow("img",img)
        if cv2.waitKey(10)  ==27:
            break
        
    video_writer.release()
    cv2.destroyAllWindows()



#maskn = mask_tensor.cpu().squeeze().detach().numpy()
#mask_tensor.cpu().squeeze().numpy().shape
    
#    
#    
#img = Image.open("timg.jpg")
#masks, boxes, pred_cls = get_prediction(img,0.5)
#colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
#r = np.zeros_like(masks[0]).astype(np.uint8)
#g = np.zeros_like(masks[0]).astype(np.uint8)
#b = np.zeros_like(masks[0]).astype(np.uint8)
#r[masks[0] == 1], g[masks[0] == 1], b[masks[0] == 1] = colours[random.randrange(0,10)]
#coloured_mask = np.stack([r, g, b], axis=2)
#    
    
#img = cv2.imread("timg.jpg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#for i in range(len(masks)):
#    rgb_mask = random_colour_masks(masks[i])
#    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    #cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
    #cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    

#plt.figure(figsize=(20,30))
#plt.imshow(img)
#plt.xticks([])
#plt.yticks([])
#plt.show()    

    