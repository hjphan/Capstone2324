import cv2
import numpy as np
#from google.colab.patches import cv2.imshow
net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_training.cfg')
classes = []

with open('classesv1.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.imread("images/image_11.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            #print(detection)
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
print(confidences)
print(class_ids)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.2)
#indexes = [0,1,2,3,4]
print(indexes)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        #label = str(classes[class_ids[i]] + " C: "+f"{confidences[i]:.2f}")
        label = str("C: "+f"{confidences[i]:.2f}")
        #label = str(i)
        print(classes[class_ids[i]])
        print(confidences[i])
        print(i)
        print(class_ids)
        #print(class_ids[i])
        #color = colors[i]
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x-50, y + 90), font, 2, color, 3)
        
cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()