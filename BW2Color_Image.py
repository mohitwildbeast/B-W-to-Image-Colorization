import numpy as np
import cv2

prototxt = 'model/colorization_deploy_v2.prototxt'
model = 'model/colorization_release_v2.caffemodel'
points = 'model/pts_in_hull.npy'
image = 'images/img01.jpg'

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId('class8_ab')
conv8 = net.getLayerId('conv8_313_rh')
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

img = cv2.imread(image)
scaled = img.astype('float32') / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224,224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original", img)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
