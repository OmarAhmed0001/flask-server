from frames import solve
import torch
import torchvision
from torchvision import transforms, models
# from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch.autograd import Variable
import time
import os
import sys
import os
from torch import nn
from torchvision import models
from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
# from keras.models import load_model
from PIL import Image
import numpy as np
import requests
from torch import nn
from torchvision import models


import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import requests

app = Flask(__name__)

# Create a blueprint for static files in the 'uploads' folder
uploads_bp = Blueprint(
    'faces', __name__, static_url_path='/faces', static_folder='faces')
app.register_blueprint(uploads_bp)

# Create a blueprint for static files in the 'images' folder
images_bp = Blueprint('frames', __name__,
                      static_url_path='/frames', static_folder='frames')
app.register_blueprint(images_bp)
CORS(app)


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim,
                            lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

    im_size = 112


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(
    mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))


def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    # cv2.imwrite('./v2.jpg',image*255)
    return image


def predict(model, img, path='./'):
    fmap, logits = model(img.to('cpu'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item()*100
    print('confidence of prediction:',
          logits[:, int(prediction.item())].item()*100)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(
        fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255*predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :])
    result = heatmap * 0.5 + img*0.8*255
    # cv2.imwrite('v1.png',result)
    result1 = heatmap * 0.5/255 + img*0.8
    r, g, b = cv2.split(result1)
    result1 = cv2.merge((r, g, b))
    return [int(prediction.item()), confidence]


class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
        self.face_cascade = cv2.CascadeClassifier(
            './haarcascade_frontalface_default.xml')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            try:
                (x, y, w, h) = faces[0]
                frame = frame[y:y+h, x:x+w]
            except:
                pass

            frames.append(self.transform(frame))
            if (len(frames) == self.count):
                break

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# Specify the file ID of the file you want to download from Google Drive
file_id = '1Klw2YvgxCMoODEwlY0mpaApJ4Q_mVL6c'

# Make sure the URL is accessible by anyone
download_url = f'https://drive.google.com/uc?id={file_id}'

# Specify the file name to save as
file_name = 'model_84_acc_10_frames_final_data.pt'

# Check if the file already exists in the directory
if os.path.isfile(file_name):
    print(f'File "{file_name}" already exists in the directory.')
else:
    # Perform the file download
    response = requests.get(download_url)
    response.raise_for_status()

    # Save the file
    with open(file_name, 'wb') as file:
        file.write(response.content)

    print(f'Successfully downloaded file: {file_name}')

@app.route('/', methods=['POST'])
def hello():

    data = request.get_json()
    img_path = os.environ("NODE_URL")+"/deepfakeVideos/" + data['video']
    response = requests.get(img_path)

    with open('video.mp4', 'wb') as f:
        f.write(response.content)

    video_path = 'video.mp4'
    face_list, frame_list = solve(video_path, "./faces", "./frames")
    videos = []
    videos.append(video_path)

    video_dataset = validation_dataset(
        videos, sequence_length=20, transform=train_transforms)
    model = Model(2).cpu()
    path_to_model = './model_84_acc_10_frames_final_data.pt'

    model.load_state_dict(torch.load(
        path_to_model, map_location=torch.device('cpu')))
    model.eval()
    for i in range(0, len(videos)):
        print("Loading.......")
        prediction = predict(model, video_dataset[i], './')
        if prediction[0] == 1:
            return jsonify({"Message": "REAL", "face_list": face_list, "frame_list": frame_list})
        else:
            return jsonify({"Message": "FAKE", "face_list": face_list, "frame_list": frame_list})


if __name__ == '__main__':
    app.run()
