import streamlit as st
import torch.nn as nn
from PIL import Image,ImageFilter
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import *
from torch import *
import torchvision.transforms 
st.title('Cancer detection')
class MyEnsemble(nn.Module):
    def __init__(self, alexnet, vgg):
        super(MyEnsemble, self).__init__()
        self.alexnet = alexnet
        self.lin1 = nn.Linear(9216,4096)
        self.vgg = vgg
        self.lin2 = nn.Linear(25088,4096)
        self.classifier = nn.Linear(8192, 2)
        
    def forward(self, x):
        x1 = self.alexnet(x)
        x1 = self.lin1(torch.flatten(x1,1))
        x2 = self.vgg(x)
        x2 = self.lin2(torch.flatten(x2,1))
        # x1 = self.lin1(torch.flatten(x2,1))
        x = torch.cat((x1, x2), axis = -1)
        x = self.classifier(F.relu(x))
        return x
alexnet = models.alexnet(True)
alexnet= nn.Sequential(*list(alexnet.children())[:-1])
vgg16 = models.vgg16(True)
vgg16= nn.Sequential(*list(vgg16.children())[:-1])
TLModel = MyEnsemble(alexnet,vgg16)#.to(torch.device('cpu'))
TLModel.load_state_dict(torch.load("Downloads/max_specificityTL.pt",map_location=torch.device('cpu')))
st.markdown('model loaded')
file_type = 'tif'
uploaded_file = st.file_uploader("Choose a  file",type = file_type)
if uploaded_file != None:
    image = Image.open(uploaded_file)
    _, col2,_ = st.columns(3)
    with col2:
        st.image(image.resize((240, 240)),caption = "Input image (resized to 240x240 px).")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = transform(image).view(1,3,96,96)
    outputs = TLModel(image)
    proba = flatten(F.softmax(outputs.data),1).view(2,1)
    _, predicted = torch.max(outputs.data, 1)
    if proba[0] >= 0.5:
        st.markdown('<h4 style = "text-align:center">The tissue image seems to be not cancerous with probability %.3f.<h4>'%(proba[0].item()),unsafe_allow_html = True)
    else:
        st.markdown(f'<h4 style = "text-align:center">The tissue image is cancerous with probability %.3f.<h4>'%(proba[0].item()),unsafe_allow_html = True)
