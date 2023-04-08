#pip3 install "clip-by-openai"

import pandas as pd
import os
img_path="../input/sports-classification/train/"
rice_label=os.listdir("../input/sports-classification/train/")
img_list = []
label_list = []
caption_number=[]
id_=[]
id_count=0
caption_number_count=0
for label in rice_label:
    for img_file in os.listdir(img_path+label):
        img_list.append((img_path+label+'/'+img_file))#.split("/")[-1])
        label_list.append(label)
        caption_number.append(caption_number_count)
        id_.append(id_count)
        caption_number_count+=1
    id_count+=1
    caption_number_count=0
        
df = pd.DataFrame({'image':img_list,"caption_number":caption_number, 'caption':label_list,"id":id_})
img_path="../input/sports-classification/valid/"
rice_label=os.listdir("../input/sports-classification/valid/")
img_list_val = []
label_list_val = []
for label in rice_label:
    for img_file in os.listdir(img_path+label):
        img_list_val.append((img_path+label+'/'+img_file))#.split("/")[-1])
        label_list_val.append(label)
df_val = pd.DataFrame({'image':img_list_val,'caption':label_list_val})


import clip
from PIL import Image
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

class image_title_dataset(Dataset):
    def __init__(self, list_image_path,list_txt):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image,title

# use your own data
dataset = image_title_dataset(img_list,label_list)
train_dataloader = DataLoader(dataset,batch_size = 64) #Define your own dataloader

dataset_val = image_title_dataset(img_list_val,label_list_val)
train_dataloader_df_val = DataLoader(dataset,batch_size = 64) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
    model.float()
else :
    clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.
for epoch in range(10):
    #break
    print("Here is epoch",epoch)
    model.train()
    total_lo=[]
    for batch in train_dataloader :

        optimizer.zero_grad()

        images,texts = batch 

        images= images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_lo.append(float(total_loss))
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        # Evaluate the model
    print("Train loss",np.mean(total_lo))
    print("-----------------------------------------------")
    print("valuation time")
    print("-----------------------------------------------")
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        total_lo=[]
        for batch in train_dataloader_df_val:

            images,texts = batch 
            images= images.to(device)
            texts = texts.to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_lo.append(float(total_loss))
            _, predicted = torch.max(logits_per_image, dim=1)
            total_correct += (predicted == torch.arange(predicted.size(0)).to(device)).sum().item()
            total_samples += predicted.size(0)
    print("Test loss",np.mean(total_lo))
    accuracy = total_correct / total_samples
    print(f"Epoch {epoch}, Test accuracy: {accuracy:.4f}")
    print("-----------------------------------------------")

torch.save(model.state_dict(), 'clip_mode5.pt')

