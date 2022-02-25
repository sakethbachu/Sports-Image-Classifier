import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from transformers import DistilBertTokenizer
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import config as CFG
from dataset import CLIPDataset, get_transforms, CLIPDataset_classification
from CLIP import CLIPModel
from utils import AvgMeter, get_lr
from modules import finetune_network

FILE_PATH = '/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification'
TRAIN_PATH = os.path.join(FILE_PATH,'train')

TRAIN_IMAGES = os.path.join(TRAIN_PATH,'train')

def make_train_valid_dfs_fine():
    # dataframe = pd.read_csv(train_df)

    val_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(os.listdir(TRAIN_IMAGES))
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    
    return train_sampler, valid_sampler, val_indices



def build_loaders_test(tokenizer, mode, val_sampler, train_sampler):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset_classification(
        csv_file=os.path.join(FILE_PATH,'train_labels.csv'),
        tokenizer=tokenizer,
        transforms=transforms,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        sampler=train_sampler,
        
    )

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        sampler=val_sampler,
      
    )
    return train_loader, val_loader


#modifying the above defined functions to suit for the task of classifying the sports images
def train_epoch(model, train_loader, optimizer, lr_scheduler, step, loss_fn):
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()

    batch_acc = 0
    for batch in tqdm_object:
        # batch = batch.to(CFG.device)

        outputs = model(batch["image"].to(CFG.device))
        labels = batch["caption"].to(CFG.device)
        labels = labels.to(torch.float64)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)



        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()




        #using argmax for labels aswell because it is a one hot vector        
        acc_labels = torch.argmax(labels, dim=1)
        acc_outputs = torch.argmax(outputs, dim=1)
        # print(acc_labels.shape)
        # print(acc_outputs)
        #we calculate the accuracy of the predictions and return it to the main function
        accuracy = (acc_outputs == acc_labels).sum()
        accuracy = accuracy/batch["image"].size(0)
        accuracy = int(accuracy*100)

        batch_acc += accuracy


        # indices, predicted = torch.max(outputs.data, 0)
        # print(predicted.shape)
        # print(accuracy)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), batch_accuracy=accuracy)

    fin_batch = batch_acc/len(train_loader)
    print("Train epoch accuracy is:", fin_batch)
    return loss_meter, fin_batch


def valid_epoch(model, valid_loader, loss_fn):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()

    batch_acc = 0

    
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        # batch = batch.to(CFG.device)

        outputs = model(batch["image"].to(CFG.device))
        labels = batch["caption"].to(CFG.device)
        labels = labels.to(torch.float64)
        output = (torch.argmax(outputs)).float()
        accuracy = (output == labels).float().sum()


        loss = loss_fn(outputs, labels)
        acc_labels = torch.argmax(labels, dim=1)
        acc_outputs = torch.argmax(outputs, dim=1)
        # print(acc_labels.shape)
        # print(acc_outputs)
        accuracy = (acc_outputs == acc_labels).sum()
        accuracy = accuracy/batch["image"].size(0)
        accuracy = int(accuracy*100)
        batch_acc += accuracy



        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        accuracy_meter.update(accuracy, count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg, batch_accuracy=accuracy)

    fin_batch = batch_acc/len(valid_loader)
    print("Valid epoch accuracy is:", fin_batch)

    return loss_meter, fin_batch


def initi_model2(model_path):
  # tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)    
  model = CLIPModel().to(CFG.device)
  model.load_state_dict(torch.load(model_path, map_location=CFG.device))
  img_enc = model.image_encoder
  img_projector = model.image_projection

  fine_network = finetune_network().to(CFG.device)
  end2end = nn.Sequential(img_enc, img_projector, fine_network)
  return end2end



def main_finetune():
    train_sampler, valid_sampler, _ = make_train_valid_dfs_fine()

    # train_sampler = SubsetRandomSampler(short_train)
    # valid_sampler = SubsetRandomSampler(test_indices)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader, valid_loader = build_loaders_test(tokenizer, mode="train", train_sampler=train_sampler, val_sampler = valid_sampler)



    #here, we load the pretrained clip model's encoder
    model = initi_model2(model_path="/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification/models/best.pth")
    model = model.to(CFG.device)
    # params = [
    #     {"params": model.parameters(), "lr": CFG.image_encoder_lr},
    #     {"params": itertools.chain(
    #         model.image_projection.parameters(), model.text_projection.parameters()
    #     ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    # ]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr = CFG.finetune_lr, weight_decay=0.)
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    ckpt_folder = "/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification/models/"

    if os.path.exists("/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification/models/model_weights.pth"):
        checkpoint = torch.load("/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification/models/model_weights.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainLoss = checkpoint['train_loss']
        valLoss = checkpoint['val_loss']
        start_epoch = checkpoint['epoch']


        print("Loaded checkpoint from")
        print("---- Epoch: {} ----\ntrain_loss: {}\n val_loss: {}\n".format( \
                start_epoch+1, trainLoss[-1], valLoss[-1]))
        
    else:
        print("No checkpoint specified. Training from scratch")
        trainLoss = []
        valLoss = []




    step = "epoch"

    best_loss = float('inf')
    correct = 0

    train_l = []
    val_l = []
    train_a = []
    val_a = []
    for epoch in range(CFG.finetune_epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, lr_scheduler, step, loss_fn)

        train_a.append(train_accuracy)
        train_l.append(train_loss)

        trainLoss.append(train_loss)

        val_size = len(valid_loader)
        model.eval()
        with torch.no_grad():
            valid_loss, valid_accuracy = valid_epoch(model, valid_loader, loss_fn)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification/models/best_finetuned.pth")
            print("Saved Best Model!")

        val_a.append(valid_accuracy)
        val_l.append(valid_loss)
        valLoss.append(valid_loss)
        
        lr_scheduler.step(valid_loss.avg)

        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_dict': lr_scheduler.state_dict(),
            'train_loss': trainLoss,
            'val_loss': valLoss,
            }, "/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification/models/model_weights_finetuned.pth")
        

    return train_l, train_a, val_l, val_a



if __name__ == "__main__":
    train_l, train_a, val_l, val_a = main_finetune()
