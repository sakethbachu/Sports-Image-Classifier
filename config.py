import torch

debug = True
image_path = "C:/Users/saket/Datasets/Flicker-8k/Images"
captions_path = "C:/Users/saket/Datasets/Flicker-8k"
batch_size = 8
num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#the image encoder is a resnet50 model
model_name = 'resnet50'
#image embedding dimension

image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"

#text embedding dimension
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = False # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1

#finetuning parameters
finetune_lr = 0.00003
finetune_epochs = 10
finetune_warmup_length = 500
finetune_mode = 'freeze'