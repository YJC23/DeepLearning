import os
import pandas as pd
from collections import Counter
from pathlib import Path
import spacy

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

#location of the data 
data_location =  Path("./flickr8k") 
caption_file = data_location / "captions.txt"

#using spacy for the better text tokenization 
spacy_eng = spacy.load("en_core_web_sm") 

class Vocabulary:
    def __init__(self, freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"} # integer to string 
        
        #string to int tokens (reverse of dict self.itos)
        self.stoi = {v:k for k,v in self.itos.items()} # string to integer
        
        self.freq_threshold = freq_threshold

    def __len__(self): 
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minimum frequency threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the 'text', return its corresponding index form our vocab list """
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]   

# #testing the vocab class 
# v = Vocabulary(freq_threshold=1)
# v.build_vocab(["This is a good place to find a city"])
# print(v.stoi)
# print(v.numericalize("This is a good place to find a city here!!"))

class FlickrDataset(Dataset):
    """ Creating Flickr Dataset """
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        #Get image and caption column from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold) #Vocabulary class 
        self.vocab.build_vocab(self.captions.tolist()) #build vocab from the captions 

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)

#defining the transform to be applied
transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

def show_image(inp, title=None):
    """ Imshow for Tensor. """
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

#testing the dataset class
dataset = FlickrDataset(
    root_dir = data_location / "Images",
    captions_file = data_location / "captions.txt",
    transform = transforms
)

# img, caption = dataset[5]
# show_image(img, "Image")
# print("Token:", caption)
# print("Sentence:")
# print([dataset.vocab.itos[token] for token in caption.tolist()])

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets

def get_data_loader(dataset,batch_size,shuffle=False,num_workers=1):
    """
    Returns torch dataloader for the flicker8k dataset
    
    Parameters
    -----------
    dataset: FlickrDataset
        custom torchdataset named FlickrDataset 
    batch_size: int
        number of data to load in a particular batch
    shuffle: boolean,optional;
        should shuffle the datasests (default is False)
    num_workers: int,optional
        numbers of workers to run (default is 1)  
    """

    pad_idx = dataset.vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx,batch_first=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader

# if __name__ == '__main__': 
#     data_loader = get_data_loader(dataset=dataset, batch_size=4, shuffle=True) 
#     #generating the iterator from the dataloader
#     dataiter = iter(data_loader)

#     # getting the next batch
#     batch = next(dataiter)

#     # unpacking the batch
#     images, captions = batch

#     #showing info of image in single batch
#     for i in range(4):
#         img,cap = images[i],captions[i]
#         caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]
#         eos_index = caption_label.index('<EOS>')
#         caption_label = caption_label[1:eos_index]
#         caption_label = ' '.join(caption_label)                      
#         show_image(img, caption_label)
#         plt.show()