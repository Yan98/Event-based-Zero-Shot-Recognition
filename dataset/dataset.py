#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
from .imagenet_zeroshot_data import IMAGENET_CLASS
from .event_utlities import event_historgram, slice_event, reshape_event_no_sample
import torchvision
from functools import partial
from PIL import Image

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, train = True, representation="histogram", transform_event=None, args=None):
        super(BaseDataset, self).__init__()
        
        if representation == "histogram":
            self.representation = partial(event_historgram, remove_hot_pixel=True)
            
            to_rgb = _convert_to_rgb
            if transform_event is not None:
                normalize = transform_event.transforms[-1]            
        else:
            raise SystemExit(f"Unknown representation: {representation}")
            
  
        self.event_transforms = torchvision.transforms.Compose([
            to_rgb,
            transform_event.transforms[0],
            transform_event.transforms[1],
            normalize, 
            ])
     
    def __len__(self):
        return len(self.file) 
      

class ImageNetDataset(BaseDataset):
    def __init__(self, root, input_filename, train = True, use_sdb = False, image_size = [224,224], representation="histogram", transform_event=None, args=None):
        
        self.TIME_SCALE = 1000000   
        self.SENSOR_H = 480
        self.SENSOR_W = 640
        self.IMAGE_H, self.IMAGE_W = image_size
        self.slice_augment_width=2000
        
        if representation == "histogram":
            self.event_length = 15000
            
        super().__init__(train=train, representation=representation, transform_event=transform_event, args=args)
        self.file = [os.path.join(root, i.strip()) for i in open(input_filename, 'r').readlines()]
        self.args = args
            

        
    def load_event(self,event_path):
        event = np.load(event_path)
        if "event_data" in event:
            event = event['event_data']
            event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
        else:
            event = np.vstack([event['x_pos'], event['y_pos'], event['timestamp'], event['polarity'].astype(np.uint8)]).T
    
        event = event.astype(np.float32)
        event[:, 2] /= self.TIME_SCALE
    
        # Account for zero polarity
        if event[:, 3].min() >= -0.5:
            event[:, 3][event[:, 3] <= 0.5] = -1
            
        event = torch.from_numpy(event)
        event = slice_event(event, self.event_length, self.slice_augment_width)
        event = reshape_event_no_sample(event, self.SENSOR_H, self.SENSOR_W, self.IMAGE_H, self.IMAGE_W)
        return event
            
    def __getitem__(self, idx):
        event_path = self.file[idx]
        
        label = IMAGENET_CLASS[event_path.split("/")[-2]]
        event = self.load_event(event_path)
        event = self.representation(event,[self.IMAGE_H,self.IMAGE_W])
        event = self.event_transforms(event)
        return event, torch.FloatTensor([int(label[0]) - 1])

    def __len__(self):
        return len(self.file)    


def _convert_to_rgb(image):
    image_rgb = image.mean(0,True).expand(3,-1,-1)
    return image_rgb.float()


def get_dataset(args,transform_event, train = True, val=True):
    val_dataset = ImageNetDataset(
                    root = args.root,
                    input_filename=args.input_filename,
                    train = False,
                    transform_event=transform_event,
                    representation=args.representation,
                    )
    
    return val_dataset
    
def load_and_preprocess(event_path, 
                        SENSOR_H,
                        SENSOR_W,
                        event_length,
                        representation,
                        event_viz_path,
                        preprocess,
                        ):
    if event_path.endswith(".npz"):
        event = np.load(event_path)
        if "event_data" in event:
            event = event['event_data']
            event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
        else:
            event = np.vstack([event['x_pos'], event['y_pos'], event['timestamp'], event['polarity'].astype(np.uint8)]).T
    else:
        raise SystemExit("Unknown event file type")
    event = event.astype(np.float32)
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1
        
    event = torch.from_numpy(event)
    event = slice_event(event, event_length, 0)
    if representation == "histogram":
        representation = partial(event_historgram, remove_hot_pixel=True)
        to_rgb = _convert_to_rgb
        normalize = preprocess.transforms[-1]            
    else:
        raise SystemExit(f"Unknown representation: {representation}")
        
    preprocess = torchvision.transforms.Compose([
        to_rgb,
        preprocess.transforms[0],
        preprocess.transforms[1],
        normalize, 
        ])
    event = representation(event,[SENSOR_H,SENSOR_W])
    if event_viz_path is not None:
        viz = event.mean(0)
        viz = (viz - viz.min()) / (viz.max() - viz.min())
        viz = (viz.numpy() * 255).astype(np.uint8)
        Image.fromarray(viz).save(event_viz_path)
    return preprocess(event)

    
