# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import torch
import tqdm
from dataset.imagenet_zeroshot_data import DATASET_TO_CLASSNAME, get_template
from eva_clip import get_tokenizer,create_model_and_transforms
import util.misc as misc
from dataset.dataset import get_dataset
import sys
import argparse

def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm.tqdm(classnames, disable=args.rank>0):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = torch.nn.functional.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True)) for k in topk]


def run(model, classifier, dataloader, args):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm.tqdm(dataloader, unit_scale=args.batch_size, disable=args.rank>0):
            images = images.to(args.device).half()
            target = target.to(args.device)
            image_size = images.size(0)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                # predict
                if args.distributed:
                    image_features = model.module.encode_image(images)
                    logit_scale = model.module.logit_scale
                else:
                    image_features = model.encode_image(images)
                    logit_scale = model.logit_scale
                logit_scale = logit_scale.exp()
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
                logits = logit_scale * image_features @ classifier # 100. * (image_features * mask) @ classifier #(classifier * mask.transpose(0,1))
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += image_size
            
    top1 = misc.all_reduce_mean(top1 / n)
    top5 = misc.all_reduce_mean(top5 / n)
    return top1, top5

@torch.no_grad()
def evaluate(data_loader, model, args):
    model.eval()
    CLASS_NAME = DATASET_TO_CLASSNAME[args.dataset]
    
    classifier = zero_shot_classifier(model, CLASS_NAME, get_template(args.dataset), args)
    top1, top5 = run(model, classifier, data_loader, args)

    print('* Acc@1 {top1:.5f} Acc@5 {top5:.5f} '
          .format(top1=top1, top5=top5))

    return {"acc1":top1, "acc5":top5}


def get_args_parser(args=None):
    parser = argparse.ArgumentParser('CLIP Fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--model', default='EVA02-CLIP-bigE-14-plus', type=str)
    parser.add_argument('--pretrained', default='ZSR-CLIP-bigE-14-plus.pt', type=str)
    parser.add_argument('--force_custom_clip', default=False, action='store_true')
    parser.add_argument('--grad_checkpointing', default=False, action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # dataset
    parser.add_argument('--representation', default="histogram", choices=["histogram"], type=str)
    parser.add_argument('--root', default=None, type=str)
    parser.add_argument('--input_filename', default=None, type=str)
    parser.add_argument('--dataset', default="imagenet", choices=["imagenet"], type=str)
    
    args = parser.parse_args(args)
    
    return args

if __name__ == "__main__":
    args = get_args_parser(sys.argv[1:])
    print("{}".format(args).replace(', ', ',\n'))
    
    args.distributed = False
    args.rank = -1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    model, _, transform_event = create_model_and_transforms(args.model, None, force_custom_clip=args.force_custom_clip, force_patch_dropout=None, cache_dir="./clip_model",)
    
    tokenizer = get_tokenizer(args.model)
    
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(checkpoint,strict=False)
        
    model.to(device)
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
            get_dataset(args, transform_event=transform_event, train=False), 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
        )
    evaluate(dataloader,model, args)
