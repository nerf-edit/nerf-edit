import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import sys
sys.path.append('/home/sarthakbhagat/code/nerf-edit')
from lang_seg import utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import test_batchify_fn 
from encoding.models.sseg import BaseNet
from lang_seg.modules.lseg_module import LSegModule
from lang_seg.utils import Resize
import cv2
import math
import types
import functools
import torchvision.transforms as torch_transforms
import copy
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import clip
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from lang_seg.data import get_dataset
from lang_seg.additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule
import torchvision.transforms as transforms
from skimage.transform import resize

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ade20k",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--weights", type=str, default='/data/sarthakbhagat/demo_e200.ckpt', help="checkpoint to test"
        )
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )

        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )
        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )
        # test option
        parser.add_argument(
            "--data-path", type=str, default=None, help="path to test image folder"
        )
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )
        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )
        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
        )
        parser.add_argument(
            "--jobname",
            type=str,
            default="default",
            help="select which dataset",
        )
        parser.add_argument(
            "--no-strict",
            dest="strict",
            default=True,
            action="store_false",
            help="no-strict copy the model",
        )
        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )
        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )
        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

def save_panoptic(predictions, text_feats, out_filename):

    with gzip.open(out_filename, "wb") as fid:
        torch.save(
            {
                "mask": torch.argmax(predictions, dim=1),
                "segments": torch.argmax(predictions, dim=1),
                #"mask_notta": mask_notta,
                #"segments_notta": segments_notta,
                #"confidences_notta": confidences_notta,
                "probabilities": predictions,
                # "confidences": confidence,
                "text_feats": text_feats
            }, fid
        )

def get_text_features(out_filename):
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count() 

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path='/data/sarthakbhagat/',
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=args.arch_option,
        strict=args.strict,
        block_depth=args.block_depth,
        activation=args.activation,
    )
    input_transform = module.val_transform
    num_classes = module.num_classes

    # dataset
    testset = get_dataset(
        args.dataset,
        root='/data/sarthakbhagat/',
        split="val",
        mode="testval",
        transform=input_transform,
    )

    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
    )
    test_data = data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=test_batchify_fn,
        **loader_kwargs
    )

    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval()
    model = model.cpu()

    # print(model)
    # if args.acc_bn:
    #     from encoding.utils.precise_bn import update_bn_stats

    #     data_kwargs = {
    #         "transform": input_transform,
    #         "base_size": args.base_size,
    #         "crop_size": args.crop_size,
    #     }
    #     trainset = get_dataset(
    #         args.dataset, split=args.train_split, mode="train", **data_kwargs
    #     )
    #     trainloader = data.DataLoader(
    #         ReturnFirstClosure(trainset),
    #         root='/data/sarthakbhagat/',
    #         batch_size=args.batch_size,
    #         drop_last=True,
    #         shuffle=True,
    #         **loader_kwargs
    #     )
    #     print("Reseting BN statistics")
    #     model.cuda()
    #     update_bn_stats(model, trainloader)

    if args.export:
        torch.save(model.state_dict(), args.export + ".pth")
        return

    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )  

    evaluator = LSeg_MultiEvalModule(
        model, num_classes, scales=scales, flip=True
    ).cuda()
    evaluator.eval()
    print ('model======>', model)

    # metric = utils.SegmentationMetric(testset.num_class)
    tbar = tqdm(test_data)

    f = open("logs/log_test_{}_{}.txt".format(args.jobname, args.dataset), "a+")
    per_class_iou = np.zeros(testset.num_class)
    cnt = 0
    # for i, (image, dst) in enumerate(tbar):
    
    img_path = './lseg_test.jpg'
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.open(img_path)
    image = np.array(image)
    # image = image.resize(image, 480)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0).cuda()
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5

    transform = transforms.Resize((480, 480))
    image = transform(image)

    label_src = 'plant,grass,cat,stone,other'

    labels = []
    print('** Input label value: {} **'.format(args.label_src))
    lines = args.label_src.split(',')
    for line in lines:
        label = line
        labels.append(label)
    
    with torch.no_grad():

        print ('image shape==',image.shape)
        # predicts = evaluator.parallel_forward(image)[0]
        predicts, text_feats = model(image)
        print ('pixel wise class prediction==',predicts.shape)
        print ('text features==',text_feats.shape)
            
        # metric.update(dst, predicts)
        # pixAcc, mIoU = metric.get()
        
        # _, _, total_inter, total_union = metric.get_all()
        # per_class_iou += 1.0 * total_inter / (np.spacing(1) + total_union)
        # cnt+=1
        
        # tbar.set_description("pixAcc: %.4f, mIoU: %.4f" % (pixAcc, mIoU))
    
    save_panoptic(predicts, text_feats, out_filename + ".ptz")

    # if args.eval:
    #     each_classes_iou = per_class_iou/cnt
    #     print("pixAcc: %.4f, mIoU: %.4f" % (pixAcc, mIoU))
    #     print(each_classes_iou)
    #     f.write("dataset {} ==> pixAcc: {:.4f}, mIoU: {:.4f}\n".format(args.dataset, pixAcc, mIoU))
    #     for per_iou in each_classes_iou: f.write('{:.4f}, '.format(per_iou))
    #     f.write('\n')


class ReturnFirstClosure(object):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        outputs = self._data[idx]
        return outputs[0]


if __name__ == "__main__":
    get_text_features('./')