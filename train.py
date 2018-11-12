from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from azureml.core.run import Run

def main(opt, run=None):

    # Get data configuration
    base_path = Path(opt.base_path)
    assert base_path.exists() and base_path.is_dir(), 'invalid data directory ({})'.format(opt.base_path)

    use_cuda = opt.use_cuda.strip().lower() in ['true', 'yes', 'ok']
    class_path = opt.class_path
    model_file = opt.model_config_path
    data_file = base_path.joinpath('trainvalno5k.txt').resolve()
    weight_file = base_path.joinpath('yolov3.weights').resolve()

    batch_size = opt.batch_size
    n_cpu = opt.n_cpu
    epochs = opt.epochs
    checkpoint_interval = opt.checkpoint_interval
    checkpoint_dir = opt.checkpoint_dir

    cuda = torch.cuda.is_available() and use_cuda

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    classes = load_classes(class_path)

    # Get hyper parameters
    hyperparams = parse_model_config(model_file)[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])

    if run != None:
        run.log('base_path', base_path)
        run.log('use_cuda', use_cuda)
        run.log('class_path', class_path)
        run.log('model_file', model_file)
        run.log('data_file', data_file)
        run.log('weight_file', weight_file)
        run.log('batch_size', batch_size)
        run.log('n_cpu', n_cpu)
        run.log('epochs', epochs)
        run.log('checkpoint_interval', checkpoint_interval)
        run.log('checkpoint_dir', checkpoint_dir)
        run.log('learning_rate', learning_rate)
        run.log('momentum', momentum)
        run.log('decay', decay)
        run.log('burn_in', burn_in)

    # Initiate model
    model = Darknet(model_file)
    # model.load_weights(weight_file)
    model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.train()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(base_path, data_file), batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    epochs,
                    batch_i,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )
            
            if run != None:
                if batch_i % 100 == 0:
                    run.log('x', model.losses["x"])
                    run.log('y', model.losses["y"])
                    run.log('w', model.losses["w"])
                    run.log('h', model.losses["h"])
                    run.log('conf', model.losses["conf"])
                    run.log('cls', model.losses["cls"])
                    run.log('total', loss.item())
                    run.log('recall', model.losses["recall"])
                    run.log('precision', model.losses["precision"])

            model.seen += imgs.size(0)

        if epoch % checkpoint_interval == 0:
            model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--base_path", type=str, default="data/coco", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="model/yolov3.cfg", help="path to model config file")
    parser.add_argument("--weights_path", type=str, default="model/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="model/coco.names", help="path to class label file")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
    )
    parser.add_argument("--use_cuda", type=str, default='false', help="whether to use cuda if available")
    opt = parser.parse_args()
    print(opt)
    try:
        run = Run.get_submitted_run()
    except:
        run = None

    main(opt, run)