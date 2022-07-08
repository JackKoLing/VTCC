import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, network, contrastive_loss, vtcc
from utils import yaml_config_hook, save_model
from torch.utils import data


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "RSOD":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/RSOD',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 4
    elif args.dataset == "UC-Merced":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/UC-Merced',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 21
    elif args.dataset == "SIRI-WHU":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/SIRI-WHU',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 12
    elif args.dataset == "AID":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/AID',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 30
    elif args.dataset == "D0":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/D0',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 40
    elif args.dataset == "DTD":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/DTD',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 47
    elif args.dataset == "Chaoyang":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/Chaoyang',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 4
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="./datasets",
            download=True,
            train=True,
            transform=transform.Augmentation(size=args.image_size),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./datasets",
            download=True,
            train=False,
            transform=transform.Augmentation(size=args.image_size),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    else:
        raise NotImplementedError
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
   
    vtcc = vtcc.vit_small()
    model = network.Network_VTCC(vtcc, args.feature_dim, class_num)
    model = model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.reload:
        print("reload training.")
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 100 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
