import torch as T
import torchvision
import torchvision.transforms as tr

from tqdm.auto import tqdm
from argparse import ArgumentParser

from backbone import ImageBackbone
from unet import Generator
from loss import DTLoss


def parse_args():
    p = ArgumentParser()
    p.add_argument("--lr", default=1e-3, type=float, required=False)
    p.add_argument("--num_epochs", default=300, type=int, required=False)
    p.add_argument("--device", default="cuda:1", type=str, required=False)
    p.add_argument("--batch_size", default=8, type=int, required=False)
    p.add_argument("--gamma", default=10, type=float, required=False)
    p.add_argument("--image_size", default=256, type=int, required=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_transforms = tr.Compose([
        tr.CenterCrop(args.image_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Lambda(lambda t: (t * 2) - 1),
    ])
    valid_transforms = tr.Compose([
        tr.CenterCrop(args.image_size),
        tr.ToTensor(),
        tr.Lambda(lambda t: (t * 2) - 1),
    ])

    imagenet_train = torchvision.datasets.ImageNet("./imagenet", split="train", transform=train_transforms)
    imagenet_valid = torchvision.datasets.ImageNet("./imagenet_valid", split="val", transform=valid_transforms)

    trainloader = T.utils.data.DataLoader(
        imagenet_train, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    validloader = T.utils.data.DataLoader(
        imagenet_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    device = T.device(args.device)
    loss_fn = DTLoss(args.gamma)
    
    g = Generator()
    f = ImageBackbone([512, 512, 1024, 1024], args.image_size)

    g.to(device)
    f.to(device)

    optimizer = T.optim.SGD(f.parameters(), lr=args.lr, nesterov=True, momentum=0.99)

    for e in range(args.num_epochs):
        for images, _ in tqdm(trainloader):
            images = images.to(device)

            f_r_l = f(images)
            f_g_l = g(images)

            loss = loss_fn(f_g_l, f_r_l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()