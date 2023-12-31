import torch as T
import lovely_tensors as lj
import torch.distributed as dist
import torchvision.transforms as tr

from datasets import load_dataset
from tqdm.auto import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import HFDataset
from backbone import ImageBackbone
from unet import Generator
from loss import DTLoss


lj.monkey_patch()


def parse_args():
    p = ArgumentParser()
    p.add_argument("--lr", default=1e-3, type=float, required=False)
    p.add_argument("--num_epochs", default=300, type=int, required=False)
    p.add_argument("--device", default="cuda:1", type=str, required=False)
    p.add_argument("--batch_size", default=32, type=int, required=False)
    p.add_argument("--gamma", default=10, type=float, required=False)
    p.add_argument("--image_size", default=256, type=int, required=False)
    return p.parse_args()


def train():
    args = parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % T.cuda.device_count()

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

    dataset = load_dataset("./imagenet")

    imagenet_train = HFDataset(dataset["train"], train_transforms)
    imagenet_valid = HFDataset(dataset["validation"], valid_transforms)

    trainsampler = T.utils.data.distributed.DistributedSampler(
        imagenet_train,
        num_replicas=T.cuda.device_count(),
        rank=rank,
    )

    trainloader = T.utils.data.DataLoader(
        imagenet_train, batch_size=args.batch_size, pin_memory=True, sampler=trainsampler
    )
    validloader = T.utils.data.DataLoader(
        imagenet_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    device = T.device(args.device)
    loss_fn = DTLoss(args.gamma)
    
    g = Generator().to(device_id)
    f = ImageBackbone([512, 512, 1024, 1024], args.image_size).to(device_id)

    g = DDP(g, device_ids=[device_id])
    f = DDP(f, device_ids=[device_id])

    optimizer = T.optim.SGD(f.parameters(), lr=args.lr, nesterov=True, momentum=0.99)

    run_name = "DT-Exp2"
    writer = SummaryWriter(f"./logs/{run_name}")

    global_iters = 0
    for e in range(args.num_epochs):
        for images, _ in tqdm(trainloader):
            with T.autocast(device_type="cuda", dtype=T.bfloat16):
                images = images.to(device)
    
                f_r_l = f(images)
                f_g_l = g(images)
    
                loss = loss_fn(f_g_l, f_r_l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                writer.add_scalar("monitor/loss", loss.item(), global_iters)
                global_iters += 1

    if rank == 0:
        T.save(f.feature_extractor.state_dict(), "./model.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    train()