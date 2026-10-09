# CT2: train.py + local spatial consistency. Only the sampler and the loss change; the parser, model,
# optimizer and training loop are train.py's own, so the run stays comparable to the baseline.
import os
import torch
import torch.backends.cudnn as cudnn
import train as T
from datasets.make_dataloader import make_dataset
from datasets.Dataloader_University import train_collate_fn
from losses.loss import Loss
from models.taskflow import make_model
from optimizers.make_optimizer import make_optimizer
from tool.utils import copyfiles2checkpoints, set_seed
from geo_utils import load_gps, class_xy, GeoSampler, local_consistency_loss

LC_WEIGHT = 1.0


class GeoLoss(Loss):
    def __init__(self, opt, xy):
        super().__init__(opt)
        self.xy = xy

    def forward(self, outputs, outputs2, labels, labels2):
        loss, cls_loss, triplet_loss, kl_loss = super().forward(outputs, outputs2, labels, labels2)
        lc = LC_WEIGHT * local_consistency_loss(torch.cat((outputs[1], outputs2[1])),
                                                self.xy[torch.cat((labels, labels2))])
        # train.py logs four fixed numbers, so the LC term is reported inside "Triplet_Loss"
        return loss + lc, cls_loss, triplet_loss + lc, kl_loss


if __name__ == '__main__':
    set_seed(666)
    opt = T.get_parse()
    opt.use_gpu = torch.cuda.is_available()
    torch.cuda.set_device(int(opt.gpu_ids.split(',')[0]))
    cudnn.benchmark = True

    loader, class_names, dataset_sizes = make_dataset(opt)
    xy = class_xy(class_names, load_gps(os.path.dirname(opt.data_dir.rstrip('/'))))
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=opt.batchsize, sampler=GeoSampler(xy),
                                         num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    opt.nclasses = len(class_names)
    model = make_model(opt)
    optimizer, scheduler = make_optimizer(model, opt)
    model = model.cuda()
    copyfiles2checkpoints(opt)
    xy_t = torch.tensor(xy, dtype=torch.float32).cuda()
    T.Loss = lambda o: GeoLoss(o, xy_t)  # train_model builds its loss with Loss(opt)
    T.train_model(model, opt, optimizer, scheduler, loader, dataset_sizes)
