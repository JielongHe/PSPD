import argparse
import gc
import utils
from ruamel.yaml import YAML
from data import create_dataset, create_loader
from gen_model.blip import blip_decoder
import torch
from utils import cosine_lr_schedule
import os


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)

        loss = model(image, caption)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return model




def train_caption(configs, device, train_loader, sava_path):

    img_size = configs['image_size']

    generating_model = blip_decoder(pretrained=configs['pretrained'], image_size=img_size, vit=configs['vit'],
                                    vit_grad_ckpt=configs['vit_grad_ckpt'], vit_ckpt_layer=configs['vit_ckpt_layer'],
                                    prompt=configs['prompt'])
    generating_model = generating_model.to(device)

    optimizer = torch.optim.AdamW(params=generating_model.parameters(), lr=configs['init_lr'], weight_decay=configs['weight_decay'])

    for epoch in range(0, configs['max_epoch']):

        cosine_lr_schedule(optimizer, epoch, configs['max_epoch'], configs['init_lr'], configs['min_lr'])

        generating_model = train(generating_model, train_loader, optimizer, epoch, device)

        generating_model.eval()
        print("Evaluating model on validation set...")
        output_dir  = 'checkpoint'
        model_save_path = os.path.join(output_dir, sava_path)
        torch.save(generating_model.state_dict(), model_save_path)




def create_and_load_datasets(config):


    cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset = create_dataset(config['pre_train_dataset'], config)
    sampler = [None, None, None]
    cuhk_pre_loader, rstp_pre_loader, icfg_pre_loader = create_loader(
        [cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset ], sampler,
        batch_size=[configs['batch_size']] * 3,
        num_workers=[4, 4, 4],
        is_trains=[False, False, False],
        collate_fns=[None, None, None]
    )

    del cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return cuhk_pre_loader, rstp_pre_loader, icfg_pre_loader


def main(configs):
    device = 'cuda'

    cuhk_pre_loader, rstp_pre_loader, icfg_pre_loader = create_and_load_datasets(configs)

    # train_caption(configs, device, cuhk_pre_loader, 'cuhk_best_model_epoch.pth')

    train_caption(configs, device, rstp_pre_loader, 'rstp_best_model_epoch.pth')

    train_caption(configs, device, icfg_pre_loader, 'icfg_best_model_epoch.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='./configs/caption.yaml')
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    yaml = YAML(typ='safe')

    with open(args.configs, 'r') as file:
        configs = yaml.load(file)


    main(configs)