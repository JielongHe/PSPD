import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
# from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from models.pspd import build_model
import utils
from utils import cosine_lr_schedule, cos_with_warmup_lr_scheduler
from data import create_dataset, create_sampler, create_loader
import seaborn as sns

@torch.no_grad()
def evaluation(image_ids, txt_ids, model, data_loader, device, config, itm=False):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []



    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=73,
                                     return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
        # image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = text_embeds @ image_embeds.t()

    txt2pid = data_loader.dataset.txt2pid

    img2pid = data_loader.dataset.img2pid

    txt2pid = torch.tensor(txt2pid)
    img2pid = torch.tensor(img2pid)

    positive_mask = (txt2pid.unsqueeze(1) == img2pid.unsqueeze(0)).float().to('cuda')

    dia_t2i_similarity = sims_matrix * positive_mask

    t2i_average_similarity = dia_t2i_similarity.sum() / positive_mask.sum()
    t2i_average_similarity = t2i_average_similarity.item()
    print(t2i_average_similarity)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)):
        if itm == True:
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
            encoder_output = image_feats[topk_idx.cpu()].to(device)
            bs = encoder_output.shape[0]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                        attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        )

            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]

            score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    score_matrix_t2i = score_matrix_t2i + sims_matrix

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_t2i, txt2img, img2txt, img2pid, txt2pid):  # txt2img: corresponding id

    img2pid = np.asarray(img2pid)
    txt2pid = np.asarray(txt2pid)

    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in txt2img[index]:
            tmp = np.where(inds == i)[0][0]  # position
            if tmp < rank:
                rank = tmp  # for each text find the highest rank from all corresponding image
        ranks[index] = rank

    # =====rank=====
    indices = np.argsort(-scores_t2i, axis=1)
    pred_labels = img2pid[indices]
    matches = np.equal(txt2pid.reshape(-1, 1), pred_labels)

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = np.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    # ==============

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = ir_mean + mAP

    eval_result = {
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'img_r_mean': ir_mean,
        'mAP': mAP,
        'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    num_classes = 6000

    #### Dataset ####
    print("Creating retrieval dataset")
    val_dataset, test_dataset = create_dataset(config['test_dataset'], config)

    val_loader, test_loader = create_loader([val_dataset, test_dataset], [None, None],
                                                           [config['batch_size_test']] * 2,
                                                          num_workers=[4, 4],
                                                          is_trains=[ False, False],
                                                          collate_fns=[None, None])

    #### Model ####
    print("Creating model")
    if args.load_head:
        mode = 'train'
    else:
        mode = 'eval'
    model = build_model(pretrained=config['pretrained'], mode=mode, num_classes=num_classes,
                        image_size=config['image_size'], vit=config['vit'],
                        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)

    if args.evaluate:

        checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['model']

        model_state_dict = {k: v for k, v in model_state_dict.items() if 'classifier' not in k}

        model.load_state_dict(model_state_dict, strict=False)

        # # 重新初始化分类层（根据需要）
        # model.classifier = nn.Linear(in_features=256, out_features=3102)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = config['init_lr']
    #     weight_decay = config['weight_decay']
    #
    #     if 'mlm_head' in key:
    #         lr = config['init_lr'] * 5
    #
    #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    start_time = time.time()

    score_test_t2i = evaluation(test_loader.dataset.img2pid,
                               test_loader.dataset.txt2pid,model_without_ddp, test_loader, device, config, itm=True)


    if args.evaluate:


        test_result = itm_eval(score_test_t2i, test_loader.dataset.txt2img,
                               test_loader.dataset.img2txt, test_loader.dataset.img2pid,
                               test_loader.dataset.txt2pid)

        print(test_result)

        log_stats = {
            **{f'test_{k}': v for k, v in test_result.items()},

        }
        with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_rstp.yaml')
    parser.add_argument('--output_dir', default= './output/RSTP/2025-07-21_15-27-55')
    parser.add_argument('--evaluate', default=True, )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--max_epoch', default=0, type=int)
    parser.add_argument('--batch_size_train', default=0, type=int)
    parser.add_argument('--batch_size_test', default=0, type=int)
    parser.add_argument('--init_lr', default=0.0001, type=float)
    parser.add_argument('--epoch_eval', default=1, type=int)
    parser.add_argument('--load_head', action='store_true')
    parser.add_argument('--k_test', default=32, type=int)
    parser.add_argument('--pretrained', default='./checkpoint/model_base.pth')
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.max_epoch > 0:
        config['max_epoch'] = args.max_epoch
    if args.batch_size_train > 0:
        config['batch_size_train'] = args.batch_size_train
    if args.batch_size_test > 0:
        config['batch_size_test'] = args.batch_size_test

    config['init_lr'] = args.init_lr
    if args.evaluate:
        config['pretrained'] = args.pretrained

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))


    main(args, config)