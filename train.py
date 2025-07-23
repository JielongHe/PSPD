import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from ruamel.yaml import YAML
import torch
import gc
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import openai

from models.pspd import build_model
import utils
from utils import cosine_lr_schedule, cos_with_warmup_lr_scheduler
from data import create_dataset, create_sampler, create_loader
from processor.processor import pre_train

import pickle


def train(model, data_loader, optimizer, epoch, device, config):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_glb', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_loc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))




    header = f'Train Epoch: [{epoch}]'
    print_freq = 100

    for i, (image, caption, img_pid, gpt_captions) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 延迟移动数据到 GPU
        image = image.to(device, non_blocking=True)
        img_pid = img_pid.to(device, non_blocking=True)

        loss_glb, loss_loc= model(image, caption, gpt_captions, idx=img_pid)
        loss = loss_glb + loss_loc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_loc=loss_loc.item())
        metric_logger.update(loss_glb=loss_glb.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        del image, loss, caption, img_pid
        torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.8f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def start_train(i, args, combined_ret, device, generating_model= None):
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config['dataset'], config, datasets=combined_ret)

    del combined_ret
    torch.cuda.empty_cache()
    num_classes = train_dataset.num_classes


    samplers = [None, None, None]
    # #
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])



    #### Model ####
    print("Creating model")
    if args.load_head:
        mode = 'train'
    else:
        mode = 'eval'
    model = build_model(pretrained=config['pretrained'], mode=mode, num_classes=num_classes,
                        image_size=config['image_size'], vit=config['vit'],
                        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    # if i != 0:
    #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model'])
    #     del checkpoint, checkpoint_path

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = config['init_lr']
        weight_decay = config['weight_decay']

        if 'mlm_head' in key:
            lr = config['init_lr'] * 5

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.AdamW(params=params, lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0
    epoch_eval = args.epoch_eval

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, config)
        if epoch % epoch_eval != 0:
            continue

        score_val_t2i = evaluation(model_without_ddp, val_loader, device, config, generating_model, itm=True)
        score_test_t2i = evaluation(model_without_ddp, test_loader, device, config,
                                                        generating_model, itm=True)

        score_val_glb = evaluation(model_without_ddp, val_loader, device, config, generating_model, itm=False)



        if utils.is_main_process():
            test_result = itm_eval(score_test_t2i, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2pid,
                                   test_loader.dataset.txt2pid)
            val_result = itm_eval(score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2pid, val_loader.dataset.txt2pid)
            val_glb_result = itm_eval(score_val_glb, val_loader.dataset.txt2img,
                                      val_loader.dataset.img2pid, val_loader.dataset.txt2pid)
            print("test", test_result)
            print("val", val_result)
            print("glb", val_glb_result)

            if test_result['r_mean'] > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                if not args.evaluate:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = test_result['r_mean']
                best_epoch = epoch

                best_log_stats = {
                    **{f'test_{k}': v for k, v in test_result.items()},
                    # **{f'test_glb_{k}': v for k, v in test_glb_result.items()},
                }

            if not args.evaluate:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                #

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        if epoch == config['max_epoch'] - 1 or args.evaluate:
            with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                f.write(json.dumps(best_log_stats) + "\n")

        if args.evaluate:
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    del model, model_without_ddp, optimizer, train_loader, val_loader, test_loader

    torch.cuda.empty_cache()  # Clear CUDA memory cache
    gc.collect()  # Trigger garbage collection to release unused memory






@torch.no_grad()
def evaluation(model, data_loader, device, config, generating_model, gen_config=None, itm=False):
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
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=73, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    gen_text_ids = []
    gen_text_embeds = []
    gen_text_atts = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:,0,:])
        image_embed = F.normalize(image_embed,dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

        if gen_config != None:
            gen_caption = generating_model.generate(image, sample=False, num_beams=gen_config['num_beams'],
                                                max_length=gen_config['max_length'],
                                                min_length=gen_config['min_length'])
            # gen_caption.extend(caption)
            text_input = model.tokenizer(gen_caption, padding='max_length', truncation=True, max_length=73,
                                             return_tensors="pt").to(device)
            text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             mode='text')
            text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
            gen_text_embeds.append(text_embed)
            gen_text_ids.append(text_input.input_ids)
            gen_text_atts.append(text_input.attention_mask)

    if gen_config != None:
        gen_text_embeds = torch.cat(gen_text_embeds, dim=0)
        gen_text_ids = torch.cat(gen_text_ids, dim=0)
        gen_text_atts = torch.cat(gen_text_atts, dim=0)
        gen_text_ids[:, 0] = model.tokenizer.enc_token_id




    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = text_embeds @ image_embeds.t()


    if gen_config != None:
        gen_sims_matrix = gen_text_embeds @ image_embeds.t()
        gen_score_matrix_t2i = torch.full((len(data_loader.dataset.image), len(data_loader.dataset.image)), -100.0).to(
            device)


    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)


    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    if itm == True:
        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)):
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

        if gen_config != None:
            for i, sims in enumerate(metric_logger.log_every(gen_sims_matrix[start:end], 500, header)):
                topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
                encoder_output = image_feats[topk_idx.cpu()].to(device)
                bs = encoder_output.shape[0]
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
                output = model.text_encoder(gen_text_ids[start + i].repeat(config['k_test'], 1),
                                            attention_mask=gen_text_atts[start + i].repeat(config['k_test'], 1),
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True,
                                            )

                score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]

                gen_score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    score_matrix_t2i = score_matrix_t2i + sims_matrix
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


    if gen_config != None:
        gen_score_matrix_t2i = gen_score_matrix_t2i + gen_sims_matrix
        return score_matrix_t2i.cpu().numpy(), gen_score_matrix_t2i.cpu().numpy()

    return score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_t2i, txt2img, img2pid, txt2pid):  # txt2img: corresponding id

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
    r_mean =  ir1+mAP

    eval_result = {
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                    'mAP':mAP,
                   'r_mean': r_mean}
    return eval_result


def create_and_load_datasets(config):

    pre_train_dataset, source_train_dataset = create_dataset(config['pre_dataset'], config)
    sampler = [None, None]
    pre_train_loader, source_train_loader = create_loader(
        [pre_train_dataset, source_train_dataset], sampler,
        batch_size=[100, 100],
        num_workers=[4, 4],
        is_trains=[False, False],
        collate_fns=[None, None]
    )

    del pre_train_dataset, source_train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return pre_train_loader, source_train_loader


def load_model_and_checkpoint(config, args, i, num_classes):

    mod = 'eval'
    pre_model = build_model(pretrained=config['pretrained'], mode=mod, num_classes=num_classes,
                            image_size=config['image_size'], vit=config['vit'],
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    if i != 0:

        checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        checkpoint = torch.load(checkpoint_path)
        pre_model.load_state_dict(checkpoint['model'])

        del checkpoint, checkpoint_path

    gc.collect()
    torch.cuda.empty_cache()

    return pre_model


def save_gen_data(gen_ret):
    # 保存并加载生成的数据
    with open('data.pkl', 'wb') as file:
        pickle.dump(gen_ret, file)

def load_gen_data():

    with open('data.pkl', 'rb') as file:
        gen_ret = pickle.load(file)

    return gen_ret


def modify_gen_data(gen_ret, n_clusters):
    # 处理生成的数据
    gen_rets = []
    for data in gen_ret:
        label, img_path, caption = data
        modified_label = label + n_clusters
        gen_rets.append((modified_label, img_path, caption))
    return gen_rets



def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    iter_finetune_epoch = 5
    min_samples = 4
    num_classes = 671

    eps0_all = 1.011

    for i in np.arange(iter_finetune_epoch):

        pre_train_loader, source_train_loader = create_and_load_datasets(config)

        pre_model = load_model_and_checkpoint(config, args, i, num_classes).to(device)

        gen_ret, gen_n_clusters, eps0_all = pre_train( i, pre_train_loader, source_train_loader,
                                             pre_model, min_samples, config, eps0_all = eps0_all)
        num_classes = gen_n_clusters


        start_train(i, args, gen_ret, device)

        del gen_ret
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_cuhk.yaml')
    # parser.add_argument('--configs', default='./configs/caption.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_Person')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--max_epoch', default=0, type=int)
    parser.add_argument('--batch_size_train', default=0, type=int)
    parser.add_argument('--batch_size_test', default=0, type=int)
    parser.add_argument('--init_lr', default=0.0001, type=float)
    parser.add_argument('--epoch_eval',default=1,type=int)
    parser.add_argument('--load_head',action='store_true')
    parser.add_argument('--k_test',default=32,type=int)
    parser.add_argument('--pretrained',default='./checkpoint/model_base.pth')
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    yaml = YAML(typ='safe')
    # get image-text pair datasets dataloader
    # with open(args.configs, 'r') as file:
    #     configs = yaml.load(file)

    if args.max_epoch > 0:
        config['max_epoch'] = args.max_epoch
    if args.batch_size_train > 0:
        config['batch_size_train'] = args.batch_size_train
    if args.batch_size_test > 0:
        config['batch_size_test'] = args.batch_size_test

    config['init_lr'] = args.init_lr
    if args.evaluate:
        config['pretrained'] = args.pretrained
    config['k_test'] = args.k_test

    now = datetime.datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    # formatted_now  = '2025-03-11_10-44-04'

    args.output_dir = args.output_dir + '/' + formatted_now
    # args.output_dir = '/home/aorus/He/aun/unT/output/CUHK/2024-11-13_16-32-17'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)