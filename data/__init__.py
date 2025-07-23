import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.cuhk_dataset import cuhk_caption_gen_train, cuhk_pre_train, cuhk_caption_train, pre_cuhk_pede_train, cuhk_pede_train, cuhk_pre_train1, cuhk_pede_caption_eval, \
    cuhk_pede_retrieval_eval, cuhk_pede_trainset_eval, mix_pede_train
from data.icfg_dataset import icfg_caption_gen_train, icfg_pede_train, icfg_caption_train, icfg_pede_retrieval_eval, icfg_pre_train,icfg_pre_train1, pre_icfg_pede_train
from data.rstp_dataset import rstp_caption_gen_train, rstp_pede_train, rstp_caption_train, rstp_pede_retrieval_eval, rstp_pre_train, rstp_pre_train1, pre_rstp_pede_train
from transform.randaugment import RandomAugment


def create_dataset(dataset, config, datasets=None, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    if type(config['image_size']) == int:
        image_size = (config['image_size'], config['image_size'])
    elif type(config['image_size']) == list or type(config['image_size']) == tuple:
        image_size = (config['image_size'][0], config['image_size'][1])

    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Pad(10),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.4), value=(0.48145466, 0.4578275, 0.40821073)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'gen_train_caption':
        cuhk_pre_dataset = cuhk_caption_gen_train(transform_test, config['cuhk_image_root'])
        rstp_pre_dataset = rstp_caption_gen_train(transform_test, config['rstp_image_root'])
        icfg_pre_dataset = icfg_caption_gen_train(transform_test, config['icfg_image_root'])
        return cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset

    elif dataset == 'gen_caption':
        cuhk_pre_dataset = cuhk_caption_train(transform_test, config['cuhk_image_root'])
        rstp_pre_dataset = rstp_caption_train(transform_test, config['rstp_image_root'])
        icfg_pre_dataset = icfg_caption_train(transform_test, config['icfg_image_root'])
        return cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset

    elif dataset == 'pre_gen_caption':
        cuhk_pre_dataset = pre_cuhk_pede_train(transform_train, config['cuhk_image_root'])
        rstp_pre_dataset = pre_rstp_pede_train(transform_train, config['rstp_image_root'])
        icfg_pre_dataset = pre_icfg_pede_train(transform_train, config['icfg_image_root'])
        return cuhk_pre_dataset, rstp_pre_dataset, icfg_pre_dataset

    elif dataset == 'pre_retrieval_cuhk':
        train_dataset = cuhk_pre_train(transform_test, config['image_root'])
        source_dataset = icfg_pre_train1(transform_test, config['source_image_root'])

        return train_dataset, source_dataset

    elif dataset == 'pre_retrieval_icfg':
        train_dataset = icfg_pre_train(transform_test, config['image_root'])
        source_dataset = cuhk_pre_train1(transform_test, config['source_image_root'])

        return train_dataset, source_dataset

    elif dataset == 'pre_retrieval_rstp':
        train_dataset = rstp_pre_train(transform_test, config['image_root'])
        source_dataset = cuhk_pre_train1(transform_test, config['source_image_root'])

        return train_dataset, source_dataset

    elif dataset == 'cuhk_trainset_eval':
        test_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        val_dataset = cuhk_pede_trainset_eval(transform_test, config['image_root'])
        return val_dataset, test_dataset

    elif dataset == 'icfg_trainset_eval':
        val_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return  val_dataset, test_dataset

    elif dataset == 'rstp_trainset_eval':
        val_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return val_dataset, test_dataset

    elif dataset == 'retrieval_cuhk':
        train_dataset = cuhk_pede_train(transform_train, datasets)
        val_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_icfg':
        train_dataset = cuhk_pede_train(transform_train, datasets)
        val_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_rstp':
        train_dataset = cuhk_pede_train(transform_train, datasets)
        val_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
