import argparse
import collections
import open3d
import torch
import numpy as np
import random
import json

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

from parse_config import ConfigParser

def main(config):
    # import module after Config init
    from data_loader.data_loader_factory import get_data_loader_by_name
    import model.losses.loss as module_loss
    import model.metric as module_metric
    from model.model_factory import get_model_by_name
    from trainer import Trainer, TrainerFragment, TrainerFragmentPair, TrainerCoord, TrainerCoordVote, TrainerFusion

    logger = config.get_logger('train')
    logger.info(json.dumps(config.config,sort_keys=True, indent=4))

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', get_data_loader_by_name(config['data_loader']['type']), config)
    valid_data_loader = data_loader.split_validation()
    
    # build model architecture, then print to console
    model = config.init_obj('arch', get_model_by_name(config['arch']['type']), config)
    logger.debug(model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    criterion = config.init_obj('loss', module_loss, config)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if config['trainer']['type'] == 'TrainerFragment':
        trainer = TrainerFragment(model, criterion, metrics, optimizer,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
    elif config['trainer']['type'] == 'TrainerFragmentPair':
        trainer = TrainerFragmentPair(model, criterion, metrics, optimizer,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
    elif config['trainer']['type'] == 'TrainerCoord':
        trainer = TrainerCoord(model, criterion, metrics, optimizer,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
    elif config['trainer']['type'] == 'TrainerCoordVote':
        trainer = TrainerCoordVote(model, criterion, metrics, optimizer,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
    elif config['trainer']['type'] == 'TrainerFusion':
        trainer = TrainerFusion(model, criterion, metrics, optimizer,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
    elif config['trainer']['type'] == 'Trainer':
        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='train args')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--info', default=None, type=str,
                      help='training info (default: NONE)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--se', '--start_epoch'], type=int, target='force_start_epoch'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
