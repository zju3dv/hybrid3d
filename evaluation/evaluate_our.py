import os
import sys
sys.path.append('.')
import uuid
import subprocess
from datetime import datetime
import argparse


def get_info_from_name(name, prefix='FragmentDataLoader_'):
    name = os.path.split(os.path.dirname(name))[-1]
    return name[name.find(prefix) + len(prefix):]


def find_test_log(path, unique_str):
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if unique_str in name and name[:4] == 'log_':
                return name
    return None


def append_log_to_file(filepath, log, print_to_console=False):
    if print_to_console:
        print(log)
    with open(filepath, "a") as file_object:
        file_object.write(log + '\n')


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def get_checkpoint_epoch(model_path):
    import torch
    import parse_config
    checkpoint = torch.load(model_path)
    return checkpoint['epoch']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test script args')
    parser.add_argument('-f', '--file', default=None, type=str, help='model file path')
    parser.add_argument('-r', '--result', default=None, type=str, help='directly given result folder when test inference has been finished')
    parser.add_argument('-k', '--kpt_num', default='250,500', type=str, help='keypoint number setting')
    parser.add_argument('-i', '--info', default='', type=str, help='test info')
    parser.add_argument('--random_score', default=False, action='store_true')
    parser.add_argument('--redwood', default=False, action='store_true')
    args = parser.parse_args()

    # keypoint_num_list = [250, 500, 1000, 2000, 5000]
    keypoint_num_list = [int(e) if e.isdigit() else e for e in args.kpt_num.split(',')]

    if args.result is None:
        model_path = args.file
        checkpt_epoch = get_checkpoint_epoch(model_path)

        unique_str = str(uuid.uuid4())[:6]

        extra_info = '' if args.info == '' else '_' + args.info
        extra_info = extra_info + '_random_score' if args.random_score else extra_info
        extra_info = extra_info + '_redwood' if args.redwood else extra_info
        info = f'{get_info_from_name(model_path)}_epoch{checkpt_epoch}{extra_info}_{unique_str}'

        start_timestamp = get_timestamp()
        text_result = ''

        # test inference
        log_info = '{}: Inference...'.format(start_timestamp)
        print(log_info)
        test_inference_cmd = ['python', 'test.py',
            '-c', 'config/test_3dmatch_frag.yaml',
            '-r', f'{model_path}',
            '-i', f'{info}']
        if args.random_score:
            test_inference_cmd.append('--random_score')
        text_result = subprocess.check_output(test_inference_cmd).decode(sys.stdout.encoding)

        test_log_dir = find_test_log('./saved/', unique_str)
        metric_log_filepath = f'saved/{test_log_dir}/metric_info.log'
        append_log_to_file(metric_log_filepath, log_info)
        append_log_to_file(metric_log_filepath, text_result)
    else:
      test_log_dir =  os.path.basename(args.result)

    metric_log_filepath = f'saved/{test_log_dir}/metric_info.log'

    gt_base_dir = 'data/redwood_lidar/fragments_adaptive' if args.redwood else 'data/3dmatch/geometric_registration_adaptive'

    # repetability
    append_log_to_file(metric_log_filepath, '{}: Repetability...'.format(
        get_timestamp()), True)
    text_result = subprocess.check_output(
        ['python', 'evaluation/repeatability_3dmatch.py', gt_base_dir, f'saved/{test_log_dir}/clustered_kpts_weights.pkl']
    ).decode(sys.stdout.encoding)
    print(text_result)
    append_log_to_file(metric_log_filepath, text_result)

    for keypoint_num in keypoint_num_list:
        # feature matching recall
        append_log_to_file(metric_log_filepath, '{}: Feature Matching Recall at {} kpts...'.format(
            get_timestamp(), keypoint_num), True)
        text_result = subprocess.check_output(
            ['python', 'evaluation/registration_3dmatch.py',
                str(keypoint_num), gt_base_dir,
                f'saved/{test_log_dir}']
        ).decode(sys.stdout.encoding)
        text_result = text_result[text_result.find('********'):]
        print(text_result)
        append_log_to_file(metric_log_filepath, text_result)

        # registration recall
        append_log_to_file(metric_log_filepath, '{}: Registration Recall at {} kpts...'.format(
            get_timestamp(), keypoint_num), True)
        text_result = subprocess.check_output(
            ['matlab', '-nosplash', '-nodisplay', '-nodesktop', '-r',
                'cd evaluation/3dmatch/; evaluate_cmd ../../saved/{}/log_result_{} ../../{}; exit'
             .format(test_log_dir, keypoint_num, gt_base_dir)]
        ).decode(sys.stdout.encoding)
        text_result = text_result[text_result.find('totalRecall'):]
        print(text_result)
        append_log_to_file(metric_log_filepath, text_result)
