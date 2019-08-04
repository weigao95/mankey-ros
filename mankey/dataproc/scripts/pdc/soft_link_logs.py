import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--src_log_dir', type=str, default='/home/wei/data/pdc/logs_mugs')
parser.add_argument('--target_log_dir', type=str, default='/home/wei/data/pdc/logs_proto')
args = parser.parse_args()


def main():
    src_log_dir: str = args.src_log_dir
    target_dir: str = args.target_log_dir
    assert os.path.exists(src_log_dir)
    assert os.path.exists(target_dir)
    for log in os.listdir(src_log_dir):
        full_log_path = os.path.join(src_log_dir, log)
        target_log_path = os.path.join(target_dir, log)
        command = 'ln -s %s %s' % (full_log_path, target_log_path)
        os.system(command)


if __name__ == '__main__':
    main()
