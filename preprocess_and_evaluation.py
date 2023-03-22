import argparse
import os
import os
import shutil
from runners.utils import make_dir
from evaluation.LPIPS import calc_LPIPS, find_max_min_LPIPS
from evaluation.diversity import calc_diversity


def rename_sample_files(source_dir: str, target_dir: str):
    flist = os.listdir(source_dir)
    flist.sort()
    make_dir(target_dir)

    total = len(flist)
    for i in range(total):
        if i % 1000 == 0:
            print(f"{i} samples")
        fpath = os.path.join(source_dir, flist[i])
        if os.path.isdir(fpath):
            shutil.copytree(os.path.join(source_dir, flist[i]),
                            os.path.join(target_dir, str(i)))
        elif os.path.isfile(fpath):
            shutil.copy(os.path.join(source_dir, flist[i]),
                        os.path.join(target_dir, f"{str(i)}.png"))
        else:
            raise NotImplementedError


def copy_sample_files(source_dir: str, target_dir: str):
    flist = os.listdir(source_dir)
    flist.sort()
    make_dir(target_dir)

    total = len(flist)
    for i in range(total):
        if i % 1000 == 0:
            print(f"{i} samples")
        shutil.copy(os.path.join(source_dir, flist[i], 'output_0.png'),
                    os.path.join(target_dir, f'{flist[i]}.png'))


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('-f', '--func_name', type=str, default=None, help='Path to the config file')
    parser.add_argument('-r', '--root_dir', type=str, default=None, help='Path to the config file')
    parser.add_argument('-s', '--source_dir', type=str, default=None, help='Path to the source directory')
    parser.add_argument('-t', '--target_dir', type=str, default=None, help='Path to the target directory')
    parser.add_argument('-n', '--num_samples', type=int, default=1, help='Path to the destination directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args_and_config()
    if args.func_name == "rename_samples":
        source_dir = os.path.join(args.root_dir, args.source_dir)
        target_dir = os.path.join(args.root_dir, args.target_dir)
        print(f"copy sample files from {source_dir} to {target_dir}")
        rename_sample_files(source_dir=source_dir, target_dir=target_dir)
    elif args.func_name == "copy_samples":
        source_dir = os.path.join(args.root_dir, args.source_dir)
        target_dir = os.path.join(args.root_dir, args.target_dir)
        print(f"rename sample files from {source_dir} to {target_dir}")
        copy_sample_files(source_dir=source_dir, target_dir=target_dir)
    elif args.func_name == "LPIPS":
        print(f"calculate LPIPS {args.source_dir}")
        calc_LPIPS(data_dir=args.source_dir, gt_dir=args.target_dir, num_samples=args.num_samples)
    elif args.func_name == "max_min_LPIPS":
        print(f"calculate max_min_LPIPS {args.source_dir}")
        find_max_min_LPIPS(data_dir=args.source_dir, gt_dir=args.target_dir, num_samples=args.num_samples)
    elif args.func_name == "diversity":
        print(f"calculate diversity {args.source_dir}")
        calc_diversity(data_dir=args.source_dir, num_samples=args.num_samples)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
