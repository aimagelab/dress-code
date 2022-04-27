import argparse


def get_conf(train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--category", default='all', type=str)
    parser.add_argument("--dataroot", type=str, default="<Dress Code Path here>")
    parser.add_argument("--data_pairs", default="{}_pairs")

    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='save checkpoint infos')

    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=0)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--display_count", type=int, default=1000)
    parser.add_argument("--shuffle", default=True, action='store_true', help='shuffle input data')

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--radius", type=int, default=5)

    args = parser.parse_args()
    print(args)
    return args
