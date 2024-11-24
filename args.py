from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    parser.add_argument('--city', type = str, default='bj')
    parser.add_argument('--cell_size', type = int, default=2000)
    parser.add_argument('--test_batch_size', type = int, default=100)
    parser.add_argument('--use_multiple', type= bool, default= False)

    args, unknown = parser.parse_known_args()
    if len(unknown)!= 0 and not args.ignore_unknown_args:
        print("some unrecognised arguments {}".format(unknown))
        raise SystemExit

    return args