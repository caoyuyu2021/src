import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
parser.add_argument('integers', type=str, nargs='+',help='传入的数字')
args = parser.parse_args()

print(args.integers)
