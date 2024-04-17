import argparse

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--family', type=str,help='姓')
parser.add_argument('--name', type=str,help='名')
args = parser.parse_args()

#打印姓名
print(args.family+args.name)
