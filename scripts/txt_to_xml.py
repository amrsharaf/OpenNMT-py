from argparse import ArgumentParser

def parse_arguments():
    ap = ArgumentParser()
    ap.add_argument('--data', '-d', required=True)
    ap.add_argument('--output', '-o', required=True)
    args = ap.parse_args()
    return args

def cleanup_line(line):
    splits = line.strip().split()
    splits = splits[2:-1]
    splits.append('\n')
    return ' '.join(splits)

def main():
    args = parse_arguments()
    print 'reading data from: ', args.data
    with open(args.data, 'r') as reader:
        lines = [line for line in reader]
        lines = lines[2:-2]
        lines = map(cleanup_line, lines)
        print 'writing data to: ', args.output
        with open(args.output, 'w') as writer:
            writer.writelines(lines)
    print 'done processing files!'

if __name__ == '__main__':
    main()
