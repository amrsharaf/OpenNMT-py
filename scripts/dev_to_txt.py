from argparse import ArgumentParser

def parse_arguments():
    ap = ArgumentParser()
    ap.add_argument('--input', '-i', required=True)
    ap.add_argument('--output', '-o', required=True)
    return ap.parse_args()

def clean_line(line):
    splits = line.split(',')
    return ','.join(splits[1:])

def main():
    args = parse_arguments()
    print('reading data from: ', args.input)
    with open(args.input, 'r') as reader:
        lines = (line for line in reader)
        lines = map(clean_line, lines)
        print('writing data to: ', args.output)
        with open(args.output, 'w') as writer:
#@            lines = map(lambda x: x + '\n', lines)
            writer.writelines(lines)
    print('done processing data file!')

if __name__ == '__main__':
    main()
