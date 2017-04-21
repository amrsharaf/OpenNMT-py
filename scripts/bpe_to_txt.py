from argparse import ArgumentParser

def parse_arguments():
    ap = ArgumentParser()
    ap.add_argument('--input', '-i', required=True)
    ap.add_argument('--output', '-o', required=True)
    return ap.parse_args()

def is_bpe(word):
    if len(word) < 2:
        return False
    return word[-2:] == '@@'

def txt_generator(words):
    i = 0
    word = ''
    while i < len(words):
        word += words[i]
        if not is_bpe(word):
            yield word
            word = ''
        else:
            word = word[:-2]
        i += 1

def bpe_to_txt(line):
    splits = line.strip().split()
    txt = list(txt_generator(splits))
    txt.append('\n')
    return ' '.join(txt)

def main():
    args = parse_arguments()
    print 'reading data from: ', args.input
    with open(args.input, 'r') as reader:
        lines = [line for line in reader]
        lines = map(bpe_to_txt, lines)
        print 'writing data to: ', args.output
        with open(args.output, 'w') as writer:
            writer.writelines(lines)
    print 'done processing files!'


if __name__ == '__main__':
    main()
