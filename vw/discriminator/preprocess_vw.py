from argparse import ArgumentParser
import random

def parse_arguments():
    ap = ArgumentParser()
    ap.add_argument('--old', '-o', required=True, help='Old domain data!')
    ap.add_argument('--new', '-n', required=True, help='New domain data!')
    ap.add_argument('--old_test', '-ot', required=True, help='Old domain data!')
    ap.add_argument('--new_test', '-nt', required=True, help='New domain data!')
    args = ap.parse_args()
    return args

def text_to_vw(line):
    return line.replace(':', 'COLON').replace('|', 'PIPE')
    
def read_sentences(file_path):
    with open(file_path, 'r') as reader:
        lines = [line.strip() for line in reader]
    return map(text_to_vw, lines)

def write_to_vw_file(filename, examples):
    with open(filename, 'w') as writer:
        for ex in examples:
            print >>writer, ex

def process_sentences(old_sentences, new_sentences, output_file):
    random.shuffle(old_sentences)
    random.shuffle(new_sentences)
    old_sentences = map(lambda x: '+1 | ' + x, old_sentences)
    print old_sentences[0]
    new_sentences = map(lambda x: '-1 | ' + x, new_sentences)
    print new_sentences[0]
    print 'balancing old and new data...'
    min_length = min(len(new_sentences), len(old_sentences))
    old_sentences = old_sentences[:min_length]
    new_sentences = new_sentences[:min_length]
    examples = old_sentences + new_sentences
    print 'Total number of examples from old domain: ', len(old_sentences)
    print 'Total number of examples from new domain: ', len(new_sentences)
    print 'Total number of examples: ', len(examples)
    random.shuffle(examples)
    write_to_vw_file(output_file, examples)

def process_files(old_file, new_file, output_file):        
    old_sentences = read_sentences(old_file)
    new_sentences = read_sentences(new_file)
    process_sentences(old_sentences, new_sentences, output_file)

def main():
    random.seed(1234)
    args = parse_arguments()
    print 'Reading old domain data from: ', args.old
    print 'Reading new domain data from: ', args.new
    process_files(args.old, args.new, 'data/train.vw')
    process_files(args.old_test, args.new_test, 'data/test.vw')

if __name__ == '__main__':
    main()