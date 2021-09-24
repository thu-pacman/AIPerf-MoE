import json
import argparse

parser = argparse.ArgumentParser(
    description='convert text to *.json, divide text to docs by empty line')
parser.add_argument('--input-file', type=str, default=None)
parser.add_argument('--output-prefix', type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    output = {}
    fin = open(args.input_file, mode='r', encoding='utf-8')
    fout = open(args.output_prefix + '.json', mode='w')
    doc = ''
    for line in fin:
        if len(line) > 1:
            doc += line
        else:  # empty line
            if len(doc) < 1:  # empty doc
                continue
            output['text'] = doc
            doc = ''
            fout.write(json.dumps(output) + '\n')
