import sys
from tqdm import tqdm
from argparse import ArgumentParser

def fileRead(fname,index):
    sent = ""
    sentences=[]
    with open(fname) as f:
        for line in f:
            if len(line) > 1:
                line = line.rstrip('\r\n')
                values = line.split()

                if (len(sent) > 0):
                    sent = sent + " " +  values[index]
                else:
                    sent =  values[index]
            else:
                sentences.append(sent)
                sent = ""
    return sentences

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="src",
                    help="file containing source text and labels", metavar="FILE")
parser.add_argument("-s", "--sIndex", dest="s",
                    help="source Index", metavar="FILE")
parser.add_argument("-l", "--lIndex", dest="l",
                    help="label Index", metavar="FILE")

args = parser.parse_args()

source=fileRead(args.src,int(args.s)) # 1 is the index of source words
labels=fileRead(args.src,int(args.l)) # 4 is the index of POS in the file

count = 0

for i, sentence in tqdm(enumerate(source)):
    print (sentence + "\t" + labels[i])
    
		
