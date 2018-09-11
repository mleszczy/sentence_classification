import argparse 
import numpy as np 
import os 

import collect_scores 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Models, options: [lstm, la, cnn]') 
    parser.add_argument('--wiki', action='store_true', help='Use results')
    return parser.parse_args()

def main():
    args = parse_args()
    listdir = '/data/scripts/paper_runs'
    tags = ['sgd', 'w2v_sg', 'w2v_cbow', 'glove', 'ft_sg', 'ft_cbow']
    if args.wiki:
        tags = ['wiki_pi', 'wiki_sgd', 'wiki_w2v_cbow', 'wiki_glove']
    model = args.model
    
    for t in tags: 
        filepath = os.path.join(listdir, t+'_runs')
        with open(filepath, 'r') as f:
            embpaths = [e.strip() for e in f.readlines()]
        results = []
        for e in embpaths:
            emb_name = os.path.basename(e)
            test_mean, test_std = collect_scores.get_scores(emb_type=t, model=model, emb_name=emb_name) 
            results.append(test_mean)
        print(t, np.mean(results), np.std(results))

if __name__ == "__main__":
    main()
