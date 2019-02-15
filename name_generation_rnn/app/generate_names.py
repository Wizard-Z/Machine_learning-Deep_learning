import pickle
from utils import sample
from utils import read_pickle
from utils import print_sample
import numpy as np
## Generate Names
ix_to_char,char_to_ix = read_pickle('datasets/ix_char_ix.pickle')[0]
def generate(parameters,seed,names = 20):
    for name in range(names):
        # Sample indices and print them
        sampled_indices = sample(parameters, char_to_ix, seed)
        print_sample(sampled_indices, ix_to_char)
        seed += 1   
        

def main():
    print('Program Loaded!')
    seed = np.random.randint(100)    
    sel = input('1 for Boys Names\n2 for Girls Names\n3 for Combined\n>>>')
    if sel in ['1','2','3']:
        names = int(input('ENTER NUMBER OF NAMES TO BE DISPLAYED\n>>>'))
        print('>>>')
        if sel == '1':
            parameters = read_pickle('datasets/boy_params.txt')[0]
            generate(parameters,seed,names)
        if sel == '2':
            parameters = read_pickle('datasets/girl_params.txt')[0]
            generate(parameters,seed,names)
        if sel == '3':
            parameters = read_pickle('datasets/c_params.txt')[0]
            generate(parameters,seed)
    else:
        print('-- Invalid Choice --' )
    ch = input('\nRegenerate (y/n) \n>>> ')
    if ch == 'y':
        print('\nProgram Loading . . .')
        main()
    else :
        print(' -- Thanks for using -- ')
      
if __name__ == '__main__':
    print('\nProgram Loading . . .')
    main()