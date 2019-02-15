
import pandas as pd 
import pickle
from utils import get_chars_data_size_vocab_size,get_dicss
# Getting Datasets
names_list = pd.read_csv('../yob2017.txt',header=None)
boy_names = names_list[names_list[1]=='M'][0].apply(str.lower)
girl_names = names_list[names_list[1] == 'F'][0].apply(str.lower)
combined = names_list[0]
# Preparing Index Arrays ( will be used as a input to RNN)
# For Boys Names
chars, boy_strings,data_size,vocab_size= get_chars_data_size_vocab_size(boy_names)
ix_to_char,char_to_ix = get_dicss(chars)
with open('boys_string.pickle','wb') as file:
    pickle.dump(boy_strings, file)
with open ('ix_char_ix.pickle','wb') as file:
    pickle.dump((ix_to_char,char_to_ix),file)
print('Boys Pickle Done there are {} names'.format(data_size))

# For Girls Names
chars, girl_strings,data_size,vocab_size= get_chars_data_size_vocab_size(girl_names)
with open('girls_string.pickle','wb') as file:
    pickle.dump(girl_strings,file)
print('Girl Pickle Done there are {} names'.format(data_size))
# For Combined Names
chars, combined_strings,data_size,vocab_size= get_chars_data_size_vocab_size(combined)
with open ('combined_string.pickle','wb') as file:
    pickle.dump(combined_strings,file)
print('Combined Pickle Done there are {} names'.format(data_size))
print('Pickle Done')
