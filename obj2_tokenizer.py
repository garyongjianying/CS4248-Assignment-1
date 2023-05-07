'''
    NUS CS4248 Assignment 1 - Objective 2 (Tokenization, Zipf's Law)

    Class Tokenizer for handling Objective 2

    Important: please strictly comply with the input/output formats for
               the method of tokenize_sentence, as we will call it in testing
'''
###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

from lib2to3.pgen2.tokenize import tokenize
import matplotlib.pyplot as plt     # Requires matplotlib to create plots.
import numpy as np    # Requires numpy to represent the numbers
import re # important library for regex
import collections # for usage of bpe
import sys # to retrieve args from command line
import matplotlib.pyplot as plt # for plotting graphs
from operator import itemgetter

def draw_plot(r, f, imgname):
    # r is rank, f is frequency, imgname is what you want to name the file of your plot.
    # Data for plotting
    x = np.asarray(r)
    y = np.asarray(f)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Rank (log)', ylabel='Frequency (log)',
        title='Word Frequency v.s. Rank (log)')
    ax.grid()
    fig.savefig(f"../plots/{imgname}")
    plt.show()

#################################### Classs defined for byte-pair encoding for a corpus_text #########################################################



# Specify classes for byte-pair-encoding for token learner and token segmenter.


#####################################################################################################################################################

class Tokenizer:

    # # initialize the vocab that will be learned from the regex/bpe tokenization from large txt file as an empty list.
    # vocab = [] # Class Attribute for vocab

    def __init__(self, path, bpe=False, lowercase=True):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()
        
        self.bpe = bpe
        self.lowercase = lowercase

        # initialize the corpus & vocab that will be learned from the regex/bpe tokenization from large txt file as an empty list.
        self.corpus = []
        self.vocab = []

        # Initialize the regex here to be used in tokenize and tokenize_sentence
        self.regex = r"[\d]+-[\d]+|[\w]+-[\w]+-[\w]+|[\d]+[.\/][\d]+|(?<=[a-zA-Z])'s|[!#$%&\'()*+,.:;-<=>?@\\\/^_`'{|}~]|[a-zA-Z]+|[\d]+"
        # 1 - handle phone number XXX-XXXX
        # 2 - handle numbers that follow 12.5 or 1/2 format
        # 3 - handle case like 'ten-year-old'
        # 4 - handle possessive case "Elle's book" --> "Elle" and "'s'" --> use a positive lookbehind to find "'s'" only when it is preceded by a word.
        # 5 - handle punctuations
        # 6 - handle full words
        # 7 - handle all digits



        # Specify inputs required for BPE token learner
        self.n_iterations = 100
        # initialize token list to be returned as a result of tokenization after referencing to the vocab.
        self.valid_token_list = []

        self.subword_tokens = {} # to store all the final vocabulary from BPE

    # Specify classes for byte-pair-encoding for token learner and token segmenter.
    ###########################################################################################################################
    # class for token learner for BPE
    class bpe_token_learner:
        
        # initailize our bpe_token_learner class
        def __init__(self, corpus_text):
            self.corpus_text = corpus_text
            self.word_freq_dict = collections.defaultdict(int) # initialize the word_freq_dict
            self.words = corpus_text.strip().split(" ") # first split the corpus text by white spaces
            self.bpe_codes = {} # store the best pair during each iteration for encoding new vocabulary
            self.subword_tokens = {} # Store the final vocab for byte-pair encoding.
            # iniialize word_freq_dict
            for word in self.words:
                self.word_freq_dict[' '.join(word) + ' _'] += 1 # adding an underscore to identify end of a word at the end of each word in word_freq_dict

        
        # def get_pairs(self, word_freq_dict):
        #     pairs = collections.defaultdict(int)
        #     for word, freq in word_freq_dict.items():
        #         chars = word.split()
        #         for i in range(len(chars)-1):
        #             pairs[chars[i], chars[i+1]] += freq # accumulate adjacent pairs of characters and sum their frequencies to get the total count.
        #     return pairs


        def get_pairs(self, word_freq_dict):
            pairs = collections.defaultdict(int)
            for word, freq in word_freq_dict.items():
                chars = word.split()
                for i in range(len(chars)-1):
                    pairs[chars[i], chars[i+1]] += freq # accumulate adjacent pairs of characters and sum their frequencies to get the total count.
            return pairs


        # Step 2: Split the word into characters and then calculate the character frequency and append into char_freq_dict
        def get_subword_tokens(self, word_freq_dict):
            char_freq_dict = collections.defaultdict(int)
            for word, freq in word_freq_dict.items():
                chars = word.split()
                for char in chars:
                    char_freq_dict[char] += freq
            return char_freq_dict


        def merge_byte_pairs(self, best_pair, word_freq_dict):
            merged_dict = {}
            bigram = re.escape(' '.join(best_pair[0]))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            for word in word_freq_dict:
                # print(word)
                # p.sub will replace all occurrences found by the regex match in p for any word in word_freq_dict with the new best pair as in BPE algorithm.
                w_out = p.sub(''.join(best_pair[0]), word)
                merged_dict[w_out] = word_freq_dict[word]
            return merged_dict


        def byte_pair_encoding(self,n_iterations):
            # Byte-pair encoding to get tokens with hyperparameter n_iterations
            for i in range(n_iterations):
                # get counts of all adjacent pairs at the current iteration i.
                pairs = self.get_pairs(self.word_freq_dict)

                # find out the best pairs, if there is a tie, use lexographical order to pick.
                # take the first entry's value, since we have already sorted it by value and it should represent the max count.
                max_count = sorted(pairs.items(), key=lambda x: (x[1]), reverse=True)[0][1] # [0] takes first entry after sorting by descending value, [1] will take the value

                # find all the tiebreaker entries in the dictionary
                tiebreaker_dict = {}
                for key,val in pairs.items():
                    if val == max_count:
                        tiebreaker_dict[key] = val

                # if we only have one tiebreaker, take this as the correct pair, else, we need to arrange by key (lexographical order)
                # sort the tiebreaker_dict by key to achieve a lexographical ordering
                # and always pick the first entry
                best_pair = sorted(tiebreaker_dict.items(), key=lambda x: x[0], reverse=False)[0]

                # store the best pair into bpe_codes for visualization, value = i is the ranking at which the bpe_code was learnt.
                self.bpe_codes[best_pair[0]] = i


                print("------------------------------------------")
                print(f"Iteration {i}: ")
                # iterate through word_freq_dict and replace any part of the word within word_freq_dict that matches with the current best adjacent pair and updates our word_freq_dict.
                self.word_freq_dict = self.merge_byte_pairs(best_pair, self.word_freq_dict)
                # print(word_freq_dict)
                self.subword_tokens = self.get_subword_tokens(self.word_freq_dict)
                print(f"Best pair: {best_pair}")
                # print(f"Current vocabulary:{self.subword_tokens}")
                # print(f"Current length of {len(self.subword_tokens)}")
                print("------------------------------------------")

            # print(f'\nFinal Vocabulary: {self.subword_tokens}')
            # print(f'\nByte-Pair Encodings: {self.bpe_codes}')

            # return the bpe_codes for bpe_token_segmenter after learning the tokens from corpus.
            return self.bpe_codes
        
    
    # class for token segmenter for BPE
    class bpe_token_segmenter:

        def __init__(self, bpe_codes):
            self.bpe_codes = bpe_codes

        # gets the set of possible bigram symbol
        def get_pairs(self, word):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char

            return pairs

        # Creates the new words with merging
        def create_new_word(self, word, pair_to_merge):
            first, second = pair_to_merge
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(first)
                    i += 1

            return new_word


        # encompass all things together and encodes for a given bpe_codes that was initially given to bpe_token_segmenter class object.
        def encode(self, sentence):
            if len(sentence) == 1:
                return self.sentence

            word = list(sentence)
            word.append('_')

            while True:
                pairs = self.get_pairs(word)
                bpe_codes_pairs = [(pair, self.bpe_codes[pair]) for pair in pairs if pair in self.bpe_codes]
                if not bpe_codes_pairs:
                    break

                pair_to_merge = min(bpe_codes_pairs, key=itemgetter(1))[0]
                word = self.create_new_word(word, pair_to_merge)

            return word

    ###########################################################################################################################

    def tokenize(self):
        ''' Returns/Saves a set of word tokens for the loaded textual file
        After calling this function, you should get a vocab set for the tokenizer,
        which you can utilize later in the method of tokenize_sentence.

        For the default setting, we do not want to make use of BPE, but only REGEX, and use re. 
        
        make sure you consider cases of:
            1) Words ending with punctuation (e.g., 'hiking.' ——> ['hiking', '.']);
            2) Numbers (e.g., '1/2', '12.5')
            3) Possessive case (e.g., "Elle's book" ——> ["elle's", "book"]). It's also fine if you follow 
               nltk's output for this case, we won't make a strict rule here.
            4) Title abbrevation - Mr., Mrs., Ms. ——> you can either remove the '.' before tokenization 
               or remain the token as a whole (e.g., 'Mr.' ——> 'mr.'). 
               You don't need to consider other special abbr like U.S., Ph.D., etc.

            For other corner cases (e.g. emoticons such as :D, units such as $, and other ill-formed English), you can check the output of 
            nltk.word_tokenize for reference. We won't punish you if your results are not the same as its outputs on other cases.

        For the bpe setting, 
            1) Tune the number of iterations so the vocab size will be close to the 
               default one's (approximately, the vocab size is about 13,000)
            2) During merge, for sub-sequences of the same frequency, break the tie 
               with left-to-right byte order precedence - meaning to use lexicographical order (alphabetical order) to break the ties.
        
        PS: This method is mandatory to implement to get a vocab 
            which you can utilize in tokenize_sentence
        '''
        
        # TODO Modify the code here
        # remove the '.' before tokenization for title abbreviations
        title_abbreviations = ['Mr.', 'Mrs.','Ms.']
        for title in title_abbreviations:
            self.text = self.text.replace(title, title[:-1]) # removes the . behind Mr./Mrs./Ms.


        if bpe: # case where bpe is TRUE
            print('within bpe loop')

            if lowercase: # case where lowercase is TRUE
                # “
                # ”
                # ""
                # Since we are running into problems with the corpus handling “ or ”, we replace these in the corpus with " instead.
                # self.text = re.sub('“', '"', self.text)
                # self.text = re.sub('”', '"', self.text)
                # Instantiate the BPE token learner from the class bpe_token_learner
                btl = self.bpe_token_learner(self.text.lower())
                # generate the learnt vocab that we will use in tokenize_sentence method.
                self.vocab = btl.byte_pair_encoding(self.n_iterations) 
                self.subword_tokens = btl.subword_tokens
            else: # case where lowercase is FALSE
                btl = self.bpe_token_learner(self.text)
                self.vocab = btl.byte_pair_encoding(self.n_iterations)
                self.subword_tokens = btl.subword_tokens

        else: # case where bpe is FALSE, use REGEX for tokenization
            print('within regex loop')
            print(f"{bpe}")
            print(f"{lowercase}" )

            # The vocab should be a set of the original corpus tokens.
            if lowercase: # case where lowercase is TRUE
                self.corpus = re.findall(self.regex, self.text.lower())
                self.vocab = set(self.corpus)
                return(self.vocab)
            else: # case where lowercase is FALSE
                self.corpus = re.findall(self.regex, self.text)
                self.vocab = set(self.corpus)
                return(self.vocab)



    
    def tokenize_sentence(self, sentence):
        '''
        tokenize_sentence function here will make use of the function tokenize. You need to tokenize the sentence based on the vocab from the input corpus, as we use the corpus to init the the class Tokenizer.
        To verify your implementation, we will test this method by 
        input a sentence specified by us.  
        Please return the list of tokens as the result of tokenization.

        E.g. basic tokenizer (default setting)
        [In] sentence="I give 1/2 of the apple to my ten-year-old sister."
        [Out] return ['i', 'give', '1/2', 'of', 'the', 'apple', 'to', 'my', 'ten-year-old', 'sister', '.']
        
        Hint: For BPE, you may need to fix the vocab before tokenizing
              the input sentence
        
        PS: This method is mandatory to implement with the method signature as-is. 
        '''
        # TODO Modify the code here
        


        if self.bpe: # Case where bpe is true
            if lowercase: # for lowercase BPE token segmentation
                # initialize bpe token segmenter to take in vocab (learnt bpe codes) from large corpus
                bts = self.bpe_token_segmenter(self.vocab)
                self.valid_token_list = bts.encode(sentence.lower())

                return self.valid_token_list
            else: # for non-lowercase BPE token segmentation
                # initialize bpe token segmenter to take in vocab (learnt bpe codes) from large corpus
                bts = self.bpe_token_segmenter(self.vocab)

                return self.valid_token_list
            
        else: # Case where bpe is false, use regex
            if lowercase: # for lowercase REGEX
                # Use the vocabulary as in self.vocab to find the tokens that are required, using the same regex as before.
                sentence_token_list = re.findall(self.regex, sentence.lower())

                for token in sentence_token_list:
                    if token in self.vocab:
                        self.valid_token_list.append(token)
                    else:
                        self.valid_token_list.append('UNK') # unknown token that is not found in the corpus
                self.plot_word_frequency() # Plot Graph
                return self.valid_token_list
            
            else: # for non-lowercase REGEX
                # Use the vocabulary to find the tokens that are required, using the same regex as before.
                sentence_token_list = re.findall(self.regex, sentence)

                for token in sentence_token_list:
                    if token in self.vocab:
                        self.valid_token_list.append(token)
                    else:
                        self.valid_token_list.append('UNK') # unknown token that is not found in the corpus
                self.plot_word_frequency() # Plot Graph
                return self.valid_token_list

        
    
    def plot_word_frequency(self):
        '''
        Plot relative frequency (y) versus rank of word (x) to check
        Zipf's law
        You may want to use matplotlib and the function shown 
        above to create plots
        Relative frequency f = Number of times the word occurs /
                                Total number of word tokens
        Rank r = Index of the word according to word occurence list
        '''
        # TODO Modify the code here




        if bpe:
            f_list = [] # initailize freq list
            rank_list = [] # initialize rank list


            # get the length of all words in the corpus from our final vocab self.subword_tokens
            count_words = 0
            for key,val in self.subword_tokens.items():
                count_words += val
            

            # Sort the self.subword_tokens dictionary by descending order.
            rank_dict = {key: rank for rank, key in enumerate(sorted(self.subword_tokens, key=self.subword_tokens.get, reverse=True), 1)}

            # get an updated f_list
            for key,val in rank_dict.items():
                rank_list.append(np.log(val))
                f_list.append(np.log(self.subword_tokens.get(key)))

            # Plot chart using plt.subplot
            fig, ax = plt.subplots(nrows=1,ncols=1)
            ax.plot(rank_list, f_list, '.-')
            ax.grid()
            ax.set_xlabel('log(Rank of Word)')
            ax.set_ylabel('log(Relative Frequency)')
            ax.set_title('Relative Frequency vs Rank of Word (BPE')
            plt.show()
            
            

        else: # if bpe is false, use the other plotting method.
            # check the word list from self.corpus
            corpus_length = len(self.corpus) # total number of word tokens in corpus

            # get a dictionary for the {word: total number of times it appears}
            word_count_dict = {}
            freq_count_dict = {}

            # Initialize the frequency and rank list to be appended to.
            f_list = []
            rank_list = []

            # Search the corpus for each word and count them for word_count_dict
            for word in self.corpus:
                if word not in word_count_dict.keys():
                    # first occurrence of word, initialize the count to 1
                    word_count_dict[word] = 1
                else:
                    word_count_dict[word] += 1 # add 1 to the count if word already exists.

            # get frequency dictionary {key:word, value:frequency}
            for key in word_count_dict.keys():
                freq_count_dict[key] = word_count_dict[key] / corpus_length

            # get the rank / index of the word according to their frequency
            rank_dict = {key: rank for rank, key in enumerate(sorted(freq_count_dict, key=freq_count_dict.get, reverse=True), 1)}

            # Append to f_list and rank_list by looping through rank_dict
            for key,val in rank_dict.items():
                # Use np.log for both values
                rank_list.append(np.log(val))
                f_list.append(np.log(freq_count_dict.get(key)))

            # Plot chart using plt.subplot
            fig, ax = plt.subplots(nrows=1,ncols=1)
            ax.plot(rank_list, f_list, '.-')
            ax.grid()
            ax.set_xlabel('log(Rank of Word)')
            ax.set_ylabel('log(Relative Frequency)')
            ax.set_title('Relative Frequency vs Rank of Word (REGEX)')
            plt.show()





    
if __name__ == '__main__':

    # Get command line inputs
    path_input = sys.argv[1]
    bpe_input = sys.argv[2]
    lowercase_input = sys.argv[3]


    # Deal with our arguments coming from the command line
    if bpe_input.lower() == 'no':
        bpe = False
    elif bpe_input.lower() == 'yes':
        bpe = True
    else:
        raise ValueError('Invalid input for bpe')

    if lowercase_input.lower() == 'no':
        lowercase = False
    elif lowercase_input.lower() == 'yes':
        lowercase = True
    else:
        raise ValueError('Invalid input for lowercase')




    ##=== tokenizer initialization ===##
    basic_tokenizer = Tokenizer('../data/Pride_and_Prejudice.txt')
    bpe_tokenizer = Tokenizer('../data/Pride_and_Prejudice.txt', bpe=True)

    ##=== build the vocab ===##
    try:
        _ = basic_tokenizer.tokenize()  # for those which have a return value
    except:
        basic_tokenizer.tokenize()
    
    try:
        _ = bpe_tokenizer.tokenize()  # for those which have a return value
    except:
        bpe_tokenizer.tokenize()

    ##=== run on test cases ===##
    
    # you can edit the test_cases here to add your own test cases
    test_cases = ["""The Foundation's business office is located at 809 North 1500 West, 
        Salt Lake City, UT 84116, (801) 596-1887."""]

    for case in test_cases:
        rst1 = basic_tokenizer.tokenize_sentence(case)
        rst2 = bpe_tokenizer.tokenize_sentence(case)


        ##= check the basic tokenizer =##
        # ['the', "foundation's", 'business', 'office', 'is', 'located', 'at', 
        # '809', 'north', '1500', 'west', ',', 'salt', 'lake', 'city', ',', 'ut', 
        # '84116', ',', '(', '801', ')', '596-1887', '.']
        # or
        # ['the', 'foundation', "'s", 'business', 'office', 'is', 'located', 'at', 
        # '809', 'north', '1500', 'west', ',', 'salt', 'lake', 'city', ',', 'ut', 
        # '84116', ',', '(', '801', ')', '596-1887', '.']
        print(rst1)
        print('hi rst1')


        ##= check the bpe tokenizer =##
        # ['the_', 'f', 'ou', 'n', 'd', 'a', 'ti', 'on', "'", 's_', 'bu', 
        # 's', 'in', 'es', 's_', 'o', 'f', 'f', 'i', 'c', 'e_', 'is_', 'l', 
        # 'o', 'c', 'at', 'ed_', 'at_', '8', '0', '9', '_', 'n', 'or', 'th_', 
        # '1', '5', '0', '0', '_', 'w', 'es', 't', ',_', 's', 'al', 't_', 'l', 
        # 'a', 'k', 'e_', 'c', 'it', 'y', ',_', 'u', 't_', '8', '4', '1', '1', 
        # '6', ',_', '(', '8', '0', '1', ')', '_', '5', '9', '6', '-', '1', '8', 
        # '8', '7', '._']
        print(rst2)
        print('hi rst2')