'''
    NUS CS4248 Assignment 1 - Objective 3 (n-gram Language Model)

    Class NgramLM for handling Objective 3

    Important: please strictly comply with the input/output formats for
               the methods of generate_word & generate_text & get_perplexity, 
               as we will call them during testing

    Sentences for Task 3:
    1) "They just entered a beautiful walk by"
    2) "The rabbit hopped onto a beautiful walk by the garden."
    3) "They had just spotted a snake entering"
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import random, math
import collections
import re # important library for regex
import sys # to retrieve args from command line
import nltk
import numpy as np


# Based on Markov Assumptions to form ngram model
class NgramLM(object):

    def __init__(self, path: str, n: int, k: float):
        '''This method is mandatory to implement with the method signature as-is.

            Initialize your n-gram LM class

            Parameters:
                n (int) : order of the n-gram model
                k (float) : smoothing hyperparameter

            Suggested function dependencies:
                read_file -> init_corpus |-> get_ngrams_from_seqs -> add_padding_to_seq
                                         |-> get_vocab_from_tokens

                generate_text -> generate_word -> get_next_word_probability

                get_perplexity |-> get_ngrams_from_seqs
                               |-> get_next_word_probability

        '''
        # Initialise other variables as necessary
        # TODO Write your code here
        self.n = n # ngram model, either unigram for n = 1 or bigram model for n = 2
        self.k = k # amount of smoothing to be applied to counts of words. if k = 1, add this float to counts of individual words!

        # Update self.n & self.k from the command line inputs
        self.n = ngram_input
        self.k = k_input

        # initialize the set of vocabulary
        self.vocab = collections.defaultdict(int) # Why doesnt this list work?

        # initialize ngram dictionary
        self.ngram_dict = collections.defaultdict(int)

        self.special_tokens = {'bos': '~', 'eos': '<EOS>'} # bos - beginning of sentence, eos - end of sentence

        # initialize the unigram dictionary to be used for generating word probabilities for bigram case.
        self.unigram_counts = collections.defaultdict(int)

        # initialize values of N - total no. of counts of bigram/words occurrence & V - total no. of counts of unigram/vocab.
        self.N = 0
        self.V = 0

        # initialize the text_ngram to be used everywhere in the class
        self.text_ngram = ''

        # initialize the bigram_to_find to be used everywhere in the class
        self.bigram_to_find = ''

        # initialize the reading of the file (from hint)
        self.read_file(path)



        

    def read_file(self, path: str):
        ''' Reads text from file path and initiate n-gram corpus.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of __init__ 
        '''
        # TODO Write your code here
        # First open the file and thereafter, call the init_corpus function as mentioned in the hint.
        with open(path, 'r', encoding = 'utf8') as f:
            text = f.read()
            self.init_corpus(text.lower()) # perform case folding here so that all text will be treated the same.

    def init_corpus(self, text: str):
        ''' Initiates n-gram corpus based on loaded text
        
        PS: Change the function signature as you like. 
            This method is only a suggested method,
            which you may call in the method of read_file 
        '''
        # TODO Write your code here

        # from hint in get_vocab_from_tokens function, to get the tokens, we first need to tokenize the corpus.
        # for this purpose, we will make use of NLTK's text tokenizer to get the tokens instead of just splitting by whitespaces.
        tokenizer = nltk.tokenize.word_tokenize
        tokens = tokenizer(text)
        
        # get vocab from tokens using get_vocab_from_tokens (dictionary with key: token, value: counts of token)
        self.vocab = self.get_vocab_from_tokens(tokens)

        # get the ngrams from unigram/bigram model by using get_ngrams_from_seqs method and first split the sentences from the corpus as in the hint. Remove \n.
        ngrams = self.get_ngrams_from_seqs(text.strip().split('\n'))

        # updating ngram dictionary for learnt unigram/bigram
        for ngram in ngrams:
            self.ngram_dict[ngram] += 1 # adding to counts of ngram_dict for each key.


        

    def get_vocab_from_tokens(self, tokens):
        ''' Returns the vocabulary (e.g. {word: count}) from a list of tokens

        Hint: to get the vocabulary, you need to first tokenize the corpus.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of init_corpus.
        '''
        # TODO Write your code here
        
        # go through all the tokens in our text after stripping and splitting by white space in init_corpus and retrieve our vocab with counts of token occurrences
        for token in tokens:
            self.vocab[token] += 1
        return self.vocab


    def get_ngrams_from_seqs(self, sentences):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)] 
            where sequence context is the ngram and word is its last word

        Hint: to get the ngrams, you may need to first get split sentences from corpus,
            and add paddings to them.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of init_corpus 
        '''
        # TODO Write your code here

        # add paddings to the input sentences using add_padding_to_seq function
        padding_sentences = [self.add_padding_to_seq(sentence) for sentence in sentences]
        ngrams = []
        
        # go through the padded sentences:
        for sentence in padding_sentences:
            # split sentences by white space first
            tokens = sentence.split()
            # for each of the token found in each senence, loop through and retrieve ngram, and append the current ngram to the ngrams list.
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n]) # token is either unigram or bigram for this question where n in [1,2] i:i+1 or i:i+2
                ngrams.append((ngram[:-1], ngram[-1])) # append to the ngrams list a set consisting of the sequence context (tuple) and the last word of the ngram (str)
        
        # print(f"Last entry's sequence context of ngram: {ngram[:-1]}, {type(ngram[:-1])}")
        # print(f"Last entry's last word of ngram: {ngram[-1]}, {type(ngram[-1])}")

        return ngrams

    def add_padding_to_seq(self, sentence: str):
        '''  Adds paddings to a sentence.
        The goal of the method is to pad start token(s) to input sentence,
        so that we can get token '~ I' from a sentence 'I like NUS.' as in the bigram case.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of get_ngrams_from_seqs 
        '''
        # In higher n-gram language models, the words at the start of each sentence will not have long enough context to apply the formula for bigram.
        # hence the need for padding to the tart of the sentences with `~` sign.
        # TODO Write your code here
        # Use '~' as your padding symbol - coming from special_tokens dictionary defined in __init__
        return self.special_tokens['bos'] * (self.n - 1) + ' ' + sentence




    ### FINAL get_next_word_probability
    def get_next_word_probability(self, text: str, word: str):
        ''' Returns probability of a word occurring after specified text, 
        based on learned ngrams.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of generate_word         
        '''

        # The text input coming from here will be the context from generate_word depending on bigram or unigram case
        # context for bigram - last token of the text.
        # context for unigram - whole text.


        # TODO Write your code here
        # TOKENIZE not required here, as we already perform tokenization to give the context we need for generating word probabilities

        # ngram to look at for either unigram or ngram case
        # if bigram, it will take the last token of sentence as the unigram
        # if unigram, it will take an empty string.


        if self.n == 2:
            # bigram case, take the last token as the unigram to be predicted for, as the word's probability depends on the single word before it.
            # the text coming into generate_word_probabilities is already the sequence context, which happens to be the last token of the sentence.
            self.text_ngram = tuple(text)

            # check from the bigram counts between the last unigram of the text and the given word, see if it exists in the ngram_dict
            # if yes, take the counts from there, else 0 count.
            self.bigram_to_find = (self.text_ngram, word) # creating a tuple from 2 values. The first entry is to be a tuple to fit the format that we stored the ngram_dict in which is e.g (('~',), 'the'): 356
            count_bigram = self.ngram_dict[self.bigram_to_find] if self.bigram_to_find in self.ngram_dict else 0 

            # search for whether our text_ngram exists in the unigram_counts dict, if yes, take the counts from there, else 0 count
            count_unigram = self.unigram_counts[self.text_ngram] if self.text_ngram in self.unigram_counts else 0


            # get the probabilities of word occuring after the statement.
            
            # make sure to handle the corner case whereby k = 0 that might lead to 0 probabilities and facing errors when taking logarithm.
            if (count_bigram == 0 and self.k == 0) or (count_unigram == 0 and self.k == 0):
                prob = 1e-20 # assign small probabilities
            else:
                prob = (count_bigram + self.k) / (count_unigram + self.k * self.V)

            return prob

        elif self.n == 1: # unigram case, training the model is nothing but calculating the fractions for all unigrams in the training text.
            # Our self.vocab is our training text, our text sentence that we will see is the evaluation text.
            # the probability of a sentence is the product of the probability of words
            # IMPT: the probability of the word occurring next in the text is just the probability of the word occurring in the training text itself.

            # check if the word given is existing inside the current vocabulary, else assign 0 count.
            # raw count of the word in our vocabulary, the word is our unigram
            count_unigram = self.vocab[word] if word in self.vocab else 0

            # calculate probability of the word, since it is the unigram case
            # make sure to hadle the corner case whereby k = 0 that might lead to 0 probabilities and facing errors when taking logarithm.
            if (count_unigram == 0 and self.k == 0):
                prob = 1e-20 # assign small probabilities to prevent divide by zero error.
            else:
                prob = (count_unigram + self.k) / (self.N + self.k * self.V)

            return prob

            


    def generate_word(self, text: str):
        '''
        Generates a random word based on the specified text and the ngrams learned
        by the model

        PS: This method is mandatory to implement with the method signature as-is.
            We only test one sentence at a time, so you may not need to split 
            the text into sentences here.

        [In] string (a full sentence or half of a sentence)
        [Out] string (a word)
        '''
        # TODO Write your code here
        # As mentioned in the text, we only test one sentence at a time, so no splitting of text into sentences is required.

        # First get the ngram to be looked at, which is either unigram or bigram.
        tokenizer = nltk.tokenize.word_tokenize
        tokens = tokenizer(text.lower()) # do case-folding to lower letters


        # bigram case
        if self.n == 2:
            # if ngram is a bigram, use the last word in the text as the sequence context
            seq_context = tokens[-self.n+1:]


            # updating the unigram_counts dictionary which is to be used in get_next_word_probability
            for bigram, count in self.ngram_dict.items():
                w1, w2 = bigram
                self.unigram_counts[w1] += count

            # getting the values of N and V to be used for the calculation of probabilities in get_next_word_probability
            self.N = sum(self.unigram_counts.values()) # total number of bigrams
            self.V = len(self.unigram_counts) # total number of unigrams - which is our vocab for bigram case.

            # dictionary comprehension to get probabilities of next word by considering the context given for bigram case, which is the last token.
            probabilities = {word: self.get_next_word_probability(seq_context, word) for word in self.vocab}

            total_prob = sum(probabilities.values())

            # Normalize the probabilities
            probabilities = {word: probability / total_prob for word, probability in probabilities.items()}

            # Create normal distribution
            mean = sum([prob * i for i, prob in enumerate(probabilities.values())])
            variance = sum([prob * ((i - mean) ** 2) for i, prob in enumerate(probabilities.values())])
            sigma = variance ** 0.5
            word_index = int(random.gauss(mean, sigma))

            # Use the probabilities to generate the word by using a normal distribution instead of random.choices. 
            # Pick the word with the closest index to the generated index.
            random_word = [word for word in probabilities.keys()][min(len(probabilities) - 1, max(0, word_index))]

            return random_word
            
            # # Use a normal distribution to get our random word

            # # Find the mean of the probabilities
            # mean = sum(probabilities.values()) / len(probabilities.values())

            # # Find the standard deviation of the probabilities
            # std = np.std(list(probabilities.values()))

            # # Generate a random word from a normal distribution with mean and standard deviation calculated above
            # random_word = np.random.normal(mean, std, 1)

            # # Find the word with the closest probability to the randomly generated number
            # closest_word = min(probabilities, key=lambda x: abs(probabilities[x]-random_word))

            # return closest_word



            # generate random word with our probability dictionaries
            # random_word = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)

            # return the first word choice from the list, which represents the word with the highest probability
            # return random_word[0]

        # unigram case
        elif self.n == 1: 
            # if the ngram is a unigram, use the entire text as the context.
            seq_context = text

            # getting the values of N and V to be utilized in get_next_word_probability
            self.N = sum(self.vocab.values()) # total number of words in vocab for unigram case, in the training text which is our vocab.
            self.V = len(self.vocab) # vocab size
            
            # dictionary comprehension to get probabilities of next word by considering the context, which is the whole text for unigram case. Does not make a difference.
            probabilities = {word: self.get_next_word_probability(seq_context, word) for word in self.vocab}

            total_prob = sum(probabilities.values())

            # normalize the probabilities
            probabilities = {word: probability / total_prob for word, probability in probabilities.items()}


            # Create normal distribution
            mean = sum([prob * i for i, prob in enumerate(probabilities.values())])
            variance = sum([prob * ((i - mean) ** 2) for i, prob in enumerate(probabilities.values())])
            sigma = variance ** 0.5
            word_index = int(random.gauss(mean, sigma))

            # Use the probabilities to generate the word by using a normal distribution instead of random.choices. 
            # Pick the word with the closest index to the generated index.
            random_word = [word for word in probabilities.keys()][min(len(probabilities) - 1, max(0, word_index))]

            return random_word

                # Find the mean of the probabilities
            # mean = sum(probabilities.values()) / len(probabilities.values())

            # # Find the standard deviation of the probabilities
            # std = np.std(list(probabilities.values()))

            # # Generate a random word from a normal distribution with mean and standard deviation calculated above
            # random_word = np.random.normal(mean, std, 1)

            # # Find the word with the closest probability to the randomly generated number
            # closest_word = min(probabilities, key=lambda x: abs(probabilities[x]-random_word))

            # # return the closest word
            # return closest_word

            # # Use the probabilities to generate the wword and pick the first word in the list representing the highest probabilities
            # random_word = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)

            # return random_word[0]



    def generate_text(self, length: int):
        ''' Generate text of a specified length based on the learned ngram model 
        
        [In] int (length: number of tokens)
        [Out] string (text)

        PS: This method is mandatory to implement with the method signature as-is. 
            The length here is a reasonable int number, (e.g., 3~20)
        '''
        # TODO Write your code here
        # Start from blank text.
        text = ''

        # iterate through our length of words/tokens required after a given text
        for i in range(length):
            # use the current text as the context to generate the next word.
            next_word = self.generate_word(text)
            
            # add the next word to the text.
            text += ' ' + next_word

        return text


    def get_perplexity(self, text: str):
        '''
        Returns the perplexity of texts based on learned ngram model. 
        Note that text may be a concatenation of multiple sequences.
        
        [In] string (a short text)
        [Out] float (perplexity) 

        PS: This method is mandatory to implement with the method signature as-is. 
            The output is the perplexity, note the log form you use to avoid 
            numerical underflow in calculation.

        Hint: To avoid numerical underflow, add logs instead of multiplying probabilities.
              Also handle the case when the LM assigns zero probabilities. -- to use a very small probability!
        '''

        # Perplexity refers to the inverse probabilty of the TEST corpus
        # TODO Write your code here

        # Handle the case whereby text may be a concatenation of multiple sequences. We also need to add padding for all cases of our TEST CORPUS.
        # use our learnt get_ngram_from_seqs to apply to our text and get the list of ngrams.
        test_ngrams = self.get_ngrams_from_seqs(text.strip().split('\n'))
        # print(f"list of test_ngrams for test_corpus: {test_ngrams}")

        # loop through our test_ngrams list and find the probabilities, and add to our log_prob_sum
        log_prob_sum = 0 # initialize log probability sum to 0 initially.
        N = len(test_ngrams) # Normalization constant

        for ngram in test_ngrams:
            # our context for perplexity:
            context = ngram[0] # context will be blank for unigram case, and context will be word for the bigram case.
            word = ngram[-1] # word will be the second entry of the tuple.
            # print(f"Context: {context}, Word: {word}")

            prob = self.get_next_word_probability(context,word)

            log_prob_sum += -1 * math.log(prob, 2) # log base 2 of probability then sum them up and accumulate log_prob_sum

        perplexity = 2 ** (log_prob_sum / N) # N is the normalization factor. Move the base 2 of log to RHS and it becomes 2 to the power.

        return perplexity




if __name__ == '__main__':
    print('''[Alert] Time your code and make sure it finishes within 2 minutes!''')

    # Get command line inputs
    print(sys.argv)
    path_input = sys.argv[1]
    smooth_input = sys.argv[2] # add-k smoothing input. default is 'add-k'
    ngram_input = int(sys.argv[3]) # handle ngrams, can either be [1,2]
    k_input = float(sys.argv[4]) # decides on the amount of smoothing to use. Default is 1.0
    text_input = sys.argv[5] # must be a text on which your ngram language model will predict the next words.


    # Deal with our arguments coming from the command line
    if smooth_input == 'add-k':
        pass
    else:
        raise ValueError('Invalid input for smooth')

    # Check if ngram input is either 1 or 2 only given at the command line, else give invalid input error.
    if ngram_input in [1,2]:
        n = ngram_input # set the value of n to be value of ngram_input
    else:
        raise ValueError('Invalid input for ngram')

    # check if k_input is between 0 and 1 inclusive at the command line, else give invalid input error.
    if 0 <= k_input <= 1.0:
        k = k_input # set the value of k to be the value of k_input
    else:
        raise ValueError('Invalid input for k')

    LM = NgramLM('../data/Pride_and_Prejudice.txt', n=2, k=1.0)

    test_cases = ["The rabbit hopped onto a beautiful walk by the garden.", 
        "They just entered a beautiful walk by", 
        "They had just spotted a snake entering"]

    for case in test_cases:
        word = LM.generate_word(case)
        ppl = LM.get_perplexity(case)
        print(f'input text: {case}\nnext word: {word}\nppl: {ppl}')
    
    _len = 7
    text = LM.generate_text(length=_len)
    print(f'\npredicted text of length {_len}: {text}')