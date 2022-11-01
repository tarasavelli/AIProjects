
## author: Teddy Arasavelli

import os
import math






#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    ## create and return a bag of words from a single document
    bow = {}

    # TODO: add your code here
    fileDocument = open(filepath)

    for line in fileDocument:
        word = line.strip()
        if word in vocab: ## case where word is contained in vocabulary
            if(word in bow):
                bow.update({word : bow.get(word) + 1})
            else:
                bow.update({word : 1})
        else: ## case where word isnt contained in vocab, must add a count to the 'None' key
            if None in bow:
                bow.update({None : bow.get(None) + 1})
            else:
                bow.update({None : 1})

            

        
    
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}

    # TODO: add your code here

    ## counting the number of files that are labeled 2016 and 2020 (generalized to be able to work with more subdirectories)

    countsByYear = {}

    ## calculating the probablity of a document being written in the years given by the labe list.
    for dictionary in training_data:
        label = dictionary.get("label")
        if label in countsByYear:
            oldCount = countsByYear.get(label)
            countsByYear.update({label : oldCount + 1})
        else:
            countsByYear.update({label : 1})
    

    total = len(training_data)

    ## taking natural log of probabilities while applying smoothing technique 
    for year in countsByYear:
        count = countsByYear.get(year)
        logProbYear = math.log((count + smooth)/(total + 2))
        logprob.update({year : logProbYear})

    
    return logprob

def separate_data_by_label(training_data, label):
    labelData = {} ## should return a dictionary of all the words that are contained in all the doocuments with the label specified. Value should be the total count

    for dictionary in training_data:
        if dictionary.get("label") == label:
            bow = dictionary.get("bow")
            
            for word in bow:
                if word in labelData:
                    labelData.update({word : labelData.get(word) + bow.get(word)})
                else:
                    labelData.update({word : bow.get(word)})
                

    return labelData


#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here

    ## go through all words in vocab
    ## check them against the training_data (filter by label) and look through each file

    ## creating dictionary that only contains words from files associated with label
    wordCountsByLabel = separate_data_by_label(training_data, label)
    
    ## getting the total number oof words regardless of it being in the vocab or not
    ## must also get the count of all words not in vocabulary but are still in the files associated with the label
    noneCount = 0
    totalWords = 0
    for key in wordCountsByLabel:
        if not key in vocab:
            noneCount += wordCountsByLabel.get(key)
        totalWords += wordCountsByLabel.get(key)

    
    ## following loop will handle the words in vocab

    ## getting probabilities of words in vocab appearing in documents with appropriate label
    for word in vocab:
        wordCount = 0
        if word in wordCountsByLabel:
            wordCount += wordCountsByLabel.get(word)
            
        pWordGivenLabel = (wordCount + smooth) / (totalWords + len(vocab) + smooth)
        logPgivenLabel = math.log(pWordGivenLabel)
        word_prob.update({word:logPgivenLabel})



    ## adding the 'None' probability manually
    pNoneGivenLabel = (noneCount + smooth) / (totalWords + len(vocab) + smooth)
    logPNoneGivenLabel = math.log(pNoneGivenLabel)
    word_prob.update({None :logPNoneGivenLabel})

    return word_prob

    



##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)

     # TODO: add your code here

    
    trainingVocab = create_vocabulary(training_directory, cutoff)
    trainingData = load_training_data(trainingVocab, training_directory)

    logPrior = prior(trainingData, label_list)

    pWordsGiven2016 = p_word_given_label(trainingVocab, trainingData, "2016")
    pWordsGiven2020 = p_word_given_label(trainingVocab, trainingData, "2020")

    retval.update({"vocabulary" : trainingVocab})
    retval.update({"log prior" : logPrior})
    retval.update({"log p(w|y=2016)" : pWordsGiven2016})
    retval.update({"log p(w|y=2020)" : pWordsGiven2020})

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here

    testDoc = open(filepath)

    ## sum of probabilities that will end up being the final answer
    sumLogProb2016 = 0
    sumLogProb2020 = 0

    
    ## looping through each line and seeing if word is in vocab. If it is, add log prob of word being in a 2016/2020 document. If not, add log prob of the OOV (None) to both sums
    for line in testDoc:
        word = line.strip()
        if word in model.get("vocabulary"):
            wordProb = model.get("log p(w|y=2016)").get(word)
            sumLogProb2016 += wordProb
            wordProb = model.get("log p(w|y=2020)").get(word)
            sumLogProb2020 += wordProb
        else:
            sumLogProb2016 += model.get("log p(w|y=2016)").get(None)
            sumLogProb2020 += model.get("log p(w|y=2020)").get(None)

    ## adding the probability of document being of year 2016/2020
    sumLogProb2016 += model.get("log prior").get("2016")
    sumLogProb2020 += model.get("log prior").get("2020")

    predictedY = "2016"

    ## choosing the probability that is the highest
    if sumLogProb2020 > sumLogProb2016:
        predictedY = "2020"


    retval.update({"predicted y" : predictedY})
    retval.update({"log p(y=2016|x)" : sumLogProb2016})
    retval.update({"log p(y=2020|x" : sumLogProb2020})




    return retval


