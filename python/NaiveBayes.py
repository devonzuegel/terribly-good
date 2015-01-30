import sys
import getopt
import os
import math

class NaiveBayes:
  ##
  # Represents a set of training/testing data. self.train is a list of
  # Examples, as is self.test. 
  class TrainSplit:
    def __init__(self):
      self.train = []
      self.test = []

  ##
  # Represents a document with a label. klass is 'pos' or 'neg' by
  # convention. 'words' is a list of strings.
  class Example:
    def __init__(self):
      self.klass = ''
      self.words = []

  # NaiveBayes initialization
  def __init__(self):
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10

    # The set of all words found in the training set
    self.vocab = set()

    ##
    # Counts the number of positive & negative docs from the training set,
    # which divided by the total number of docs (self.nDocs) gives us the
    # priors for finding positive and negative documents.
    self.doc_counts = { 'pos': 0.0, 'neg': 0.0 }

    ##
    # Counts the number of times a given word occurs in positively- or
    # negatively-classified review from the training set.
    self.polarized_words = { 'pos': {}, 'neg': {} }

    # Counts the number of positive and negative words in the corpus
    self.word_counts = { 'pos': 0.0, 'neg': 0.0 }


  #############################################################################
  
  def classify(self, words):

    nDocs = self.doc_counts['pos'] + self.doc_counts['neg']
    vocab_size = len(self.vocab)

    ##
    # The baseline scores are the priors for each class (the percentage
    # of training docs that fell under that class out of the total number
    # of documents). These scores will be updated later, when we iterate
    # through the words.
    scores = { 
      'pos': math.log(self.doc_counts['pos'] / nDocs),
      'neg': math.log(self.doc_counts['neg'] / nDocs)
    }

    for word in words:
      for k in ['pos', 'neg']:
        ##
        # Calculate the conditional probability that this word would
        # occur within the given class k and add it to the logprob score
        if (word in self.polarized_words[k]):
          word_count = 1 if (self.BOOLEAN_NB) else self.polarized_words[k][word]
          cond_prob = (word_count + 1.0) / (self.word_counts[k] + vocab_size)
          scores[k] += math.log(cond_prob)
        else:
          scores[k] += math.log(1.0 / (self.word_counts[k] + vocab_size))

    if (scores['pos'] > scores['neg']):   return 'pos'
    else:                                 return 'neg'
  

  def addExample(self, klass, words):


    # Each call to addExample is another doc
    # Increment doc_count for that klass (pos/neg)
    self.doc_counts[klass] += 1.0

    for word in words:
      # Add this word to the vocabulary if it isn't already in it
      if (word not in self.vocab):    self.vocab.add(word)

      # Increment count for pos/neg word_counts according to klass
      self.word_counts[klass] += 1.0

      # If the word is not already in the list of polarized words, insert it
      if (word not in self.polarized_words[klass]):
        self.polarized_words[klass][word] = 0.0

      ##
      # Increment the count for the number of times that word is found in
      # a certain class (pos/neg) of reviews
      self.polarized_words[klass][word] += 1.0

  ##
    # Code for reading a file.  you probably don't want to modify anything
    # here, unless you don't like the way we segment files.
  def readFile(self, fileName):
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result


  # Splits lines on whitespace for file reading  
  def segmentWords(self, s):
    return s.split()

  # Takes in a trainDir, returns one TrainSplit with train set.  
  def trainSplit(self, trainDir):
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)

  # Returns a list of TrainSplits corresponding to the cross validation splits
  def crossValidationSplits(self, trainDir):
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits

  # Returns a list of labels for split.test.
  def test(self, split):
    labels = []
    for example in split.test:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      guess = self.classify(words)
      labels.append(guess)
    return labels
  
  # Builds the splits for training/testing
  def buildSplits(self, args):
    trainData = [] 
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1: 
      print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName)) 
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName)) 
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits
  
  # Filters stop words.
  def filterStopWords(self, words):
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    accuracy = 0.0
    for example in split.train:
      words = example.words
      if FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      classifier.addExample(example.klass, words)

    # print_trained(classifier)
  
    for example in split.test:
      words = example.words
      if FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0
    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy


# Sanity-checks the output of the training function
def print_trained(classifier):
  print 'len(classifier.vocab): ' + str(len(classifier.vocab))
  print 'classifier.doc_counts: ' + str(classifier.doc_counts)
  print 'classifier.polarized_words: ' + str(classifier.polarized_words)
  print 'classifier.word_counts: ' + str(classifier.word_counts)

def classifyFile(FILTER_STOP_WORDS, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print classifier.classify(testFile)
    
def main():
  FILTER_STOP_WORDS = False
  (options, args) = getopt.getopt(sys.argv[1:], 'f')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  
  if len(args) == 2 and os.path.isfile(args[1]):
    classifyFile(FILTER_STOP_WORDS, args[0], args[1])
  else:
    test10Fold(args, FILTER_STOP_WORDS)

if __name__ == "__main__":
    main()
