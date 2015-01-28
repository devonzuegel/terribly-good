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
  # convention.
  class Example:
       words is a list of strings.
    def __init__(self):
      self.klass = ''
      self.words = []


 # NaiveBayes initialization
  def __init__(self):
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10


  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # TODO: If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts

  ## 
  # TODO:
  # 'words' is a list of words to classify. Return 'pos' or 'neg'
  # classification.
  def classify(self, words):
    
    return 'pos'
  

  ## 
  # TODO:
  # Train your model on an example document with label klass ('pos' or
  # 'neg') and words, a list of strings.
  #
  # You should store whatever data structures you use for your
  # classifier  in the NaiveBayes class.
  #
  # Returns nothing
  def addExample(self, klass, words):
    pass
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################  


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

  # Returns a lsit of TrainSplits corresponding to the cross validation splits
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
