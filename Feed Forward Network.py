from kaggle import CIFAR10
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

train = CIFAR10('train', datapath='/home/jvujjini/Desktop/CIFAR/')
dataSet = ClassificationDataSet(train.images.shape[1], nb_classes=train.n_classes, class_labels=train.label_names)

for image, image_class in zip(train.images,train.y):
    dataSet.appendLinked(image, image_class.nonzero()[0])

dataSet._convertToOneOfMany( )

print "\nTraining"

feedForwardNetwork = buildNetwork(dataSet.indim, 5, dataSet.outdim, outclass=LinearLayer)
classifier = BackpropTrainer(feedForwardNetwork, dataset=dataSet, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(5):
    classifier.trainEpochs(1)
    result = percentError(classifier.testOnClassData(),dataSet['class'])
    print "epoch# %4d" % classifier.totalepochs, "  train error: %5.2f%%" % result