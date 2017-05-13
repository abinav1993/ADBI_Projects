from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import string
import matplotlib.pyplot as plt


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
    #print pwords
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    
    positive = []
    negative = []
    for temp in counts:
	#print temp
	for t in temp:
	    if t[0] == "positive":
		positive.append(t[1])
	    else:
		negative.append(t[1])
    #print positive
    #print negative
    positive_line, = plt.plot(range(len(positive)),positive,"bo-", label="positive")
    negative_line, = plt.plot(range(len(negative)),negative,"go-", label="negative")
    plt.xticks(range(len(positive)))
    plt.legend(handles=[positive_line,negative_line],loc=2)
    plt.xlabel("Time step")
    plt.ylabel("Word count")
    #print("Before printing plots!")
    plt.show()


def load_wordlist(filename):
    f = open(filename,'rU')
    return f.read().split("\n")


def updateFunction(newValues, runningCount):
    if runningCount is None:
	runningCount = 0
    return sum(newValues, runningCount)

def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    #tweets.pprint()

    words = tweets.flatMap(lambda tweet: tweet.split(" ")) \
		.map(lambda word: word.strip(string.punctuation).lower()) \
                .filter(lambda word:True if word in pwords or word in nwords else False)
                                
    #words.pprint()
    words = words.map(lambda word:("positive",1) if word in pwords else ("negative",1)) \
                .reduceByKey(lambda x,y : x+y)
    temp = words.updateStateByKey(updateFunction)
    temp.pprint()

    counts = []
    words.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    #print counts
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts

if __name__=="__main__":
    main()
