import csv
import sys
import random
from math import exp
random.seed(0)

if len(sys.argv) != 2:
	print "Usage: adwords.py <Greedy|MSSV|Balance>"
	exit(1)

def main():
	method = sys.argv[1]
	queries = open('queries.txt')
	qList = []
	spends = {}
	budgets = {}
	query = {}
	for q in queries:
		qList.append(q.strip())
	#print len(qList)

	with open('bidder_dataset.csv') as bidding:
		reader = csv.reader(bidding)
	# Skipping the header line
		next(reader) 
		for row in reader:
			advid = int(row[0])
			bid= float(row[2])
			if row[1] not in query:
				query[row[1]] = [(advid, bid)]
			else:
				query[row[1]].append((advid,bid))

			if advid not in budgets:
				budget = float(row[3])
				budgets[advid] = budget
				spends[advid] = 0.0
		#print len(budgets)
		#print len(query)
	
	
	#print query
	opt = 0.0
	als = 0.0
	for temp in budgets:
		opt = opt + budgets[temp]

	(result, als) = selectMethod(method, qList, query, budgets, spends)
	print result
	#print als
	#print opt
	competitve_ratio = als/opt
	print competitve_ratio
		
def selectMethod(method, qList, query, budgets, spends):
# As we are modifying the contents of the query map and budgets map in every call to Greedy()/MSSV()/Balance(), we need to send a copy of 
# the data structures so that subsequent calls to Greedy() will not produce unexpected results.
	queryCopy = query.copy()
	budgetsCopy = budgets.copy()
	als = 0
	result = 0

	if method == "Greedy":
		for key in query:
			query[key] = sorted(query[key], key = lambda x:x[1], reverse = True)
		queryCopy = query.copy()
		budgetsCopy = budgets.copy()

		result = Greedy(qList, queryCopy, budgetsCopy)
		for i in range(1,100):
			queryCopy = query.copy()
			budgetsCopy = budgets.copy()
			random.shuffle(qList)
			als = als + Greedy(qList, queryCopy, budgetsCopy)
	
	elif method == "Balance":
		result = Balance(qList, queryCopy, budgetsCopy)
		for i in range(1,100):
			queryCopy = query.copy()
			budgetsCopy = budgets.copy()
			random.shuffle(qList)
			als = als + Balance(qList, queryCopy, budgetsCopy)
	
	elif method == "MSSV":
		spendsCopy = spends.copy()
		result = MSSV(qList, queryCopy, budgetsCopy, spendsCopy)
		for i in range(1,100):
			queryCopy = query.copy()
			budgetsCopy = budgets.copy()
			spendsCopy = spends.copy()
			random.shuffle(qList)
			als = als + MSSV(qList, queryCopy, budgetsCopy, spendsCopy)
	else:
		print("Enter valid argument")
		exit(1)

	als = als / 100
	return (result,als)
		



def Greedy(qList, query, budgets):
	revenue = 0
	for q in qList:
		tempList = query[q]
		for item in tempList:
			initial = budgets[item[0]]
			if initial - item[1] >= 0:
				budgets[item[0]] = initial - item[1]
				revenue = revenue + item[1]
				break
	#print revenue
	return revenue 


def MSSV(qList, query, budgets, spends):
	revenue = 0
	for q in qList:
		tempList = query[q]
		max_value = -1
		adv = 0
		bid = 0
		for item in tempList:
			psi = 1 - exp(spends[item[0]] / budgets[item[0]] - 1)
			value = psi * item[1]
			if value > max_value and (spends[item[0]] + item[1]) <= budgets[item[0]]:
				max_value = value
				adv = item[0]
				bid = item[1]
		
		spends[adv] = spends[adv] + bid
		revenue = revenue + bid
	return revenue


def Balance(qList, query, budgets):
	revenue = 0
	for q in qList:
		tempList = query[q]
		max_unspent = -1
		bid = 0
		adv = 0
		for item in tempList:
			if budgets[item[0]] > max_unspent:
				max_unspent = budgets[item[0]]
				adv = item[0]
				bid = item[1]
		
		if max_unspent - bid >= 0:
			budgets[adv] = max_unspent - bid
			revenue = revenue + bid
	return revenue


if __name__ == "__main__":
	main()
