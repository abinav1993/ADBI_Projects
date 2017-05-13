require(igraph);
require(lsa);

# Collaborators: I have discussed the implementation approach with the below people. 
# 1. Ravali Pothireddy
# 2. Prashanth Rallapalli
# 3. Shiv shankar Barai
# NOTE: I have made following changes to the evaluation.R file:
# 1. The index of the vertices were increased by 1 in the original R file. 
#    It casued has caused some errors so, I have commented that incrementing part. 
# 2. Used the setwd() function to point to my local directory where the "communities.txt" file is present.
# I'm adding the evaluation.R script (that worked for me) along with the all the other documents.

# Taking alpha value from command line.
args = commandArgs(trailingOnly = TRUE)
if(length(args) == 0){
  stop("Enter the alpha value between 0 & 1", call. = TRUE)
}

setwd("C:/Users/Abinav/Google Drive/2nd Semester/BI/Projects/Project - 6/06.Topic-7.Project-6.MarketSegmentation.AttributedGraphCommunityDetection/data")

alpha <- as.numeric(args[1])
#alpha <- 1
#print(alpha)

# Constructing graph from the edgelist.
graph <- read.graph("fb_caltech_small_edgelist.txt", format = c("edgelist"))
attributeData <- read.csv("fb_caltech_small_attrlist.csv")

# Setting the maximum number of iterations, incase if the algorithm does not converge.
max_iterations <- 15

# Initial number of communities is equal to the number of nodes in the graph
communities <- 1:vcount(graph)

# Function to check the change in modularity, by moving a node from 1 cluster to another cluster.
# It returns the difference between the new modularity and the old modularity.
modularityDifference <- function(communities, i, j){
  temp <- communities
  oldModularity <- modularity(graph,temp)
  temp[i] <- j
  newModularity <- modularity(graph, temp)
  return (newModularity - oldModularity)
}


# Function to compute the phase1 of the algorithm.
Phase1 <- function(graph, communities, attributeData, alpha){
  
  oldCommunityStructure <- communities
  
  for(k in 1:max_iterations){
    print(k)
    
    for(i in 1:vcount(graph)){
      
      neighbors_i <- neighbors(graph,i)
      maxGain <- 0
      finalIndex <- 0
      
      for(j in unique(communities[neighbors_i])){
        
        difference <- modularityDifference(communities, i, j)
        indices <- which(communities == j)
        #print(length(indices))
        
        # Calculating the similarity between the nodes present in the cluster and the newly moved node from the 
        # other cluster.
        
        similarity = 0
        for(x in indices){
          similarity <- similarity + cosine(as.numeric(attributeData[i,]), as.numeric(attributeData[x,])) 
        }
        similarity <- similarity/length(indices)
        
        # Calculating gain as mentioned in the paper.
        gain <- (alpha * difference) + ((1 - alpha) * similarity)
        
        # Storing the maximum gain seen so far and its corresponding cluster number.
        if(gain > maxGain && i != j){
          maxGain <- gain
          finalIndex <- j
        }
        
      }
      # Assigning the current node to the cluster that produced maximum gain.
      if(finalIndex != 0){
        communities[i] <- finalIndex
      }
    }
    # Reducing the number of iterations, by checking the structure of the graph before and after the iteration.
    # If the graph strucutre hasn't changed in the current iteration, it implies that the algorithm has converged
    # and hence we can stop the Phase 1.
    if(all.equal(oldCommunityStructure, communities) == TRUE){
      break;
    }
    oldCommunityStructure <- communities
  }
  return (communities)
}

# Community structure after completing the Phase1 of the algorithm is stored in "communitiesAfterPhase1"
communitiesAfterPhase1 <- Phase1(graph, communities, attributeData, alpha)
#print(communitiesAfterPhase1)

temp <- communitiesAfterPhase1

# Simplifying the original graph using the Phase1 result and then again applying the Phase1 on the reduced
# graph. This is Phase2 of the algorithm.
for(i in 1:max_iterations){
  newGraph <- contract(graph, communitiesAfterPhase1)
  newGraph <- simplify(newGraph, remove.multiple = TRUE, remove.loops = TRUE)
  communitiesAfterPhase1 <- Phase1(newGraph, communitiesAfterPhase1, attributeData, alpha)
  if(all.equal(communitiesAfterPhase1, temp) == TRUE){
    break;
  }
  temp <- communitiesAfterPhase1
}

#print(communitiesAfterPhase1)


# After completion of Phase2, we get the final community strucutre. 
# Writing the communities to the "Communities.txt" file.

f <- file("communities.txt",open="w")

for(i in 1:length(unique(communitiesAfterPhase1))){
  finalCommunities <- vector("numeric")
  for(j in 1:vcount(graph)){
    if(communitiesAfterPhase1[j]==unique(communitiesAfterPhase1)[i]){
      finalCommunities <- append(finalCommunities, j, after = length(finalCommunities))
    }
  }
  #writeLines(as.character(finalCommunities),f, sep = ",")
  write(paste(finalCommunities, collapse = ","), file = f)
}

close(f)
