
# coding: utf-8

# In[62]:


# import data:

from numpy import corrcoef, mat, shape, nonzero, logical_and
import numpy.linalg as la

def loadExtData():
    # mat A
    return [[4,4,0,2,2],
            [4,0,0,3,3],
            [4,0,0,1,1],
            [1,1,1,2,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0]]

# similarity calculation
def excludSim(inA,inB):
    '''
    use l2 norm to calculate similarity
    normalize -> (0,1]
    '''
    dis = 1.0/(1.0+la.norm(inA-inB))
    return dis

def pearsSim(inA,inB):
    '''
    user pearson coefficient to calculate similarity
    normalize -> (0,1]
    '''
    if(len(inA) < 3):return 1.0
    dis = 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]
    return dis

def cosSim(inA,inB):
    '''
    cosine similarity
    normalize -> (0,1]
    '''
    tmp = float(inA.T*inB)
    dis = 0.5+0.5*tmp/(la.norm(inA)*la.norm(inB))
    return dis

def standEst(dataMat, user, simMean, item):
    '''
    calculate user's score for item
    simMean:similarity calculation method 
    '''
    if(dataMat[user, item] != 0): return dataMat[user, item]
    n = shape(dataMat)[1]  # number of items
    simTotal = 0.0
    ratSimTotal = 0.0
    for i in range(n):
        userRating = dataMat[user, i]
        if(userRating == 0 or i == item): continue
        # search for users that ever rate two items
        overLap = nonzero(logical_and(dataMat[:,i].A>0, dataMat[:,item].A>0))[0]
        if(len(overLap) == 0): similarity = 0
        else: similarity = simMean(dataMat[overLap, i], dataMat[overLap, item])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if(simTotal == 0):return 0
    else: return ratSimTotal / simTotal # return user's score for item

def recommend(dataMat, user, N=3, simMean=cosSim, estTestMean=standEst):
    '''
    recommend n items to user based on the specific dataMat
    simMean:similarity calculation method 
    estTestMean:cal user score of item
    '''
    unRatedItem = nonzero(dataMat[user,:].A == 0)[1] # .A change matrix to array
    if(len(unRatedItem) == 0):print('There is nothing to recommend')
    retScores = [] # scores of unRatedItems
    for item in unRatedItem:
        itemScore=estTestMean(dataMat, user, simMean, item) # predicton of user for item
        retScores.append((item, itemScore))
    return sorted(retScores, key=lambda j:j[1], reverse=True)[:N] # return the top N high rated items


# In[63]:


myData = mat(loadExtData())
ans = recommend(myData, 2)
print(ans)


# In[64]:


# create A=U,S,V
dataMat = loadExtData()
U,S,V = la.svd(dataMat)
print(S,"\n")

# restore A
S_3 = mat([[S[0],0,0],[0,S[1],0],[0,0,S[2]]])
restoreData = U[:,:3]*S_3*V[:3,:]
print(restoreData)


# In[139]:


# recommender items based on svd
def loadExtData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def svdEst(dataMat, user, simMeas, item, k):
    if(dataMat[user, item] != 0): return dataMat[user, item]
    n = shape(dataMat)[1]   # n,11
    simTotal = 0.0;ratSimTotal = 0.0
    U, S, V = la.svd(dataMat)
    S3 = mat(eye(k) * S[:k]) # create a diagonal matrix to save 3 eigenvalues in S
    xformedItems = dataMat.T * U[:, :k] * S3.I # reduce dimensions of items
    for j in range(n):
        userRating = dataMat[user, j]
        if(userRating == 0 or j == item): continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if(simTotal == 0):  return 0
    else:   return ratSimTotal / simTotal 

def svdRecommend(dataMat, user, N=3, simMean=cosSim, estTestMean=svdEst, k=3):
    '''
    recommend n items to user based on the specific dataMat
    simMean:similarity calculation method 
    estTestMean:cal user score of item
    k:k controls the number of eigenvalues
    '''
    unRatedItem = nonzero(dataMat[user,:].A == 0)[1] # .A change matrix to array
    if(len(unRatedItem) == 0):print('There is nothing to recommend')
    retScores = [] # scores of unRatedItems
    for item in unRatedItem:
        itemScore=estTestMean(dataMat, user, simMean, item, k=k) # predicton of user for item
        retScores.append((item, itemScore))
    return sorted(retScores, key=lambda j:j[1], reverse=True)[:N] # return the top N high rated items



# In[140]:


myData = mat(loadExtData2())
U, S, V = la.svd(myData)
S *= S
threshold = sum(S) * 0.9
k = 0

for i in range(S.shape[0]+1):
    if(sum(S[:i]) >= threshold):
        k = i
        break


# In[141]:


svdItems = svdRecommend(myData, user=3, estTestMean=svdEst, k=k)
print(svdItems)


