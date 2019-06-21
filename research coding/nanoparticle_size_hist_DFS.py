import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import sys
sys.setrecursionlimit(10000000)

## functoin(convert rgb to grayscale)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

## load image as grayscale and convert data to array
img = mpimg.imread('particle1.png')     
img = img/255
gray = rgb2gray(img) 
np_im = np.array(gray)
data = np.reshape(np_im, (-1, 1))

## setup Kmeans and get island array
kmeans = KMeans(n_clusters = 2, init = 'random', max_iter = 100, 
                n_init = 10, random_state = 0).fit(data)
y = kmeans.labels_
y2 = np.reshape(y, (960, 1280))
#y2 = [[0,1,0,1],
#      [0,1,0,1],
#      [0,1,0,0]]
row = len(y2)
col = len(y2[0])

class Graph:   
    def __init__(self, row, col, y2): 
        self.ROW = row
        self.COL = col
        self.grid = y2 
    
    def numIslands(self):
        if self.ROW == 0:
            return 0
        a, b = self.ROW, self.COL
        visited = [[0 for x in range(0, b)] for y in range(0, a)]
        
        count=0
        for i in range(0,a):
            for j in range(0,b):
                if visited[i][j]==0 and self.grid[i][j]==1:
                    self.dfs(y2,i,j,visited)
                    count += 1
        return count    

    def dfs(self, y2, i, j, visited):
        if i < 0 or i >= self.ROW:
            return
        if j < 0 or j >= self.COL:
            return
        if self.grid[i][j]==0 or visited[i][j]==1:
            return
        visited[i][j] = 1 
        self.dfs(y2, i-1, j, visited)
        self.dfs(y2, i+1, j, visited)
        self.dfs(y2, i, j-1, visited)
        #self.dfs(y2, i, j+1, visited) 
    
result = Graph(row, col, y2)
print (result.numIslands())  
