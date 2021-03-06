import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import math

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
#      [1,1,0,0]]
## BFS method
class Solution:
    def numIslands(self, y2):
        count = 0  # island number
        m, n = len(y2), len(y2[0])
        # start with True, convert to False 
        result = []
        mark = self._mark(m, n)
        for i in range(m):
            for j in range(n):
                num = 0
                if y2[i][j] == 1:
                    if mark[i][j]:  # check mark
                        count += 1
                        num += 1
                        mark[i][j] = False
                        a = self._expand(i, j, y2, mark, num)
                        result.append(a)
        return result
    
    @staticmethod
    def _mark(m, n):
        mark = []
        for i in range(m):
            mark.append([True] * n)
        return mark
    
    @staticmethod
    def _expand(i, j, y2, mark, num):
        # expand to right and bottom (breath-first search)
        t = [(i, j)]
        # check boundary
        m, n = len(y2), len(y2[0])
        add = num
        while len(t) != 0:
            i, j = t.pop()
            # left
            if i > 0:
                if mark[i - 1][j]:
                    if y2[i - 1][j] == 1:
                        add += 1
                        mark[i - 1][j] = False
                        t.append((i - 1, j))
            # right
            if i < m - 1:
                if mark[i + 1][j]:
                    if y2[i + 1][j] == 1:
                        add += 1
                        mark[i + 1][j] = False
                        t.append((i + 1, j))
            # upper
            if j > 0:
                if mark[i][j - 1]:
                    if y2[i][j - 1] == 1:
                        add += 1
                        mark[i][j - 1] = False
                        t.append((i, j - 1))
            # bottom
            if j < n - 1:
                if mark[i][j + 1]:
                    if y2[i][j + 1] == 1:
                        add += 1
                        mark[i][j + 1] = False
                        t.append((i, j + 1))
        return add
    
if __name__ == '__main__':
    s = Solution()
    particle_pixel = s.numIslands(y2)

## convert pixel to area and plot histgram
particle_pixel = [math.sqrt(float(x)) for x in particle_pixel if x >= 250]
graph_width = 6038.46 #unit nm
pixel_diameter = float(graph_width/len(y2[0]))
particle_diameter = [x*pixel_diameter for x in particle_pixel]
plt.hist(particle_diameter, facecolor='green', alpha=0.5)
plt.xlabel('particle diameter(nm)')
plt.ylabel('partile number')
plt.title('particle number vs paticle size')







