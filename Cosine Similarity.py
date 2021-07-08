import numpy as np

def cosine_similarity(a,b):
  num = np.dot(a,b)
  den = np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2))
  return num/(den+1e-15)

incep = np.array([0,1,4,10]) # standard

about_time = np.array([10,0,0,0])
dark_knight = np.array([1,1,7,3])
martian = np.array([2,0,0,10])
thenun = np.array([0,10,1,0])

movies = [about_time,dark_knight,martian,thenun]
names = ['about_time','dark_knight','martian','thenun']

scores = []
for mvec,name in zip(movies,names):
  sim = cosine_similarity(mvec,incep)
  scores.append(sim)

idx = np.argsort(scores)
recom = np.array(names)[idx]
print(np.flip(recom))