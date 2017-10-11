import numpy as np
import matplotlib.pyplot as plt

def load(Dir, id):
	with open(Dir + 'output' + id) as f:
		content = f.readlines()
		
	content = [x.strip() for x in content]
	x = []
	y = []
	y0 = []
	i = 0
	while i < len(content):
		x.append(int(content[i]))
		s1 = 0
		s2 = 0
		for j in range(5):
			i += 1
			t1 = float(content[i])
			s1 += t1
			i += 1
			t2 = float(content[i])
			s2 += t2
		y.append(s2 / 5)	
		y0.append(s1 / 5)
		i += 1

	return x, y, y0

Dir = 'outputFiles/'

for i in range(1,4):
	x, y, y0 = load(Dir, str(i))
	#fit = np.polyfit(x, y, 7)
	#fit_fn = np.poly1d(fit)

	plt.plot(x,y,'*',x,y,'-', label = str(i))

	plt.plot(x,y0,'--')
plt.legend(loc='best')
plt.show()
