import numpy as np
import matplotlib.pyplot as plt

with open('output1') as f:
	content = f.readlines()

content = [x.strip() for x in content]

x = []
y = []
y0 = []
i = 0
while i < len(content):
	x.append(int(content[i]))
	s = 0
	for j in range(5):
		i += 1
		t1 = float(content[i])
		i += 1
		t2 = float(content[i])
		s += t1 / t2
	y.append(s / 5)	
	y0.append(1.0)
	i += 1

print x
print y

fit = np.polyfit(x, y, 7)
fit_fn = np.poly1d(fit)

plt.plot(x,y,'b*',x,fit_fn(x),'--r',x,y0,'--y')
plt.show()
