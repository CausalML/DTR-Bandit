import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

# use ggplot style sheet
style.use('ggplot')

reg_greedy = np.load("reg_greedy.npy")
reg_DTR = np.load("reg_DTR.npy")
reg_4= np.load("reg_4.npy")
reg_42= np.load("reg_42.npy")

#T = np.shape(reg_DTR)[0]
T = 10000

plt.plot(range(T), reg_greedy[0:T], label='Greedy')
plt.plot(range(T), reg_DTR[0:T], label='DTRBandit')
plt.plot(range(T), reg_4[0:T], label='Static')
plt.plot(range(T), reg_42[0:T], label='Recourse')
#plt.show()

plt.xlabel('T')
plt.ylabel('Regret')
#plt.title('q=20 h=0.5')
plt.legend()

plt.savefig("q_20_h_05_10000.pdf")

