import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

# use ggplot style sheet
style.use('ggplot')

reg_dtr = np.load("reg_dtr.npy", allow_pickle=True)
reg_eps = np.load("reg_eps.npy", allow_pickle=True)
#reg_lsvi= np.load("reg_lsvi.npy", allow_pickle=True)
a = reg_eps.item().get(0)[0]


T = len(reg_eps.item().get(0)[0])
n = len(reg_eps.item().get(0))
beta_bench = 8 * (np.log(4 * T)) ** (1/2)
sigma = 1


# different q
for q in [5]:
    for delta in [0.1, 0.2, 0.3, 0.5, 1, 5]:
        dtr = 0
        for i in range(n):
            dtr += reg_dtr.item().get((q, delta))[i] / n
        plt.plot(range(T), dtr, label='DTRB'+ '(' + str(q) + ',' + str(delta) + ')')
plt.xlabel('T')
plt.ylabel('Regret')
plt.legend()
plt.savefig("different_delta.pdf")

