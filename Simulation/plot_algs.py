import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

# use ggplot style sheet
style.use('ggplot')

reg_dtr = np.load("reg_dtr.npy", allow_pickle=True)
reg_eps = np.load("reg_eps.npy", allow_pickle=True)
reg_lsvi= np.load("reg_lsvi.npy", allow_pickle=True, encoding='latin1')
a = reg_eps.item().get(0)[0]


T = len(reg_eps.item().get(0)[0])
n = len(reg_eps.item().get(0))
beta_bench = 4 * (np.log(4 * T)) ** (1/2)
sigma = 1


# different algorithms
for q in [5]:
    for delta in [0.5]:
        dtr = 0
        for i in range(n):
            dtr += reg_dtr.item().get((q, delta))[i] / n
        plt.plot(range(T), dtr, label='DTRB'+ '(' + str(q) + ',' + str(delta) + ')')
for beta in [beta_bench]:
    lsvi = 0
    for i in range(n):
        lsvi += reg_lsvi.item().get(beta)[i] / n
    plt.plot(range(T), lsvi, label='LSVI-UCB')
for e in [0.1]:
    eps = 0
    for i in range(n):
        eps += reg_eps.item().get(e)[i] / n
    plt.plot(range(T), eps, label = '0.1-Greedy')
for e in [0]:
    eps = 0
    for i in range(n):
        eps += reg_eps.item().get(e)[i] / n
    plt.plot(range(T), eps, label = 'Greedy' )

plt.xlabel('T')
plt.ylabel('Regret')
plt.legend()
plt.savefig("different_alg.pdf")
