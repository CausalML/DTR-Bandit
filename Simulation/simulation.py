from multiprocessing import Queue, Process
import numpy as np
import math
import scipy, scipy.stats
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#problem parameters
mu_as = np.array([4,1,1,2])
mu_bs = np.array([1,2,4,1])
theta1 = np.array([1,1])
theta2 = np.array([3,0])
T = 10000
d = 2
beta_bench = 4 * (np.log(4 * T)) ** (1/2)
q_list = [3, 5, 10]
delta_list = [0.1, 0.2, 0.3, 0.5, 1, 5]
eps_list = [0, 0.1]
beta_list = [beta_bench]

t_steps = np.arange(20) * T / 20


def main():

    num_procs = 20
    num_path = 40
    #how many times we repeat our algorithm

    jobs = []
    for _ in range(num_path):
        x1 = np.random.rand(T) 
        u1 = np.random.rand(T) 
        u2 = np.random.rand(T) 
        m1 = (x1 > u1) * 0 + (x1 <= u1) * 1
        m2 = (x1 > u2) * 2 + (x1 <= u2) * 3
        x21 = scipy.stats.beta.rvs(mu_as[m1], mu_bs[m1])
        x22 = scipy.stats.beta.rvs(mu_as[m2], mu_bs[m2])
        y11 = theta1[0] * x21 + theta1[1] + np.random.randn(T)
        y12 = theta2[0] * x21 + theta2[1] + np.random.randn(T)
        y21 = theta1[0] * x22 + theta1[1] + np.random.randn(T)
        y22 = theta2[0] * x22 + theta2[1] + np.random.randn(T)
        
        jobs.append({"x1": x1,
                     "x2": {1: x21, 2: x22},
                     "y": {(1,1): y11, (1,2): y12, (2,1): y21, (2,2): y22}
                     })

    job_queue = Queue()
    results_queue = Queue()
    for job in jobs:
        job_queue.put(job)
    num_jobs = len(jobs)

    for _ in range(num_procs):
        job_queue.put("STOP")

    procs = []
    for _ in range(num_procs):
        p = Process(target=worker_function, args=(job_queue, results_queue))
        p.start()
        procs.append(p)

    results_list = []
    for _ in range(num_jobs):
        next_result = results_queue.get()
        results_list.append(next_result)

    for p in procs:
        p.join()

    opt = []
    reg_dtr = {}
    reg_eps = {}
    reg_lsvi = {}
    for result in results_list:
        opt.append([result["y_opt"]])
        for q in q_list:
            for delta in delta_list:
                print("dtr", q, delta, result["regret_dtr"][(q, delta)][-1])
                if (q, delta) not in reg_dtr:
                    reg_dtr[(q, delta)] = [result["regret_dtr"][(q, delta)]]
                else:
                    reg_dtr[(q, delta)].append(result["regret_dtr"][(q, delta)])
        for eps in eps_list:
            print("eps", eps, result["regret_eps"][eps][-1])
            if eps not in reg_eps:
                reg_eps[eps] = [result["regret_eps"][eps]]
            else:
                reg_eps[eps].append(result["regret_eps"][eps])
        for beta in beta_list:
            print("lsvi", beta, result["regret_lsvi"][beta][-1])
            if beta not in reg_lsvi:
                reg_lsvi[beta] = [result["regret_lsvi"][beta]]
            else:
                reg_lsvi[beta].append(result["regret_lsvi"][beta])
        
    np.save("reg_dtr.npy", reg_dtr)
    np.save("reg_eps.npy", reg_eps)
    np.save("reg_lsvi.npy", reg_lsvi)
    np.save("opt.npy", opt)

def worker_function(job_queue, results_queue):
    for job in iter(job_queue.get, "STOP"):
        next_result = run_job(job)
        results_queue.put(next_result)



def run_job(job):
    print("a job running...")

    x1 = job["x1"]
    x2 = job["x2"]
    y = job["y"]

    """
    optimal arm
    """
    # need to recompute the parameters here if we change any parameters!
    def optimal():

        y_opt = []

        for t in range(T):
            if x1[t] > 5/14 and x2[1][t] < 0.5:
                y_opt.append(y[(1,1)][t])
            elif x1[t] > 5/14 and x2[1][t] >= 0.5:
                y_opt.append(y[(1,2)][t])
            elif x1[t] <= 5/14 and x2[2][t] < 0.5:
                y_opt.append(y[(2,1)][t])
            else:
                y_opt.append(y[(2,2)][t])

        return np.array(y_opt)



    """
    DTR Bandit
    """

    # helper functions
    def compute_forced_pulls(q):
        K = 4
        forced_pulls = np.array([-1 for _ in range(T)], dtype=object)  
        for i in range(int(math.log2(T)) + 2):
            for j in range(q):
                if (2 ** i - 1) * K * q + j < T:
                    forced_pulls[(2 ** i - 1) * K * q + j] = (1,1)
                if (2 ** i - 1) * K * q + q + j < T:
                    forced_pulls[(2 ** i - 1) * K * q + q + j] = (1,2)
                if (2 ** i - 1) * K * q + 2 * q + j < T:
                    forced_pulls[(2 ** i - 1) * K * q + 2 * q + j] = (2,1)
                if (2 ** i - 1) * K * q + 3 * q + j < T:
                    forced_pulls[(2 ** i - 1) * K * q + 3 * q + j] = (2,2)  
        return forced_pulls
    

    # DTR bandit
    def DTR_Bandit(q, delta):

        a = {1: np.zeros(T), 2: np.zeros(T)}
        x = {1: x1, 2: np.zeros(T)}
        y_dtr = np.zeros(T)
        beta_hat = {}
        beta_tilde = {}
        for i in [1,2]:
            for j in [1,2]:
                beta_hat[(i,j)] = None
                beta_tilde[(i,j)] = None

        # compute forced pulls
        forced_pulls = compute_forced_pulls(q)

        # run the algorithm
        for t in range(T):

            # choose arms
            if forced_pulls[t] != -1:
                a[1][t], a[2][t] = forced_pulls[t]
            else:
                for m in [1,2]:
                    e_tilde = [
                        beta_tilde[(1, m)].predict(x[m][t].reshape(-1,1)),
                        beta_tilde[(2, m)].predict(x[m][t].reshape(-1,1))
                        ]
                    if abs( e_tilde[0] - e_tilde[1] ) > delta/2:
                        a[m][t] = np.argmax(e_tilde) + 1
                    else:
                        a[m][t] = np.argmax([
                            beta_hat[(1, m)].predict(x[m][t].reshape(-1,1)), 
                            beta_hat[(2, m)].predict(x[m][t].reshape(-1,1))]
                            ) + 1
                    if m == 1:
                        x[2][t] = x2[a[1][t]][t]
            
            # update history
            x[2][t] = x2[a[1][t]][t]
            y_dtr[t] = y[(a[1][t], a[2][t])][t]
            
            # update parameters
            if t >= 2 * d and t < T-1 and forced_pulls[t+1] == -1:
                # update hat parameters
                for i in [1, 2]:
                    beta_hat[(i, 2)] = LinearRegression().fit(
                        x[2][a[2] == i].reshape(-1, 1), 
                        y_dtr[a[2] == i]
                        )
                y_hat = np.maximum(
                    beta_hat[(1, 2)].predict(x[2].reshape(-1, 1)), 
                    beta_hat[(2, 2)].predict(x[2].reshape(-1, 1))
                    )
                for i in [1, 2]:
                    beta_hat[(i, 1)] = LinearRegression().fit(
                        x[1][a[1] == i].reshape(-1, 1), 
                        y_hat[a[1] == i]
                        )
                # if we force pull, update tilde parameters
                if forced_pulls[t] != -1:
                    for i in [1, 2]:
                        beta_tilde[(i, 2)] = LinearRegression().fit(
                            x[2][(forced_pulls != -1) & (a[2] == i)].reshape(-1, 1), 
                            y_dtr[(forced_pulls != -1) & (a[2] == i)]
                        )
                    y_tilde = np.maximum(
                        beta_tilde[(1, 2)].predict(x[2].reshape(-1, 1)),
                        beta_tilde[(2, 2)].predict(x[2].reshape(-1, 1)),
                    )
                    for i in [1, 2]:
                        beta_tilde[(i, 1)] = LinearRegression().fit(
                            x[1][(forced_pulls != -1) & (a[1] == i)].reshape(-1, 1),
                            y_tilde[(forced_pulls != -1) & (a[1] == i)]
                        )

        return y_dtr



    """
    epsilon greedy
    """
    def Eps_Greedy(eps):

        a = {1: np.zeros(T), 2: np.zeros(T)}
        x = {1: x1, 2: np.zeros(T)}
        y_eps = np.zeros(T)
        beta_hat = {}
        for i in [1,2]:
            for j in [1,2]:
                beta_hat[(i,j)] = None

        # run the algorithm
        for t in range(T):
            
            # choose arms
            if t < d:
                a[1][t], a[2][t] = (1, 1)
            elif t < 2 * d:
                a[1][t], a[2][t] = (1, 2)
            elif t < 3 * d:
                a[1][t], a[2][t] = (2, 1)
            elif t < 4 * d:
                a[1][t], a[2][t] = (2, 2)
            else:
                for m in [1, 2]:
                    a[m][t] = np.argmax([
                        beta_hat[(1, m)].predict(x[m][t].reshape(-1,1)),
                        beta_hat[(2, m)].predict(x[m][t].reshape(-1,1))
                    ]) + 1
                    if np.random.rand() < eps:
                        a[m][t] = 3 - a[m][t]
                    if m == 1:
                        x[2][t] = x2[a[1][t]][t]                   

            # update history
            x[2][t] = x2[a[1][t]][t]
            y_eps[t] = y[(a[1][t], a[2][t])][t]

            # update parameters
            if t >= 4 * d - 1:
                # update hat parameters
                for i in [1, 2]:
                    beta_hat[(i, 2)] = LinearRegression().fit(
                        x[2][a[2] == i].reshape(-1, 1), 
                        y_eps[a[2] == i]
                        )
                y_hat = np.maximum(
                    beta_hat[(1, 2)].predict(x[2].reshape(-1, 1)), 
                    beta_hat[(2, 2)].predict(x[2].reshape(-1, 1))
                    )
                for i in [1, 2]:
                    beta_hat[(i, 1)] = LinearRegression().fit(
                        x[1][a[1] == i].reshape(-1, 1), 
                        y_hat[a[1] == i]
                        )

        return y_eps



    """
    LSVI-UCB
    """
    # helper functions
    def phi(x, a):
        if a == 1:
            return np.array([1, x, 0, 0]).reshape(2 * d, 1)
        return np.array([0, 0, 1, x]).reshape(2 * d, 1)
    
    def compute_q(wh, xht, aht, beta, Lambdah):
        temp = wh.T @ phi(xht, aht) + beta * (phi(xht, aht).T @ np.linalg.inv(Lambdah) @ phi(xht, aht)) ** (1/2)
        return temp[0][0]
    
    def compute_w(Lambda, temp):
        return np.linalg.inv(Lambda) @ temp

    def compute_temp1(x1, a1, x2, k, w2, beta, Lambda2):
        temp1 = np.zeros([2 * d, 1])
        for t in range(k): 
            max_q = max(
                compute_q(w2, x2[t], 1, beta, Lambda2),
                compute_q(w2, x2[t], 2, beta, Lambda2)
            )
            temp1 += phi(x1[t], a1[t]) * max_q
        return temp1

    # lsvi-ucb
    def LSVI_UCB(lamb, beta):

        a = {1: np.zeros(T), 2: np.zeros(T)}
        x = {1: x1, 2: np.zeros(T)}
        y_lsvi = np.zeros(T)

        Lambda = {1: lamb * np.identity(2 * d), 2: lamb * np.identity(2 * d)}
        w = {1: None, 2: None}
        temp = {1: None, 2: np.zeros([2 * d, 1])}

        # run the algorithm
        for t in range(T):

            if t in t_steps:
                print(t)

            # compute w1 and w2
            w[2] = compute_w(Lambda[2], temp[2])
            temp[1] = compute_temp1(x[1], a[1], x[2], t, w[2], beta, Lambda[2])
            w[1] = compute_w(Lambda[1], temp[1])

            # take actions and update the history
            for h in [1, 2]:
                a[h][t] = np.argmax([
                    compute_q(w[h], x[h][t], 1, beta, Lambda[h]),
                    compute_q(w[h], x[h][t], 2, beta, Lambda[h])
                ]) + 1
                if h == 1:
                    x[2][t] = x2[a[1][t]][t]
            y_lsvi[t] = y[(a[1][t], a[2][t])][t]

            # update parameters
            for h in [1, 2]:
                Lambda[h] += phi(x[h][t], a[h][t]) @ phi(x[h][t], a[h][t]).T
            temp[2] += phi(x[2][t], a[2][t]) * y_lsvi[t]

        return y_lsvi


    """
    Compute rewards and regrets
    """
    y_opt = optimal()
    regret_dtr = {}
    regret_eps = {}
    regret_lsvi = {}

    for q in q_list:
        for delta in delta_list:
            cur = y_opt - DTR_Bandit(q, delta)
            regret_dtr[(q, delta)] = cur.cumsum()
            print("dtr", q, delta, regret_dtr[(q, delta)][-1])

    for eps in eps_list:
        cur = y_opt - Eps_Greedy(eps)
        regret_eps[eps] = cur.cumsum()
        print("eps", eps, regret_eps[eps][-1])

    for beta in beta_list:
        cur = y_opt - LSVI_UCB(1, beta)
        regret_lsvi[beta] = cur.cumsum()
        for t in t_steps:
            print("t", t, "regret:", regret_lsvi[beta][int(t)])
        print("lsvi", beta, regret_lsvi[beta][-1])
        print("slope", (np.log(regret_lsvi[beta][-1]) - np.log(regret_lsvi[beta][int(T/2)-1]))/ (np.log(T) - np.log(T/2)))

    result = {
        "y_opt": y_opt, 
        "regret_dtr": regret_dtr, 
        "regret_eps": regret_eps, 
        "regret_lsvi": regret_lsvi
        }
    return result


if __name__ == "__main__":
    main()