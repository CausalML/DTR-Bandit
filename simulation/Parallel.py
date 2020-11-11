from multiprocessing import Queue, Process
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#problem parameters
d = 1
b = 5
B = 2
beta11 = b * (B-1) * np.ones(d)
beta21 = 0 * np.ones(d)
beta12 = 1 * np.ones(d)
beta22 = b * np.ones(d)
B1 = np.diag(np.ones(d))
B2 = np.diag(B*np.ones(d))
sigma = 0.1
epsilon_max = 1


T = 50000


def main():
    # problem parameters

    num_procs = 32

    num_path = 192
    #how many times we repeat our algorithm

    jobs = []
    for _ in range(num_path):
        jobs.append({"X1": np.random.randn(T, d),
                     "eta1": np.random.normal(0, sigma, size=T),
                     "epsilon": np.random.uniform(-epsilon_max, epsilon_max, size=(T, d)),
                     "eta2": np.random.normal(0, sigma, size=T)})

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

    reg_DTR = 0
    reg_greedy = 0
    reg_4 = 0
    reg_42 = 0
    for result in results_list:
        #print(result["regret_42"].mean())
        reg_DTR = reg_DTR + result["regret_DTR"].cumsum()/num_path
        reg_greedy = reg_greedy + result["regret_greedy"].cumsum()/num_path
        reg_4 = reg_4 + result["regret_4"].cumsum()/ num_path
        reg_42 = reg_42 + result["regret_42"].cumsum()/num_path

    #plt.plot(range(T), reg_greedy, range(T), reg_DTR, range(T), reg_4, range(T), reg_42)
    #plt.show()
    np.save("reg_DTR.npy", reg_DTR)
    np.save("reg_greedy.npy", reg_greedy)
    np.save("reg_4.npy", reg_4)
    np.save("reg_42.npy", reg_42)
    print(reg_DTR[T-1], reg_greedy[T-1], reg_4[T-1], reg_42[T-1])

def worker_function(job_queue, results_queue):
    for job in iter(job_queue.get, "STOP"):
        next_result = run_job(job)
        results_queue.put(next_result)


def run_job(job):
    print("a job running...")

    X1 = job["X1"]
    eta1 = job["eta1"]
    epsilon = job["epsilon"]
    eta2 = job["eta2"]

    # oprimal arm
    arm11 = np.zeros(T, dtype=bool)
    arm21 = np.zeros(T, dtype=bool)
    arm12 = np.zeros(T, dtype=bool)
    arm22 = np.zeros(T, dtype=bool)
    r1_opt = np.zeros(T)
    r2_opt = np.zeros(T)
    X2_opt = np.zeros((T, d))

    for i in range(T):

        if (X1[i].sum() > 0) & (X1[i].sum() < epsilon_max * 2 / (B + 1)):
            arm11[i] = True
            r1_opt[i] = np.dot(beta11, X1[i].T) + eta1[i]
            X2_opt[i] = np.dot(B1, X1[i].T) + epsilon[i]
        else:
            arm21[i] = True
            r1_opt[i] = np.dot(beta21, X1[i].T) + eta1[i]
            X2_opt[i] = np.dot(B2, X1[i].T) + epsilon[i]

        if np.dot(beta12, X2_opt[i]) > np.dot(beta22, X2_opt[i]):
            arm12[i] = True
            r2_opt[i] = np.dot(beta12, X2_opt[i].T) + eta2[i]
        else:
            arm22[i] = True
            r2_opt[i] = np.dot(beta22, X2_opt[i].T) + eta2[i]



            # DTR bandit
    # algorithm parameters
    q = 20
    h = 0.5
    K = 2

    arm11 = np.zeros(T, dtype=bool)  # indicate if we pull arm 1 in first period at time t
    arm21 = np.zeros(T, dtype=bool)
    arm12 = np.zeros(T, dtype=bool)
    arm22 = np.zeros(T, dtype=bool)
    r1_DTR = np.zeros(T)
    r2_DTR = np.zeros(T)

    X2_DTR = np.zeros((T, d))

    # forced pull index
    forced_index_1 = np.zeros(T, dtype=bool)
    forced_index_1[0:d] = True
    forced_index_2 = np.zeros(T, dtype=bool)
    forced_index_2[d:(2 * d)] = True

    for i in range(15):
        for j in range(q):
            if (2 ** i - 1) * K * q + j > 2 * d - 1 and (2 ** i - 1) * K * q + j < T:
                forced_index_1[(2 ** i - 1) * K * q + j] = True
            if (2 ** i - 1) * K * q + q + j > 2 * d - 1 and (2 ** i - 1) * K * q + q + j < T:
                forced_index_2[(2 ** i - 1) * K * q + q + j] = True

    # running our algorithm
    for i in range(T):

        if forced_index_1[i]:
            arm11[i] = True
            arm12[i] = True
            r1_DTR[i] = np.dot(beta11, X1[i].T) + eta1[i]
            X2_DTR[i] = np.dot(B1, X1[i].T) + epsilon[i]
            r2_DTR[i] = np.dot(beta12, X2_DTR[i].T) + eta2[i]

        elif forced_index_2[i]:
            arm21[i] = True
            arm22[i] = True
            r1_DTR[i] = np.dot(beta21, X1[i].T) + eta1[i]
            X2_DTR[i] = np.dot(B2, X1[i].T) + epsilon[i]
            r2_DTR[i] = np.dot(beta22, X2_DTR[i].T) + eta2[i]

        else:

            # estimate tilde parameters
            beta_tilde_11 = LinearRegression().fit(X1[forced_index_1 & arm11, :], r1_DTR[forced_index_1 & arm11])
            beta_tilde_21 = LinearRegression().fit(X1[forced_index_2 & arm21, :], r1_DTR[forced_index_2 & arm21])
            beta_tilde_12 = LinearRegression().fit(X2_DTR[forced_index_1 & arm12, :], r2_DTR[forced_index_1 & arm12])
            beta_tilde_22 = LinearRegression().fit(X2_DTR[forced_index_2 & arm22, :], r2_DTR[forced_index_2 & arm22])
            B_tilde_1 = LinearRegression().fit(X1[forced_index_1 & arm11, :], X2_DTR[forced_index_1 & arm11, :])
            B_tilde_2 = LinearRegression().fit(X1[forced_index_2 & arm21, :], X2_DTR[forced_index_2 & arm21, :])

            # estimate hat estimators(we may need to change it to not-forced)
            beta_hat_11 = LinearRegression().fit(X1[arm11, :], r1_DTR[arm11])
            beta_hat_21 = LinearRegression().fit(X1[arm21, :], r1_DTR[arm21])
            beta_hat_12 = LinearRegression().fit(X2_DTR[arm12, :], r2_DTR[arm12])
            beta_hat_22 = LinearRegression().fit(X2_DTR[arm22, :], r2_DTR[arm22])
            B_hat_1 = LinearRegression().fit(X1[arm11, :], X2_DTR[arm11, :])
            B_hat_2 = LinearRegression().fit(X1[arm21, :], X2_DTR[arm21, :])

            # first period
            # estimate Q Tilde 1
            temp_1 = B_tilde_1.predict(X1[i].reshape(1, -1)) - B_tilde_1.predict(X1[0:i][forced_index_1[0:i]]) + X2_DTR[0:i][forced_index_1[0:i]]
            #temp_1 = B_tilde_1.predict(X1[i].reshape(1, -1)) - B_tilde_1.predict(X1[arm11, :]) + X2_DTR[arm11, :]
            temp_2 = B_tilde_2.predict(X1[i].reshape(1, -1)) - B_tilde_2.predict(X1[0:i][forced_index_2[0:i]]) + X2_DTR[0:i][forced_index_2[0:i]]
            #temp_2 = B_tilde_2.predict(X1[i].reshape(1, -1)) - B_tilde_2.predict(X1[arm21, :]) + X2_DTR[arm21, :]
            Q_tilde_11 = beta_tilde_11.predict(X1[i].reshape(1, -1)) + np.vstack(
                (beta_tilde_12.predict(temp_1), beta_tilde_22.predict(temp_1))).max(axis=0).mean()
            Q_tilde_21 = beta_tilde_21.predict(X1[i].reshape(1, -1)) + np.vstack(
                (beta_tilde_12.predict(temp_2), beta_tilde_22.predict(temp_2))).max(axis=0).mean()

            if Q_tilde_11 > Q_tilde_21 + h / 2:
                arm11[i] = True
                r1_DTR[i] = np.dot(beta11, X1[i].T) + eta1[i]
                X2_DTR[i] = np.dot(B1, X1[i].T) + epsilon[i]
            elif Q_tilde_21 > Q_tilde_11 + h / 2:
                arm21[i] = True
                r1_DTR[i] = np.dot(beta21, X1[i].T) + eta1[i]
                X2_DTR[i] = np.dot(B2, X1[i].T) + epsilon[i]
            else:
                # estimate Q hat 1
                temp_1 = B_hat_1.predict(X1[i].reshape(1, -1)) - B_hat_1.predict(X1[arm11, :]) + X2_DTR[arm11, :]
                temp_2 = B_hat_2.predict(X1[i].reshape(1, -1)) - B_hat_2.predict(X1[arm21, :]) + X2_DTR[arm21, :]
                Q_hat_11 = beta_hat_11.predict(X1[i].reshape(1, -1)) + np.vstack(
                    (beta_hat_12.predict(temp_1), beta_hat_22.predict(temp_1))).max(axis=0).mean()
                Q_hat_21 = beta_hat_21.predict(X1[i].reshape(1, -1)) + np.vstack(
                    (beta_hat_12.predict(temp_2), beta_hat_22.predict(temp_2))).max(axis=0).mean()

                if Q_hat_11 > Q_hat_21:
                    arm11[i] = True
                    r1_DTR[i] = np.dot(beta11, X1[i].T) + eta1[i]
                    X2_DTR[i] = np.dot(B1, X1[i].T) + epsilon[i]
                else:
                    arm21[i] = True
                    r1_DTR[i] = np.dot(beta21, X1[i].T) + eta1[i]
                    X2_DTR[i] = np.dot(B2, X1[i].T) + epsilon[i]

            # second period (two possible treatments)
            if beta_tilde_12.predict(X2_DTR[i].reshape(1, -1)) > beta_tilde_22.predict(X2_DTR[i].reshape(1, -1)) + h / 2:
                arm12[i] = True
                r2_DTR[i] = np.dot(beta12, X2_DTR[i].T) + eta2[i]
            elif beta_tilde_22.predict(X2_DTR[i].reshape(1, -1)) > beta_tilde_12.predict(
                X2_DTR[i].reshape(1, -1)) + h / 2:
                arm22[i] = True
                r2_DTR[i] = np.dot(beta22, X2_DTR[i].T) + eta2[i]
            elif beta_hat_12.predict(X2_DTR[i].reshape(1, -1)) > beta_hat_22.predict(X2_DTR[i].reshape(1, -1)):
                arm12[i] = True
                r2_DTR[i] = np.dot(beta12, X2_DTR[i].T) + eta2[i]
            else:
                arm22[i] = True
                r2_DTR[i] = np.dot(beta22, X2_DTR[i].T) + eta2[i]







    # greedy

    arm11 = np.zeros(T, dtype=bool)  # indicate if we pull arm 1 in first period at time t
    arm21 = np.zeros(T, dtype=bool)
    arm12 = np.zeros(T, dtype=bool)
    arm22 = np.zeros(T, dtype=bool)
    r1_greedy = np.zeros(T)
    r2_greedy = np.zeros(T)

    X2_greedy = np.zeros((T, d))

    # running our algorithm
    for i in range(T):

        if i < d:
            arm11[i] = True
            arm12[i] = True
            r1_greedy[i] = np.dot(beta11, X1[i].T) + eta1[i]
            X2_greedy[i] = np.dot(B1, X1[i].T) + epsilon[i]
            r2_greedy[i] = np.dot(beta12, X2_greedy[i].T) + eta2[i]

        elif i < 2 * d:
            arm21[i] = True
            arm22[i] = True
            r1_greedy[i] = np.dot(beta21, X1[i].T) + eta1[i]
            X2_greedy[i] = np.dot(B2, X1[i].T) + epsilon[i]
            r2_greedy[i] = np.dot(beta22, X2_greedy[i].T) + eta2[i]

        else:

            # estimate parameters
            beta_hat_11 = LinearRegression().fit(X1[arm11, :], r1_greedy[arm11])
            beta_hat_21 = LinearRegression().fit(X1[arm21, :], r1_greedy[arm21])
            beta_hat_12 = LinearRegression().fit(X2_greedy[arm12, :], r2_greedy[arm12])
            beta_hat_22 = LinearRegression().fit(X2_greedy[arm22, :], r2_greedy[arm22])
            B_hat_1 = LinearRegression().fit(X1[arm11, :], X2_greedy[arm11, :])
            B_hat_2 = LinearRegression().fit(X1[arm21, :], X2_greedy[arm21, :])

            # first period
            # estimate Q Tilde 1
            temp_1 = B_hat_1.predict(X1[i].reshape(1, -1)) - B_hat_1.predict(X1[arm11, :]) + X2_greedy[arm11, :]
            temp_2 = B_hat_2.predict(X1[i].reshape(1, -1)) - B_hat_2.predict(X1[arm21, :]) + X2_greedy[arm21, :]
            Q_hat_11 = beta_hat_11.predict(X1[i].reshape(1, -1)) + np.vstack(
                (beta_hat_12.predict(temp_1), beta_hat_22.predict(temp_1))).max(axis=0).mean()
            Q_hat_21 = beta_hat_21.predict(X1[i].reshape(1, -1)) + np.vstack(
                (beta_hat_12.predict(temp_2), beta_hat_22.predict(temp_2))).max(axis=0).mean()

            if Q_hat_11 > Q_hat_21:
                arm11[i] = True
                r1_greedy[i] = np.dot(beta11, X1[i].T) + eta1[i]
                X2_greedy[i] = np.dot(B1, X1[i].T) + epsilon[i]
            else:
                arm21[i] = True
                r1_greedy[i] = np.dot(beta21, X1[i].T) + eta1[i]
                X2_greedy[i] = np.dot(B2, X1[i].T) + epsilon[i]

            # second period (two possible treatments)
            if beta_hat_12.predict(X2_greedy[i].reshape(1, -1)) > beta_hat_22.predict(X2_greedy[i].reshape(1, -1)):
                arm12[i] = True
                r2_greedy[i] = np.dot(beta12, X2_greedy[i].T) + eta2[i]
            else:
                arm22[i] = True
                r2_greedy[i] = np.dot(beta22, X2_greedy[i].T) + eta2[i]





    # 4 armed bandit
    # algorithm parameters
    #q = 1
    #h = 5
    K = 4

    arm11 = np.zeros(T, dtype=bool)  # indicate if we pull arm 1 in first period at time t
    arm21 = np.zeros(T, dtype=bool)
    arm12 = np.zeros(T, dtype=bool)
    arm22 = np.zeros(T, dtype=bool)
    r1_4 = np.zeros(T)
    r2_4 = np.zeros(T)

    X2_4 = np.zeros((T, d))

    # forced pull index
    forced_index_11 = np.zeros(T, dtype=bool)
    forced_index_11[0:d] = True
    forced_index_12 = np.zeros(T, dtype=bool)
    forced_index_12[d:(2 * d)] = True
    forced_index_21 = np.zeros(T, dtype=bool)
    forced_index_21[(2 * d):(3 * d)] = True
    forced_index_22 = np.zeros(T, dtype=bool)
    forced_index_22[(3 * d):(4 * d)] = True

    for i in range(15):
        for j in range(q):
            if (2 ** i - 1) * K * q + j > 4 * d - 1 and (2 ** i - 1) * K * q + j < T:
                forced_index_11[(2 ** i - 1) * K * q + j] = True
            if (2 ** i - 1) * K * q + q + j > 4 * d - 1 and (2 ** i - 1) * K * q + q + j < T:
                forced_index_12[(2 ** i - 1) * K * q + q + j] = True
            if (2 ** i - 1) * K * q + 2 * q + j > 4 * d - 1 and (2 ** i - 1) * K * q + 2 * q + j < T:
                forced_index_21[(2 ** i - 1) * K * q + 2 * q + j] = True
            if (2 ** i - 1) * K * q + 3 * q + j > 4 * d - 1 and (2 ** i - 1) * K * q + 3 * q + j < T:
                forced_index_22[(2 ** i - 1) * K * q + 3 * q + j] = True

    # running our algorithm
    for i in range(T):

        if forced_index_11[i]:
            arm11[i] = True
            arm12[i] = True

        elif forced_index_12[i]:
            arm11[i] = True
            arm22[i] = True

        elif forced_index_21[i]:
            arm21[i] = True
            arm12[i] = True

        elif forced_index_22[i]:
            arm21[i] = True
            arm22[i] = True

        else:

            # tilde parameters
            beta_tilde_11 = LinearRegression().fit(X1[forced_index_11 & arm11 & arm12, :],
                                                   r1_4[forced_index_11 & arm11 & arm12] + r2_4[
                                                       forced_index_11 & arm11 & arm12])
            beta_tilde_21 = LinearRegression().fit(X1[forced_index_21 & arm21 & arm12, :],
                                                   r1_4[forced_index_21 & arm21 & arm12] + r2_4[
                                                       forced_index_21 & arm21 & arm12])
            beta_tilde_12 = LinearRegression().fit(X1[forced_index_12 & arm11 & arm22, :],
                                                   r1_4[forced_index_12 & arm11 & arm22] + r2_4[
                                                       forced_index_12 & arm11 & arm22])
            beta_tilde_22 = LinearRegression().fit(X1[forced_index_22 & arm21 & arm22, :],
                                                   r1_4[forced_index_22 & arm21 & arm22] + r2_4[
                                                       forced_index_22 & arm21 & arm22])

            # hat estimators
            beta_hat_11 = LinearRegression().fit(X1[arm11 & arm12, :], r1_4[arm11 & arm12] + r2_4[arm11 & arm12])
            beta_hat_21 = LinearRegression().fit(X1[arm21 & arm12, :], r1_4[arm21 & arm12] + r2_4[arm21 & arm12])
            beta_hat_12 = LinearRegression().fit(X1[arm11 & arm22, :], r1_4[arm11 & arm22] + r2_4[arm11 & arm22])
            beta_hat_22 = LinearRegression().fit(X1[arm21 & arm22, :], r1_4[arm21 & arm22] + r2_4[arm21 & arm22])

            value_tilde = np.array([beta_tilde_11.predict(X1[i].reshape(1, -1)),
                                    beta_tilde_12.predict(X1[i].reshape(1, -1)),
                                    beta_tilde_21.predict(X1[i].reshape(1, -1)),
                                    beta_tilde_22.predict(X1[i].reshape(1, -1))])
            value_hat = np.array([beta_hat_11.predict(X1[i].reshape(1, -1)),
                                  beta_hat_12.predict(X1[i].reshape(1, -1)),
                                  beta_hat_21.predict(X1[i].reshape(1, -1)),
                                  beta_hat_22.predict(X1[i].reshape(1, -1))])
            value_max = value_hat[value_tilde > value_tilde.max() - h / 2].max()

            if value_max == value_hat[0] and (value_tilde > value_tilde - h / 2)[0]:
                arm11[i] = True
                arm12[i] = True
            elif value_max == value_hat[1] and (value_tilde > value_tilde - h / 2)[1]:
                arm11[i] = True
                arm22[i] = True
            elif value_max == value_hat[2] and (value_tilde > value_tilde - h / 2)[2]:
                arm21[i] = True
                arm12[i] = True
            else:
                arm21[i] = True
                arm22[i] = True

            if arm11[i]:
                r1_4[i] = np.dot(beta11, X1[i].T) + eta1[i]
                X2_4[i] = np.dot(B1, X1[i].T) + epsilon[i]
            else:
                r1_4[i] = np.dot(beta21, X1[i].T) + eta1[i]
                X2_4[i] = np.dot(B2, X1[i].T) + epsilon[i]

            if arm12[i]:
                r2_4[i] = np.dot(beta12, X2_4[i].T) + eta2[i]
            else:
                r2_4[i] = np.dot(beta22, X2_4[i].T) + eta2[i]



    # 4 armed bandit + 2 arm
    # algorithm parameters
    #q = 1
    #h = 5
    K = 4

    arm11 = np.zeros(T, dtype=bool)  # indicate if we pull arm 1 in first period at time t
    arm21 = np.zeros(T, dtype=bool)
    arm12 = np.zeros(T, dtype=bool)
    arm22 = np.zeros(T, dtype=bool)
    r1_42 = np.zeros(T)
    r2_42 = np.zeros(T)
    X2_42 = np.zeros((T, d))

    # forced pull index
    forced_index_11 = np.zeros(T, dtype=bool)
    forced_index_11[0:d] = True
    forced_index_12 = np.zeros(T, dtype=bool)
    forced_index_12[d:(2 * d)] = True
    forced_index_21 = np.zeros(T, dtype=bool)
    forced_index_21[(2 * d):(3 * d)] = True
    forced_index_22 = np.zeros(T, dtype=bool)
    forced_index_22[(3 * d):(4 * d)] = True

    for i in range(15):
        for j in range(q):
            if (2 ** i - 1) * K * q + j > 4 * d - 1 and (2 ** i - 1) * K * q + j < T:
                forced_index_11[(2 ** i - 1) * K * q + j] = True
            if (2 ** i - 1) * K * q + q + j > 4 * d - 1 and (2 ** i - 1) * K * q + q + j < T:
                forced_index_12[(2 ** i - 1) * K * q + q + j] = True
            if (2 ** i - 1) * K * q + 2 * q + j > 4 * d - 1 and (2 ** i - 1) * K * q + 2 * q + j < T:
                forced_index_21[(2 ** i - 1) * K * q + 2 * q + j] = True
            if (2 ** i - 1) * K * q + 3 * q + j > 4 * d - 1 and (2 ** i - 1) * K * q + 3 * q + j < T:
                forced_index_22[(2 ** i - 1) * K * q + 3 * q + j] = True

    # running our algorithm
    for i in range(T):

        if forced_index_11[i]:
            arm11[i] = True
            arm12[i] = True

        elif forced_index_12[i]:
            arm11[i] = True
            arm22[i] = True

        elif forced_index_21[i]:
            arm21[i] = True
            arm12[i] = True

        elif forced_index_22[i]:
            arm21[i] = True
            arm22[i] = True

        else:

            # first period
            # tilde parameters
            beta_tilde_11 = LinearRegression().fit(X1[forced_index_11 & arm11 & arm12, :],
                                                   r1_42[forced_index_11 & arm11 & arm12] + r2_42[
                                                       forced_index_11 & arm11 & arm12])
            beta_tilde_21 = LinearRegression().fit(X1[forced_index_21 & arm21 & arm12, :],
                                                   r1_42[forced_index_21 & arm21 & arm12] + r2_42[
                                                       forced_index_21 & arm21 & arm12])
            beta_tilde_12 = LinearRegression().fit(X1[forced_index_12 & arm11 & arm22, :],
                                                   r1_42[forced_index_12 & arm11 & arm22] + r2_42[
                                                       forced_index_12 & arm11 & arm22])
            beta_tilde_22 = LinearRegression().fit(X1[forced_index_22 & arm21 & arm22, :],
                                                   r1_42[forced_index_22 & arm21 & arm22] + r2_42[
                                                       forced_index_22 & arm21 & arm22])

            # hat estimators
            beta_hat_11 = LinearRegression().fit(X1[arm11 & arm12, :], r1_42[arm11 & arm12] + r2_42[arm11 & arm12])
            beta_hat_21 = LinearRegression().fit(X1[arm21 & arm12, :], r1_42[arm21 & arm12] + r2_42[arm21 & arm12])
            beta_hat_12 = LinearRegression().fit(X1[arm11 & arm22, :], r1_42[arm11 & arm22] + r2_42[arm11 & arm22])
            beta_hat_22 = LinearRegression().fit(X1[arm21 & arm22, :], r1_42[arm21 & arm22] + r2_42[arm21 & arm22])

            value_tilde = np.array([beta_tilde_11.predict(X1[i].reshape(1, -1)),
                                    beta_tilde_12.predict(X1[i].reshape(1, -1)),
                                    beta_tilde_21.predict(X1[i].reshape(1, -1)),
                                    beta_tilde_22.predict(X1[i].reshape(1, -1))])
            value_hat = np.array([beta_hat_11.predict(X1[i].reshape(1, -1)),
                                  beta_hat_12.predict(X1[i].reshape(1, -1)),
                                  beta_hat_21.predict(X1[i].reshape(1, -1)),
                                  beta_hat_22.predict(X1[i].reshape(1, -1))])
            value_max = value_hat[value_tilde > value_tilde.max() - h / 2].max()

            if value_max == value_hat[0] and (value_tilde > value_tilde - h / 2)[0]:
                arm11[i] = True
            elif value_max == value_hat[1] and (value_tilde > value_tilde - h / 2)[1]:
                arm11[i] = True
            else:
                arm21[i] = True

            if arm11[i]:
                r1_42[i] = np.dot(beta11, X1[i].T) + eta1[i]
                X2_42[i] = np.dot(B1, X1[i].T) + epsilon[i]
            else:
                r1_42[i] = np.dot(beta21, X1[i].T) + eta1[i]
                X2_42[i] = np.dot(B2, X1[i].T) + epsilon[i]

            # second period
            # tilde parameters
            beta_tilde_1 = LinearRegression().fit(X2_42[(forced_index_11 | forced_index_21) & arm12, :],
                                                  r2_42[(forced_index_11 | forced_index_21) & arm12])
            beta_tilde_2 = LinearRegression().fit(X2_42[(forced_index_12 | forced_index_22) & arm22, :],
                                                  r2_42[(forced_index_12 | forced_index_22) & arm22])

            # hat estimators
            beta_hat_1 = LinearRegression().fit(X2_42[arm12, :], r2_42[arm12])
            beta_hat_2 = LinearRegression().fit(X2_42[arm22, :], r2_42[arm22])

            if beta_tilde_1.predict(X2_42[i].reshape(1, -1)) > beta_tilde_2.predict(X2_42[i].reshape(1, -1)) + h / 2:
                arm12[i] = True
            elif beta_tilde_2.predict(X2_42[i].reshape(1, -1)) > beta_tilde_1.predict(X2_42[i].reshape(1, -1)) + h / 2:
                arm22[i] = True
            elif beta_hat_1.predict(X2_42[i].reshape(1, -1)) > beta_hat_2.predict(X2_42[i].reshape(1, -1)):
                arm12[i] = True
            else:
                arm22[i] = True

            if arm12[i]:
                r2_42[i] = np.dot(beta12, X2_42[i].T) + eta2[i]
            else:
                r2_42[i] = np.dot(beta22, X2_42[i].T) + eta2[i]

    regret_DTR = (r1_opt + r2_opt) - (r1_DTR + r2_DTR)
    regret_greedy = (r1_opt + r2_opt) - (r1_greedy + r2_greedy)
    regret_4 = (r1_opt + r2_opt) - (r1_4 + r2_4)
    regret_42 = (r1_opt + r2_opt) - (r1_42 + r2_42)

    result = {"regret_DTR": regret_DTR, "regret_greedy": regret_greedy, "regret_4": regret_4, "regret_42": regret_42}
    return result


if __name__ == "__main__":
    main()