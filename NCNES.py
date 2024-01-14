import numpy as np
import math
import time as tm

from cec2017.functions import all_functions
from model import SearchProcess

problemSet = range(0, 30)
results = np.zeros((31, 2))
start_time = tm.time()
for problem in problemSet:
    f = all_functions[problem - 1]
    bound = [-100, 100]
    D = 10
    # env = TestEnv(D, problem)
    # lu = np.mat(np.tile(test_func_bound[problem], (D, 1)))
    lu = np.mat(np.tile(bound, (D, 1)))
    lambd = math.ceil(math.log(D))
    mu = 4 + math.floor(3 * math.log(D))
    phi_init = 0.00005
    eta_m_init = 1
    eta_c_init = (3 + math.log(D)) / (5 * math.sqrt(D))
    vl = np.tile(lu[:, 0], (1, mu))
    vu = np.tile(lu[:, 1], (1, mu))

    MAXFES = 10000 * D
    total_time = 10

    outcome = np.ones(total_time) * 1e300

    sp = [SearchProcess(D, mu) for _ in range(lambd)]

    time = 1
    print(f'The {problem}th problem started!----------------------------------')

    while time <= total_time:
        min_f = 1e300
        FES = 0

        for i in range(lambd):
            sp[i].mean = lu[:, 0] + np.multiply(np.random.rand(D, 1), (lu[:, 1] - lu[:, 0]))
            sp[i].cov = (lu[:, 1] - lu[:, 0]) / 2 * lambd
            sp[i].cov = np.power(sp[i].cov, 2)

        while FES < MAXFES:
            eta_m = eta_m_init * ((np.exp(1) - np.exp(FES / MAXFES)) / (np.exp(1) - 1))
            eta_c = eta_c_init * ((np.exp(1) - np.exp(FES / MAXFES)) / (np.exp(1) - 1))

            for i in range(lambd):
                spi = sp[i]
                spi.x = np.tile(spi.mean, (1, mu)) + np.multiply(np.random.randn(D, mu), np.tile(np.sqrt(spi.cov), (1, mu)))
                # try:
                #     check_nan_2d('x', spi.x)
                # except Exception:
                #     print('mean:')
                #     print(spi.mean)
                #     print('cov:')
                #     print(spi.cov)
                if problem != 7 and problem != 25:
                    spi.x = np.where(spi.x < vl, 2 * vl - spi.x, spi.x)
                    spi.x = np.where(spi.x > vu, 2 * vu - spi.x, spi.x)
                    spi.x = np.where(spi.x < vl, vl, spi.x)

                # check_nan_2d('x', spi.x)

                spi.fit = f(spi.x.T)
                FES += mu

                min_f = min(min(spi.fit), min_f)

                order = np.argsort(spi.fit)
                rank = np.argsort(order)
                rank = rank + 1

                tempU = np.maximum(0, np.log(mu / 2 + 1) - np.log(rank))
                utility = tempU / np.sum(tempU) - 1 / mu

                # check_nan_1d('utility', utility)


                invCov_i = 1 / spi.cov
                # check_nan_1d('invCov', invCov_i)

                difXtoMean = spi.x - np.tile(spi.mean, (1, mu))
                # check_nan_2d('dif', difXtoMean)
                deltaMean_f = np.multiply(invCov_i,
                                          np.reshape(np.mean(np.multiply(difXtoMean, np.tile(utility, (D, 1))), 1),
                                                     (D, 1)))
                # check_nan_1d('deltaMean_f', deltaMean_f)
                deltaCov_f = np.multiply(np.power(invCov_i, 2),
                                         np.reshape(
                                             np.mean(np.multiply(np.power(difXtoMean, 2), np.tile(utility, (D, 1))), 1),
                                             (D, 1))) / 2
                # check_nan_1d('deltaCov_f', deltaCov_f)
                deltaMean_d = np.zeros((D, 1))
                deltaCov_d = np.zeros((D, 1))
                for j in range(lambd):
                    temp1 = 1 / (spi.cov + sp[j].cov) * 2
                    temp2 = np.multiply(temp1, spi.mean - sp[j].mean)
                    deltaMean_d = deltaMean_d + temp2 / 4
                    deltaCov_d = deltaCov_d + (temp1 - np.power(temp2, 2) / 4 - invCov_i) / 4
                # check_nan_1d('deltaMean_d', deltaMean_d)
                # check_nan_1d('deltaCov_d', deltaCov_d)
                meanFisher = np.multiply(np.power(invCov_i, 2), np.reshape(np.mean(np.power(difXtoMean, 2), 1), (D, 1)))
                # check_nan_2d('meanFisher', meanFisher)
                covFisher = np.reshape(np.mean(np.power(
                    np.multiply(np.tile(np.power(invCov_i, 2), (1, mu))
                                , np.power(difXtoMean, 2)) - np.tile(invCov_i, (1, mu))
                    , 2),
                    1), (D, 1)) / 4
                # check_nan_2d('covFisher', covFisher)
                spi.mean = spi.mean + np.multiply(1 / meanFisher, deltaMean_f + deltaMean_d * phi_init) * eta_m
                spi.cov = spi.cov + np.multiply(1 / covFisher, deltaCov_f + deltaCov_d * phi_init) * eta_c
                spi.cov = np.abs(spi.cov)

                if problem != 7 and problem != 25:
                    spi.mean = np.where(spi.mean < lu[:, 0], 2 * lu[:, 0] - spi.mean, spi.mean)
                    spi.mean = np.where(spi.mean > lu[:, 1], 2 * lu[:, 1] - spi.mean, spi.mean)
                    spi.mean = np.where(spi.mean < lu[:, 0], lu[:, 0], spi.mean)
            print(f"The best result at the {FES}th FE is {min_f}")
        outcome[time - 1] = min_f
        time += 1

    results[problem][0] = np.mean(outcome)
    results[problem][1] = np.std(outcome)
    print(f"the {problem}th problem result is:")
    print(f"the mean result is: {np.mean(outcome)} and the std is {np.std(outcome)}")

    with open("./newresults", "a") as f:
        f.write(f"the {problem}th problem result is:")
        f.write(f"the mean result is: {results[problem][0]} and the std is {results[problem][1]}")
        f.write('\n')

end_time = tm.time()
duration = format(end_time - start_time, '.3f')
print('{}s'.format(duration))

with open("./newresults", "a") as f:
    f.write(f"Total time: {duration}s")
    f.write('\n\n')
