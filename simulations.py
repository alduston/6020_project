import pandas as pd
import numpy as np
import random
from copy import copy
import time
from sklearn.datasets import fetch_rcv1
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def f(x, A, b, lamb1 = 10 ** (-4), lamb2 = 10 ** (-4)):
    alpha = 0
    n = len(A)
    for i, ai in enumerate(A):
        bi = b[i]
        alpha += np.log(1 + np.exp(-1 * bi * np.dot(x, ai)))
    beta = lamb1 * np.linalg.norm(x, ord = 1)
    gamma = .5 * lamb2 * (np.linalg.norm(x) ** 2)
    v = (alpha/n) + beta + gamma
    return v


def fix_strX(X, col_lim = 500):
    n = len(X)
    arrays = []
    for str in X[X.columns[0]]:
        str_list = [item for item in str.split(' ')[:col_lim] if len(item)]
        array = np.asarray([float(item) for item in str_list])
        arrays.append(array)
    X = np.asarray(arrays).reshape(n, len(arrays[0]))
    return pd.DataFrame(X)


def prox_decision(x, lamb, t):
    if x > lamb * t:
        return x - lamb * t
    elif x < -lamb * t:
        return x + lamb * t
    return 0


def prox_opp(x,t, lamb = 2):
    px = []
    if isinstance(x, float):
        return prox_decision(x, lamb, t)
    return np.asarray([prox_decision(xi, lamb, t) for xi in x]).reshape(x.shape)


def get_cordinate_grads(A,b, lamb2 = 10 ** (-4)):
    cordinate_grads = []
    for i, ai in enumerate(A):
        bi = b[i]
        def cord_grad(x, lamb2 = lamb2, bi = bi, ai = ai):
            v = -1 * bi * ai
            alpha = np.exp(np.dot(v ,x)) * v * (1/(1 + np.exp(np.dot(v ,x))))
            return (alpha + (lamb2 * x))
        cordinate_grads.append(copy(cord_grad))
    return cordinate_grads


def grad_F(x, grad_fis):
    l = len(grad_fis)
    return (1/l) * np.sum([grad_fi(x) for grad_fi in grad_fis], axis = 0)


def G(x, t, lamb1, grad_fis, grad = grad_F):
    v = x - (t * grad(x, grad_fis))
    Gtx = (x - prox_opp(v, t, lamb1))/t
    return Gtx


def diag_hess_g(beta, A, b, lamb2):
    l = len(beta)
    hess = np.zeros((l, l))
    for j in range(l):
        for k in range(l):
            if j == k:
                q1 = -1 * b[k] * (np.dot(A[k] @ b))
                q2 = -1 * b[k] * (np.dot(A[k]))
                Q1 = lamb2[k] + np.exp(q1) * (q2[k]**2) * (1/(1 + np.exp(q1)))
                Q2 = (np.exp(q1)**2) * q2[k] * (1/((1 + np.exp(q1))**2))
                hess[j, k] += Q1 - Q2
    return hess


def newton_prox_decision(x, lamb, t, diag):
    if x > lamb * t * diag:
        return x - lamb * t * diag
    elif x < -lamb * t * diag:
        return x + lamb * t * diag
    return 0


def newton_prox_opp(x,t, B_inv, lamb = 2):
    px = []
    diag = list(np.diag(B_inv))
    if isinstance(x, float):
        return newton_prox_decision(x, lamb , t, * diag[0])
    return np.asarray([newton_prox_decision(xi, lamb, t, diag[i]) for (i,xi) in enumerate(x)]).reshape(x.shape)


def quasi_newton_descent(objective, x_0, lamb1, lamb2, grad_fis, grad_F, output = True):
    t = 1
    i = 0
    xi = x_0
    xtildes = [xi]

    while f(v) > objective:
        grad = grad_F(xi, grad_fis)
        B_inv = np.linalg.inv(diag_hess_g(xi, lamb2))
        G = (xi - newton_prox_opp(xi  - t*B_inv @ grad, t, B_inv, lamb1))/t
        v = xi - t * G

        while delta < 0:
            t *= .5
            G = (xi - newton_prox_opp(xi  - t*B_inv @ grad, t, B_inv, lamb1)) / t
            v = xi - t * G

        xi = v
        i += 1
        t = 1
        xtildes.append(xi)

    if output:
        print(f'After {i} iterations, proximal newton descent converged to objective {f(xi)}')
    return xtildes


def acc_proximal_descent(objective, x_0, f,  lamb1, grad_fis, grad_F, t,  output = True):
    i = 0
    xi = x_0.copy()
    x_tildes = [xi]

    for i in range(2):
        v = x_tildes[-1] - (t * grad_F(x_tildes[-1], grad_fis))
        x_tildes.append(prox_opp(v, t, lamb1))
        i += 1

    while f(x_tildes[-1]) > objective:
        gamma = x_tildes[-1] + ((i - 2) / (i + 1)) * (x_tildes[-1] - x_tildes[-2])
        v = gamma - (t * grad_F(gamma, grad_fis))
        x_tildes.append(prox_opp(v, t, lamb1))
        i += 1
        print(f(x_tildes[-1]))

    x = x_tildes[-1]
    if output:
        print(f'After {i} iterations, accelerated proximal descent converged to objective {f(x)}')
    return x_tildes


def stoch_acc_prox_descent(objective, x_0, f,  lamb1, grad_fis, grad_F, t,  output = True):
    i = 0
    xi = x_0.copy()
    x_tildes = [xi]
    n = len(grad_fis)
    for i in range(2):
        grad = random.choice(grad_fis)
        v = x_tildes[-1] - (t * grad(x_tildes[-1]))
        x_tildes.append(prox_opp(v, t, lamb1))
        i += 1

    while f(x_tildes[-1]) > objective and i < 100000:
        grad = random.choice(grad_fis)
        gamma = x_tildes[-1] + ((i - 2) / (i + 1)) * (x_tildes[-1] - x_tildes[-2])
        v = gamma - (t * grad(gamma))
        x_tildes.append(prox_opp(v, t, lamb1))
        i += 1
        t = 1/i
        print(f(x_tildes[-1]))

    x = x_tildes[-1]
    if output:
        print(f'After {i} iterations, accelerated proximal descent converged to objective {f(x)}')
    return x_tildes



def proximal_descent(objective, x_0, f, lamb1, grad_fis, t, output = True):
    i = 0
    xi = x_0
    x_tildes = [xi]
    while f(xi) > objective:
        Gtx = G(xi,t, lamb1, grad_fis)
        v = xi - t * Gtx

        xi = v
        i += 1

        x_tildes.append(xi)
        print(f(xi))
    if output:
        print(f'After {i} iterations, proximal descent converged to objective {f(xi)}')
    return x_tildes


def Prox_SVRG(x_0, f, grad_fis, t, m, lamb1, objective, output = True):
    n = len(x_0)
    l = len(grad_fis)

    outer_loops = 0
    x_tilde = x_0
    x_tildes = [x_tilde]
    while f(x_tilde) > objective:
        x_ks = [copy(x_tilde)]
        full_grad = (1/l) * np.sum([grad_fi(x_tilde) for grad_fi in grad_fis], axis = 0)
        for i in range(m - 1):
            x_km1 = x_ks[-1]
            k = random.randrange(n)
            grad = grad_fis[k]

            v = grad(x_km1) - grad(x_tilde) + full_grad
            x_k = prox_opp(x_km1 - t*v, t, lamb1)
            x_ks.append(x_k)

        w = (1/m) * np.sum(x_ks, axis = 0)
        x_tilde = w
        x_tildes.append(x_tilde)
        outer_loops += 1
        print(f(x_tilde))

    if output:
        print(f'After {outer_loops} outer loop iterations, converged to objective {f(x_tilde)}')
    return x_tildes


def Prox_SVRG_var(x_0, f, grad_fis, t, m, lamb1, objective, alt = False, output = True):
    n = len(x_0)
    l = len(grad_fis)

    vars = []
    reduced_vars = []

    outer_loops = 0
    x_tilde = x_0
    x_tildes = [x_tilde]
    while f(x_tilde) > objective:
        pass_vars = []
        pass_reduced_vars = []
        x_ks = [copy(x_tilde)]
        full_grad = (1/l) * np.sum([grad_fi(x_tilde) for grad_fi in grad_fis], axis = 0)
        for i in range(m - 1):
            x_km1 = x_ks[-1]
            k = random.randrange(n)
            grad = grad_fis[k]

            v = grad(x_km1) - grad(x_tilde) + full_grad
            x_k = prox_opp(x_km1 - t*v, t, lamb1)
            x_ks.append(x_k)

            pass_vars.append(np.linalg.norm(full_grad - grad(x_km1))**2)
            pass_reduced_vars.append(np.linalg.norm(full_grad - v)**2)

        vars.append(np.mean(pass_vars))
        reduced_vars.append(np.mean(pass_reduced_vars))

        if not alt:
            w = (1/m) * np.sum(x_ks, axis = 0)
        else:
            w = x_ks[-1]

        x_tilde = w
        x_tildes.append(x_tilde)
        outer_loops += 1
        print(f(x_tilde))

    if output:
        print(f'After {outer_loops} outer loop iterations, converged to objective {f(x_tilde)}')
    return vars, reduced_vars


def Prox_SVRG_2(x_0, f, grad_fis, t, m, lamb1, objective, output = True):
    n = len(x_0)
    l = len(grad_fis)

    outer_loops = 0
    x_tilde = x_0
    x_tildes = [x_tilde]
    while f(x_tilde) > objective:
        x_ks = [copy(x_tilde)]
        full_grad = (1/l) * np.sum([grad_fi(x_tilde) for grad_fi in grad_fis], axis = 0)
        for i in range(m - 1):
            x_km1 = x_ks[-1]
            k = random.randrange(n)
            grad = grad_fis[k]

            v = grad(x_km1) - grad(x_tilde) + full_grad
            x_k = prox_opp(x_km1 - t*v, t, lamb1)
            x_ks.append(x_k)

        x_tilde = x_ks[-1]
        x_tildes.append(x_tilde)
        outer_loops += 1
        print(f(x_tilde))

    if output:
        print(f'After {outer_loops} outer loop iterations, converged to objective {f(x_tilde)}')
    return x_tildes


def normalize(X):
    X_normal = []
    for xi in X:
        m = np.linalg.norm(xi)
        if m:
            X_normal.append((1/m)* xi)
        else:
            X_normal.append(xi)
    return np.asarray(X_normal).reshape(X.shape)


def experiment_1(X, y, f, lamb1, lamb2, objective):
    L = 1 / 4
    t = 1 / L
    grad_fis = get_cordinate_grads(X, y, lamb2)
    colors = ['blue', 'red', 'green', 'black', 'violet']
    i = 0
    q = 10 ** (-1 * (len(str(objective)) - 2))
    max_min_obj = 0
    for k in [1, 2, 4, 6, 8]:
        m = k * len(X)
        x_0 = np.zeros(X.shape[1])
        x_tildes = Prox_SVRG(x_0, f, grad_fis, t, m, lamb1, objective)

        objetives = [f(x) - objective + q for x in x_tildes]
        passes = [(k+1) * i for i in range(len(x_tildes))]
        max_min_obj = max(max_min_obj, objetives[-1])
        plt.plot(passes, objetives, label=f'm = {k}n', color=colors[i])
        plt.yscale('log')
        i += 1

    plt.xlabel('Number of effective passes')
    plt.ylabel(r'Objective gap $P(x_k) - P_*$')
    plt.ylim(bottom=max_min_obj)
    plt.legend()
    plt.savefig('plots/experiment_1_plot.png')
    plt.close('all')


def experiment_2(X, y, f, lamb1, lamb2, objective):
    L = 1 / 4
    grad_fis = get_cordinate_grads(X, y, lamb2)
    colors = ['blue', 'red', 'green', 'black', 'violet']
    i = 0
    q = 10 ** (-1 * (len(str(objective)) - 2))
    max_min_obj = 0
    for k in [1,.1,.01]:
        t = k / L
        m = 4 * len(X)
        x_0 = np.zeros(X.shape[1])
        x_tildes = Prox_SVRG(x_0, f, grad_fis, t, m, lamb1, objective)

        objetives = [f(x) - objective + q for x in x_tildes]
        max_min_obj = max(max_min_obj, objetives[-1])
        passes = [3*i for i in range(len(x_tildes))]
        plt.plot(passes, objetives, label= r'$\eta$ = ' + f'{k}/L', color=colors[i])
        plt.yscale('log')
        i += 1

    plt.xlabel('Number of effective passes')
    plt.ylabel(r'Objective gap $P(x_k) - P_*$')
    plt.ylim(bottom = max_min_obj)
    plt.legend()
    plt.savefig('plots/experiment_2_plot.png')
    plt.close('all')


def experiment_3(X, y, f, lamb1, lamb2, objective):
    L = 1 / 4
    m = 2 * len(X)
    t = 1 / L
    grad_fis = get_cordinate_grads(X, y, lamb2)
    x_0 = np.zeros(X.shape[1])

    q = 10 ** (-1 * (len(str(objective)) - 2))
    x_tildes_1 = Prox_SVRG(x_0, f, grad_fis, t, m, lamb1, objective)
    objetives_1 = [f(x) - objective + q for x in x_tildes_1]
    passes_1 = [3*i for i in range(len(x_tildes_1))]
    plt.plot(passes_1, objetives_1, label='Prox-SVRG', color='blue')
    plt.yscale('log')

    x_tildes_2 = acc_proximal_descent(objective, x_0, f, lamb1, grad_fis, grad_F, t)
    objetives_2 = [f(x) - objective + q for x in x_tildes_2]
    passes_2 = [i for i in range(len(x_tildes_2))]
    plt.plot(passes_2, objetives_2, label='Accelerated Prox-FG', color='red')
    plt.yscale('log')

    plt.xlabel('Number of effective passes')
    plt.ylabel(r'Objective gap $P(x_k) - P_*$')
    plt.ylim(bottom = max(objetives_1[-1], objetives_2[-1]))
    plt.legend()
    plt.savefig('plots/experiment_3_plot.png')
    plt.close('all')


def experiment_4(X, y, f, lamb1, lamb2, objective):
    L = 1 / 4
    grad_fis = get_cordinate_grads(X, y, lamb2)
    colors = ['blue', 'red', 'green', 'black', 'violet']
    i = 0
    q = 10 ** (-1 * (len(str(objective)) - 2))
    for k in [1,.1,.01]:
        t = k / L
        m = 2 * len(X)
        x_0 = np.zeros(X.shape[1])
        x_tildes = Prox_SVRG(x_0, f, grad_fis, t, m, lamb1, objective)
        NNZ = [np.linalg.norm(x, ord = 0) for x in x_tildes]
        passes = [3*i for i in range(len(x_tildes))]
        plt.plot(passes, NNZ, label=f't = L/{k}', color=colors[i])
        i += 1

    plt.xlabel('Number of effective passes')
    plt.ylabel('Number of nonzero cordinates')
    plt.legend()
    plt.savefig('plots/experiment_4_plot.png')
    plt.close('all')


def experiment_5(X, y, f, lamb1, lamb2, objective):
    L = 1 / 4
    m = 4 * len(X)
    t = 1 / L
    grad_fis = get_cordinate_grads(X, y, lamb2)
    x_0 = np.zeros(X.shape[1])
    q = 10 ** (-1 * (len(str(objective)) - 2))

    x_tildes_1 = Prox_SVRG(x_0, f, grad_fis, t, m, lamb1, objective)
    objetives_1 = [f(x) - objective + q for x in x_tildes_1]
    passes_1 = [5*i for i in range(len(x_tildes_1))]
    plt.plot(passes_1, objetives_1, label='Prox-SVRG', color='blue')
    plt.yscale('log')

    x_tildes_2 = Prox_SVRG_2(x_0, f, grad_fis, t, m, lamb1, objective)
    objetives_2 = [f(x) - objective + q for x in x_tildes_2]
    passes_2 = [5*i for i in range(len(x_tildes_2))]
    plt.plot(passes_2, objetives_2, label='NM-Prox-SVRG', color='red')
    plt.yscale('log')

    x_tildes_3 = acc_proximal_descent(objective, x_0, f, lamb1, grad_fis, grad_F, t)
    objetives_3 = [f(x) - objective + q for x in x_tildes_3]
    passes_3 = [i for i in range(len(x_tildes_3))]
    plt.plot(passes_3, objetives_3, label='Accelerated Prox-FG', color='green')
    plt.yscale('log')

    '''
    n = len(grad_fis)
    x_tildes_4 = stoch_acc_prox_descent(objective, x_0, f, lamb1, grad_fis, grad_F, t)
    objetives_4 = [f(x) - objective + q for x in x_tildes_4]
    passes_4 = [i//n for i in range(len(x_tildes_4))]
    plt.plot(passes_4, objetives_4, label='Accelerated Prox-SG', color='black')
    plt.yscale('log')
    '''

    plt.xlabel('Number of effective passes')
    plt.ylabel(r'Objective gap $P(x_k) - P_*$')
    plt.ylim(bottom = max(objetives_1[-1], objetives_2[-1], objetives_3[-1]))
    plt.legend()
    plt.savefig('plots/experiment_5_plot.png')
    plt.close('all')


def experiment_6(X, y, f, lamb1, lamb2, objective):
    L = 1 / 4
    m = 4 * len(X)
    t = 1 / L
    grad_fis = get_cordinate_grads(X, y, lamb2)
    x_0 = np.zeros(X.shape[1])

    q = 10 ** (-1 * (len(str(objective)) - 2))
    vars1, reduced_vars1 = Prox_SVRG_var(x_0, f, grad_fis, t, m, lamb1, objective)
    passes1 = [5 * i for i in range(len(vars1))]
    plt.plot(passes1, vars1, label=r'Prox-SVRG $f_{i,k}$ variance', color='blue')
    plt.plot(passes1, reduced_vars1, label=r'Prox-SVRG $v_k$ variance', color='red')
    plt.yscale('log')


    q = 10 ** (-1 * (len(str(objective)) - 2))
    vars2, reduced_vars2 = Prox_SVRG_var(x_0, f, grad_fis, t, m, lamb1, objective, alt = True)
    passes2 = [5 * i for i in range(len(vars2))]
    plt.plot(passes2, vars2, label=r'NM-Prox-SVRG $f_{i,k}$ variance', color='green')
    plt.plot(passes2, reduced_vars2, label=r'NM-Prox-SVRG $v_k$ variance', color='black')
    plt.yscale('log')

    x_lim = min(passes1[-1], passes2[-1])
    plt.xlim(right = x_lim)
    plt.xlabel('Number of effective passes')
    plt.ylabel('Gradient variance')
    plt.legend()
    plt.savefig('plots/experiment_6_plot.png')
    plt.close('all')


def run():

    #X_0, y_0 = fetch_rcv1(subset = 'train', return_X_y=True)
    #X_0 = X_0.toarray()
    #y_0 = y_0.toarray()

    #pd.DataFrame(X_0).to_csv('data/rcv1/train.csv')
    #pd.DataFrame(y_0).to_csv('data/rcv1/target.csv')


    #X1_df = fix_strX(pd.read_csv('data/sido0/sido0_train.csv'), col_lim= 5000)
    #y1_df = pd.read_csv('data/sido0/sido0_train_target.csv')

    #X1 = np.asarray(X1_df.drop(X1_df.columns[0], axis=1))
    #y1 = np.asarray(y1_df).reshape(len(y1_df),)
    #def f1(x):
        #return f(x, X1, y1)


    X2_df = pd.read_csv('data/covertype/covtype.csv')
    X2 = normalize(np.asarray(X2_df.drop('Cover_Type', axis = 1))[:1000])
    y2 = np.asarray([1 if val == 2 else 0 for val in X2_df.Cover_Type])[:1000]

    lamb1 = 10 ** (-4)
    lamb2 = 10 **(-5)

    def f2(x):
        return f(x, X2, y2, lamb1 = lamb1, lamb2 = lamb2)

    #objective = 0.2897000432320591
    objective_1 = 0.289700045
    objective_2 = 0.2897000432320591

    experiment_1(X2, y2, f2, lamb1, lamb2, objective_1)
    experiment_2(X2, y2, f2, lamb1, lamb2, objective_1)
    #experiment_3(X2, y2, f2, lamb1, lamb2, objective)
    #experiment_4(X2, y2, f2, lamb1, lamb2, objective)
    #experiment_5(X2, y2, f2, lamb1, lamb2, objective)
    experiment_6(X2, y2, f2, lamb1, lamb2, objective_2)



if __name__ == '__main__':
    run()
