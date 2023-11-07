import numpy as np


def SpaceBound(X, Up, Low):
    Dim = len(X)
    S = (X > Up) | (X < Low)
    X = (np.random.rand(1, Dim) * (Up - Low) + Low) * S + X * (~S)
    return X


def levy(d):
    b = 1.5
    s = (np.math.gamma(1 + b) * np.sin(np.pi * b / 2) / (np.math.gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))) ** (1 / b)
    u = np.random.randn(d) * s
    v = np.random.randn(d)
    sigma = u / np.abs(v) ** (1 / b)
    return sigma



def FunRange(FunIndex):
    Dim = 30
    if FunIndex == 1:
        Low, Up = -100, 100
    elif FunIndex == 2:
        Low, Up = -10, 10
    elif FunIndex == 3:
        Low, Up = -100, 100
    elif FunIndex == 4:
        Low, Up = -100, 100
    elif FunIndex == 5:
        Low, Up = -30, 30
    elif FunIndex == 6:
        Low, Up = -100, 100
    elif FunIndex == 7:
        Low, Up = -1.28, 1.28
    elif FunIndex == 8:
        Low, Up = -500, 500
    elif FunIndex == 9:
        Low, Up = -5.12, 5.12
    elif FunIndex == 10:
        Low, Up = -32, 32
    elif FunIndex == 11:
        Low, Up = -600, 600
    elif FunIndex == 12:
        Low, Up = -50, 50
    elif FunIndex == 13:
        Low, Up = -50, 50
    elif FunIndex == 14:
        Low, Up, Dim = -65.536, 65.536, 2
    elif FunIndex == 15:
        Low, Up, Dim = -5, 5, 4
    elif FunIndex == 16:
        Low, Up, Dim = -5, 5, 2
    elif FunIndex == 17:
        Low, Up, Dim = [-5, 0], [10, 15], 2
    elif FunIndex == 18:
        Low, Up, Dim = -2, 2, 2
    elif FunIndex == 19:
        Low, Up, Dim = 0, 1, 3
    elif FunIndex == 20:
        Low, Up, Dim = 0, 1, 6
    elif FunIndex == 21:
        Low, Up, Dim = 0, 10, 4
    elif FunIndex == 22:
        Low, Up, Dim = 0, 10, 4
    else:
        Low, Up, Dim = 0, 10, 4
    return Low, Up, Dim



def BenFunctions(X, FunIndex):
    Dim = len(X)
    if FunIndex == 1:
        Fit = np.sum(X**2)
    elif FunIndex == 2:
        Fit = np.sum(np.abs(X)) + np.prod(np.abs(X))
    elif FunIndex == 3:
        Fit = np.sum([np.sum(X[:i+1])**2 for i in range(Dim)])
    elif FunIndex == 4:
        Fit = np.max(np.abs(X))
    elif FunIndex == 5:
        Fit = np.sum(100*(X[1:Dim] - (X[:Dim-1]**2))**2 + (X[:Dim-1] - 1)**2)
    elif FunIndex == 6:
        Fit = np.sum(np.floor(X + 0.5)**2)
    elif FunIndex == 7:
        Fit = np.sum([(i+1)*(X[i]**4) for i in range(Dim)]) + np.random.rand()
    elif FunIndex == 8:
        Fit = np.sum(-X * np.sin(np.sqrt(np.abs(X))))
    elif FunIndex == 9:
        Fit = np.sum(X**2 - 10*np.cos(2*np.pi*X)) + 10*Dim
    elif FunIndex == 10:
        Fit = -20*np.exp(-0.2*np.sqrt(np.sum(X**2)/Dim)) - np.exp(np.sum(np.cos(2*np.pi*X))/Dim) + 20 + np.exp(1)
    elif FunIndex == 11:
        Fit = np.sum(X**2)/4000 - np.prod(np.cos(X/np.sqrt(np.arange(1, Dim+1)))) + 1
    elif FunIndex == 12:
        a, k, m = 10, 100, 4
        Fit = (np.pi/Dim) * (10 * (np.sin(np.pi * (1 + (X[0] + 1)/4)))**2 + np.sum((((X[:Dim-1] + 1)/4)**2) * (1 + 10 * (np.sin(np.pi * (1 + (X[1:Dim] + 1)/4)))**2) + ((X[Dim-1] + 1)/4)**2) + np.sum(k * ((X - a)**m) * (X > a) + k * ((-X - a)**m) * (X < -a)) )
    elif FunIndex == 13:
        a, k, m = 10, 100, 4
        Fit = 0.1 * ((np.sin(3*np.pi*X[0]))**2 + np.sum((X[:Dim-1] - 1)**2 * (1 + (np.sin(3*np.pi*X[1:Dim]))**2) + ((X[Dim-1] - 1)**2) * (1 + (np.sin(2*np.pi*X[Dim-1]))**2)) + np.sum(k * ((X - a)**m) * (X > a) + k * ((-X - a)**m) * (X < -a)) )
    elif FunIndex == 14:
        a = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32], [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
        b = np.array([np.sum((X - a[:,j])**6) for j in range(25)])
        Fit = (1/500 + np.sum(1/(np.arange(1, 26) + b)))**(-1)
    elif FunIndex == 15:
        a = [0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
        b = 1/np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
        Fit = np.sum((a - ((X[0] * (b**2 + X[1]*b)) / (b**2 + X[2]*b + X[3])))**2)
    elif FunIndex == 16:
        Fit = 4*(X[0]**2) - 2.1*(X[0]**4) + (X[0]**6)/3 + X[0]*X[1] - 4*(X[1]**2) + 4*(X[1]**4)
    elif FunIndex == 17:
        Fit = (X[1] - (X[0]**2)*5.1/(4*(np.pi**2)) + 5/np.pi*X[0] - 6)**2 + 10*(1 - 1/(8*np.pi))*np.cos(X[0]) + 10
    elif FunIndex == 18:
        Fit = (1 + (X[0] + X[1] + 1)**2 * (19 - 14*X[0] + 3*(X[0]**2) - 14*X[1] + 6*X[0]*X[1] + 3*X[1]**2)) * (30 + (2*X[0] - 3*X[1])**2 * (18 - 32*X[0] + 12*(X[0]**2) + 48*X[1] - 36*X[0]*X[1] + 27*(X[1]**2)))
    elif FunIndex == 19:
        a = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        c = [1, 1.2, 3, 3.2]
        p = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
        Fit = np.sum([-c[i] * np.exp(-np.sum(a[i,:] * ((X - p[i,:])**2))) for i in range(4)])
    elif FunIndex == 20:
        a = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
        c = [1, 1.2, 3, 3.2]
        p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        Fit = np.sum([-c[i] * np.exp(-np.sum(a[i,:] * ((X - p[i,:])**2))) for i in range(4)])
    return Fit


def fn_EEFO(fun_index, max_it, pop_size):
    low, up, dim = FunRange(fun_index)

    lb = np.array([low] * dim)
    ub = np.array([up] * dim)

    pop_pos = np.random.rand(pop_size, dim) * (ub - lb) + lb
    pop_fit = np.zeros(pop_size)
    pop_fit[0] = BenFunctions(pop_pos[0, :], fun_index)

    best_f = pop_fit[0]
    xprey = pop_pos[0, :]

    for i in range(1, pop_size):
        if pop_fit[i] <= best_f:
            best_f = pop_fit[i]
            xprey = pop_pos[i, :]

    his_best_f = np.zeros(max_it)

    for it in range(max_it):
        direct_vector = np.zeros((pop_size, dim))
        e0 = 4 * np.sin(1 - it / max_it)

        for i in range(pop_size):
            E = e0 * np.log(1 / np.random.rand())  # Eq.(30)

            if dim == 1:
                direct_vector[i, :] = 1
            else:
                rand_num = np.ceil((max_it - it) / max_it * np.random.rand() * (dim - 2) + 2)  # Eq.(6)
                rand_dim = np.random.permutation(dim)
                direct_vector[i, rand_dim[0:int(rand_num)]] = 1

            if E > 1:
                K = [j for j in range(pop_size) if j != i]
                j = K[np.random.randint(pop_size - 1)]

                if pop_fit[j] < pop_fit[i]:
                    if np.random.rand() > 0.5:
                        new_pop_pos = pop_pos[j, :] + np.random.randn() * direct_vector[i, :] * (np.mean(pop_pos, axis=0) - pop_pos[i, :])
                    else:
                        xr = np.random.rand(dim) * (ub - lb) + lb
                        new_pop_pos = pop_pos[j, :] + 1 * np.random.randn() * direct_vector[i, :] * (xr - pop_pos[i, :])
                else:
                    if np.random.rand() > 0.5:
                        new_pop_pos = pop_pos[i, :] + np.random.randn() * direct_vector[i, :] * (np.mean(pop_pos, axis=0) - pop_pos[j, :])
                    else:
                        xr = np.random.rand(dim) * (ub - lb) + lb
                        new_pop_pos = pop_pos[i, :] + np.random.randn() * direct_vector[i, :] * (xr - pop_pos[j, :])
            else:
                if np.random.rand() < 1 / 3:
                    # resting
                    alpha = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())  # Eq.(15)
                    rn = np.random.randint(pop_size)
                    rd = np.random.randint(dim)

                  
                    Z = pop_pos[rn, rd] * np.ones(dim)

                    Ri = Z + alpha * np.abs(Z - xprey)  # Eq.(14)
                    new_pop_pos = Ri + np.random.randn() * (Ri - np.round(np.random.rand()) * pop_pos[i, :])  # Eq.(16)
                elif np.random.rand() > 2 / 3:
                    # migrating
                    rn = np.random.randint(pop_size)
                    rd = np.random.randint(dim)

                   
                    Z = pop_pos[rn, rd] * np.ones(dim)

                    alpha = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())
                    Ri = Z + alpha * np.abs(Z - xprey)  # resting area

                    beta = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())  # Eq.(21)
                    Hr = xprey + beta * np.abs(np.mean(pop_pos, axis=0) - xprey)  # hunting area

                    L = 0.01 * np.abs(levy(dim))  # Eq.(26)
                    new_pop_pos = -np.random.rand() * Ri + np.random.rand() * Hr - L * (Hr - pop_pos[i, :])  # Eq.(24)
                else:
                    # Hunting
                    beta = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())  # Eq.(21)
                    hprey = xprey + beta * np.abs(np.mean(pop_pos, axis=0) - xprey)  # Eq.(20) hunting area
                    r4 = np.random.rand()
                    eta = np.exp(r4 * (1 - it) / max_it) * (np.cos(2 * np.pi * r4))  # Eq.(23)
                    new_pop_pos = hprey + eta * (hprey - np.round(np.random.rand()) * pop_pos[i, :])  # Eq.(22) hunting

            new_pop_pos = SpaceBound(new_pop_pos, ub, lb)
            new_pop_fit = BenFunctions(new_pop_pos, fun_index)

           
            if new_pop_fit < pop_fit[i]:
                pop_fit[i] = new_pop_fit
                pop_pos[i, :] = new_pop_pos

                if pop_fit[i] <= best_f:
                    best_f = pop_fit[i]
                    xprey = pop_pos[i, :]

        his_best_f[it] = best_f

    return xprey, best_f, his_best_f
