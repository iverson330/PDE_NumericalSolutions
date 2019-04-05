import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold=np.inf)

class ConvectionEquation:
    '''
    x range: [ -X, X ]
    t range: [0, T]
    '''

    def __init__(self, a, h, tau, T, X, scheme):
        self.a = a
        self.h = h
        self.tau = tau
        self.lambda_ = tau / h
        self.T = T
        self.X = X
        self.scheme = scheme

        self.N = int(self.T / self.tau)
        self.J = int(self.X / self.h)
        # t_n: t_0=0, t_1 = tau, .... t_N = tau * self.N
        # x_j: x_{-J} = -J * h, ..., x_0 = 0, .... , x_J = J * h

        self.u_matrix = np.zeros([self.N+1, 1+2*self.J])

    def solve(self):
        self.set_initial_condition()
        self.solve_engine(scheme)
        self.error_analysis()
        self.plot_u_matrix()
        self.plot_terminal()


    def initial_condition_function(self, j):
        # x_j = j * h
        if j < 0 or j == 0:
            return 1
        else:
            return 0

    def set_initial_condition(self):
        n = 0
        for j in range(-self.J, self.J+1):
            self.set_u_matrix_elements(j, n, self.initial_condition_function(j))

    def set_u_matrix_elements(self, j, n, u):
        self.u_matrix[self.N-n][j+self.J] = u

    def get_u_matrix_elements(self, j, n):
        return self.u_matrix[self.N-n][j+self.J]

    def set_matrix_elements(self, M, j, n, element):
        M[self.N-n][j+self.J] = element

    def get_matrix_elements(self, M, j, n):
        return M[self.N-n][j+self.J]

    def print_u_matrix(self):
        print(self.u_matrix)
        # for x in range(self.N+1):
        #     line_string = ""
        #     for y in range(1+2*self.J):
        #         line_string += str(self.u_matrix[x][y])
        #     print(line_string)

    def plot_terminal(self):
        terminal_series = self.u_matrix[0]
        terminal_error_series = self.error_matrix[0]

        fig = plt.figure()
        plt.plot(np.arange(-self.X, self.X+self.h, self.h), terminal_series)
        plt.title("Terminal solution, a=%s, scheme=%s"%(str(self.a), str(self.scheme)))
        plt.xlabel("x")
        # plt.show()
        plt.savefig("figures\\Terminalsolutiona=%sscheme=%s"%(str(self.a), str(self.scheme)))

        fig = plt.figure()
        plt.plot(np.arange(-self.X, self.X + self.h, self.h), terminal_error_series)
        plt.title("Terminal relative error, a=%s, scheme=%s" % (str(self.a), str(self.scheme)))
        plt.xlabel("x")
        # plt.show()
        plt.savefig("figures\\Terminalrelativeerrora=%sscheme=%s" % (str(self.a), str(self.scheme)))

    def real_solution(self, j):
        if j * self.h > 4 * self.a:
            return 0
        else:
            return 1

    def error_analysis(self):
        self.error_matrix = np.zeros([self.N + 1, 1 + 2 * self.J])
        for n in range(0, self.N+1, 1):
            for j in range(- self.J, self.J+1):
                u = self.get_matrix_elements(M=self.u_matrix, j=j, n=n)
                real_u = self.real_solution(j=j)
                if u-real_u == 0:
                    error = 0
                else:
                    error = abs((u-real_u)/real_u)
                self.set_matrix_elements(M=self.error_matrix, j=j, n=n, element=error)

        X = np.arange(-self.X, self.X + self.h, self.h)
        T = np.arange(self.T + self.tau, 0, -self.tau)
        X, T = np.meshgrid(X, T)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, T, self.error_matrix, rstride=1, cstride=1, cmap=cm.viridis)
        ax.set_title("Relative error graph, a=%s, scheme=%s" % (str(self.a), str(self.scheme)))
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        # plt.show()
        plt.savefig("figures\\Relativeerrorgrapha=%sscheme=%s" % (str(self.a), str(self.scheme)))

    def plot_u_matrix(self):
        X = np.arange(-self.X, self.X+self.h, self.h)
        T = np.arange(self.T+self.tau, 0, -self.tau)
        X, T = np.meshgrid(X, T)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, T, self.u_matrix, rstride=1, cstride=1, cmap=cm.viridis)
        ax.set_title("Computed solution graph, a=%s, scheme=%s"%(str(self.a), str(self.scheme)))
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        # plt.show()
        plt.savefig("figures\\Computedsolutiongrapha=%sscheme=%s"%(str(self.a), str(self.scheme)))

    def solve_engine(self, scheme):
        '''
        scheme:
        1: u_{j}^{n+1} = u_j^n - a \lambda (u_j^n - u_{j-1}^n)
        2: u_{j}^{n+1} = u_j^n - a \lambda (u_j^n - u_{j-1}^n) - \frac{a \lambda }{2} (1- a \lambda )(u_j^n - 2 u_{j-1}^n + u^{n}_{j-2} )
        3: u_{j}^{n+1 } = \frac{1}{2} ( u_{j+1}^n + u_{j-1}^n ) - \frac{1}{2} a \lambda (u_{j+1}^n - u_{j-1}^n )
        4: u_{j}^{n+1 } = u_j^n - \frac{1}{2} a \lambda (u_{j+1}^n - u_{j-1}^n ) + \frac{1}{2} a^2 \lambda^2 (u_{j+1}^n - 2 u_j^n + u_{j-1}^n )

        '''
        for n in range(1, self.N+1, 1):
            for j in range(- self.J, self.J+1):
                # compute u(n,j)
                if scheme == 1:
                    if j-1 <= -self.J or np.isnan(self.get_u_matrix_elements(n=n-1, j=j)) or np.isnan(self.get_u_matrix_elements(n=n-1, j=j-1)):
                        self.set_u_matrix_elements(n=n, j=j, u=np.nan)
                    else:
                        # u_j^n: self.get_u_matrix_elements(n=n-1, j=j)
                        u = self.get_u_matrix_elements(n=n-1, j=j) - self.a * self.lambda_ * ( self.get_u_matrix_elements(n=n-1, j=j)  - self.get_u_matrix_elements(n=n-1, j=j-1)  )
                        self.set_u_matrix_elements(n=n, j=j, u=u)

                if scheme == 2:
                    if j-2 <= -self.J or np.isnan(self.get_u_matrix_elements(n=n-1, j=j)) or np.isnan(self.get_u_matrix_elements(n=n-1, j=j-1)) or np.isnan(self.get_u_matrix_elements(n=n-1, j=j-2)):
                        self.set_u_matrix_elements(n=n, j=j, u=np.nan)
                    else:
                        u = self.get_u_matrix_elements(n=n-1, j=j) - self.a * self.lambda_ * ( self.get_u_matrix_elements(n=n-1, j=j)  - self.get_u_matrix_elements(n=n-1, j=j-1)  ) \
                        - self.a*self.lambda_ / 2 * (1-self.a*self.lambda_) * (self.get_u_matrix_elements(n=n-1, j=j)  - 2 *self.get_u_matrix_elements(n=n-1, j=j-1) + self.get_u_matrix_elements(n=n-1, j=j-2))
                        self.set_u_matrix_elements(n=n, j=j, u=u)

                if scheme == 3:
                    if j-1 <= -self.J or j+1 >= self.J or np.isnan(self.get_u_matrix_elements(n=n-1, j=j-1)) or np.isnan(self.get_u_matrix_elements(n=n-1, j=j+1)):
                        self.set_u_matrix_elements(n=n, j=j, u=np.nan)
                    else:
                        u = 0.5 * (self.get_u_matrix_elements(n=n-1, j=j+1) + self.get_u_matrix_elements(n=n-1, j=j-1) ) - 0.5*self.a*self.lambda_*(self.get_u_matrix_elements(n=n-1, j=j+1) - self.get_u_matrix_elements(n=n-1, j=j-1) )
                        self.set_u_matrix_elements(n=n, j=j, u=u)

                if scheme == 4:
                    if j-1 <= -self.J or j+1 >= self.J or np.isnan(self.get_u_matrix_elements(n=n-1, j=j-1)) or np.isnan(self.get_u_matrix_elements(n=n-1, j=j+1)) or np.isnan(self.get_u_matrix_elements(n=n-1, j=j)):
                        self.set_u_matrix_elements(n=n, j=j, u=np.nan)
                    else:
                        u = self.get_u_matrix_elements(n=n-1, j=j) - 0.5*self.a*self.lambda_*(self.get_u_matrix_elements(n=n-1, j=j+1) - self.get_u_matrix_elements(n=n-1, j=j-1)) \
                        + 0.5 * self.a*self.a*self.lambda_*self.lambda_*( self.get_u_matrix_elements(n=n-1, j=j+1)  - 2 *self.get_u_matrix_elements(n=n-1, j=j) + self.get_u_matrix_elements(n=n-1, j=j-1) )
                        self.set_u_matrix_elements(n=n, j=j, u=u)

if __name__ == "__main__":
    a_list = [1, 2, 4]
    h = 0.1
    tau = 0.08
    T = 4.0
    X = 20.0
    for scheme in [1, 2, 3, 4]:
        for a in a_list:
            pdeEngine = ConvectionEquation(a=a, h=h, tau=tau, T=T, X=X, scheme=scheme)
            pdeEngine.solve()
