import numpy as np
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, N, method):
        self.N = N

        if method == 1:
            self.x_list = []
            self.x_list.append(0)
            stepSize = 1 / self.N
            for i in range(N):
                self.x_list.append(stepSize+self.x_list[-1])

        self.A_matrix = np.zeros((self.N, self.N))
        self.F_vector = np.zeros(self.N)
        self.U_vector = np.zeros(self.N)

        self.realU_vector = np.sin(self.x_list)
        self.beta = 2*np.cos(1) + np.sin(1)
        self.set_A()
        self.set_F()
        self.solve_engine()

    def a(self, i, j):
        sum = 0
        for k in range(1, self.N+1):
            # u_{k-1}: a
            # v_{k-1}: b
            # u_k: c
            # v_k: d
            a = b = c = d = 0
            if k-1 == i:
                a = 1
            if k == i:
                c = 1
            if k-1 == j:
                b = 1
            if k == j:
                d = 1
            # print(i,j,a,b,c,d)

            if a*b != 0:
                sum += ( 2/3*self.x_list[k]**3 - 2/3*self.x_list[k-1]**3 + self.x_list[k] * self.x_list[k-1]**2 - self.x_list[k]**2 * self.x_list[k-1] + self.x_list[k] - self.x_list[k-1]   ) / ( (self.x_list[k] - self.x_list[k-1]  )**2  )

            if a*d != 0:
                sum += ( -1 * self.x_list[k]**3/6 + self.x_list[k-1]**3 / 6 -self.x_list[k] +self.x_list[k-1] - 0.5*self.x_list[k-1]*self.x_list[k]**2 + 0.5 * self.x_list[k]*self.x_list[k-1]**2  )  / ( (self.x_list[k] - self.x_list[k-1]  )**2  )

            if c*b != 0:
                sum += ( -1 * self.x_list[k]**3/6 + self.x_list[k-1]**3 / 6 -self.x_list[k] +self.x_list[k-1] - 0.5*self.x_list[k-1]*self.x_list[k]**2 + 0.5 * self.x_list[k]*self.x_list[k-1]**2  )  / ( (self.x_list[k] - self.x_list[k-1]  )**2  )

            if c*d != 0:
                if k == self.N:
                    sum += ( 2/3* self.x_list[k]**3 - 2/3*self.x_list[k-1]**3 +self.x_list[k] - self.x_list[k-1] + self.x_list[k-1]**2 * self.x_list[k] - self.x_list[k-1] * self.x_list[k]**2  )  / ( (self.x_list[k] - self.x_list[k-1]  )**2  ) + 1
                else:
                    sum += ( 2/3* self.x_list[k]**3 - 2/3*self.x_list[k-1]**3 +self.x_list[k] - self.x_list[k-1] + self.x_list[k-1]**2 * self.x_list[k] - self.x_list[k-1] * self.x_list[k]**2  )  / ( (self.x_list[k] - self.x_list[k-1]  )**2  )

        return sum

    def set_A(self):
        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                if abs(i-j) >= 2:
                    pass
                else:
                    self.A_matrix[i-1][j-1] = self.a(i, j)

    def auxiliary_a(self, up_x, down_x):
        # a = -2 \cos x - x^2 \cos x
        return -2*np.cos(up_x)-up_x*up_x*np.cos(up_x) - (-2*np.cos(down_x)-down_x*down_x*np.cos(down_x))

    def auxiliary_b(self, up_x, down_x):
        # b = x^2 \sin x - x^3 \cos x
        return up_x*up_x*np.sin(up_x)-up_x**3*np.cos(up_x) - (down_x*down_x*np.sin(down_x)-down_x**3*np.cos(down_x))

    def set_F(self):
        for i in range(1, self.N+1):
            # a = -2 \cos x - x^2 \cos x
            # b = x^2 \sin x - x^3 \cos x
            sum = 0
            for k in range(1, self.N+1):
                c1 = c2 = 0
                if k-1 == i:
                    c1 = 1  # v_{k-1}
                if k == i:
                    c2 = 1  # v_k
                if c1 == 1:
                    sum += self.x_list[k]/(self.x_list[k]-self.x_list[k-1])*self.auxiliary_a(up_x=self.x_list[k], down_x=self.x_list[k-1]) \
                    - 1/(self.x_list[k]-self.x_list[k-1]) * self.auxiliary_b(up_x=self.x_list[k], down_x=self.x_list[k-1])
                if c2 == 1:
                    if k == self.N:
                        sum += 1*self.beta -self.x_list[k-1]/(self.x_list[k]-self.x_list[k-1])*self.auxiliary_a(up_x=self.x_list[k], down_x=self.x_list[k-1]) \
                        + 1 / (self.x_list[k]-self.x_list[k-1]) * self.auxiliary_b(up_x=self.x_list[k], down_x=self.x_list[k-1])
                    else:
                        sum += -1 * self.x_list[k-1]/(self.x_list[k]-self.x_list[k-1])*self.auxiliary_a(up_x=self.x_list[k], down_x=self.x_list[k-1]) \
                        + 1 / (self.x_list[k]-self.x_list[k-1]) * self.auxiliary_b(up_x=self.x_list[k], down_x=self.x_list[k-1])
            self.F_vector[i-1] = sum

    def solve_engine(self):
        self.U_vector = np.dot(np.linalg.inv(self.A_matrix), self.F_vector)
        fig = plt.figure()
        plt.plot(self.x_list[1:self.N+1], self.U_vector, 'bx')
        plt.plot(self.x_list[0:self.N+1], self.realU_vector)
        plt.xlabel("x")
        #
        # for i in range(self.N):
        #     line_string = ""
        #     for j in range(self.N):
        #         line_string += "   "+str(self.A_matrix[i][j])+"   "
        #     print(line_string)
        # print(self.F_vector)

        plt.show()

if __name__ == '__main__':
    N = 10
    method = 1
    s = Solver(N=N, method=method)