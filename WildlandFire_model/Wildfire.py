from Coefficient_Matrix import CoefficientMatrix
import numpy as np
import math


class Wildfire:
    def __init__(self, Lxi: float, Leta: float, Nxi: int, Neta: int, timesteps: int, v_x: float, v_y: float,
                 cfl: float, c: float, beta: float, select_every_n_timestep: int) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nxi > 0, f"Please input sensible values for the X grid points"
        assert Neta > 0, f"Please input sensible values for the Y grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.Y = None
        self.t = None

        self.NumConservedVar = 2

        # Private variables
        self.__Lxi = Lxi
        self.__Leta = Leta
        self.__Nxi = Nxi
        self.__Neta = Neta
        self.__timesteps = timesteps
        self.__cfl = cfl

        # Order of accuracy for the derivative matrices of the first and second order
        self.__firstderivativeOrder = "5thOrder"

        # Dimensional constants used in the model
        self.__thermaldiffusivity = 0.2136
        self.__preexponentialfactor = 0.1625
        self.__windspeed_x = np.ones(self.__timesteps) * v_x
        self.__windspeed_y = np.zeros(self.__timesteps) * v_y
        self.__temperaturerisepersecond = 187.93
        self.__scaledheattransfercoefficient = 4.8372e-5
        self.__beta = beta
        self.__Tambient = 300
        self.__speedofsoundsquare = c

        # Sparse matrices of the first and second order
        self.Mat = None

        # Concatenated data structure for the conserved variables T and S for all time steps
        self.qs = []

        # For sPOD afterwards, select every nth timestep and store
        self.select_every_n_timestep = select_every_n_timestep

    def solver(self):
        ########################################################
        # INITIAL CONDITIONS
        dx, dy, dt, q = self.__InitialConditions()

        # SOLVER
        self.__TimeIntegration(dx, dy, dt, q)  # The results of the simulation are stored in 'self.qs'

        # SOLUTION RESHAPING FOR MODEL ORDER REDUCTION
        self.qs = np.transpose(np.squeeze(self.qs).reshape((self.__timesteps // self.select_every_n_timestep, -1),
                                                           order="F" if self.__Neta != 1 else "C"))
        self.t = self.t[::self.select_every_n_timestep]
        ########################################################

    # Private function for this class
    def __InitialConditions(self):
        self.X = np.arange(1, self.__Nxi + 1) * self.__Lxi / self.__Nxi
        dx = self.X[1] - self.X[0]

        if self.__Neta == 1:
            self.Y = 0
            dy = 0
        else:
            self.Y = np.arange(1, self.__Neta + 1) * self.__Leta / self.__Neta
            dy = self.Y[1] - self.Y[0]

        dt = (np.sqrt(dx ** 2 + dy ** 2)) * self.__cfl / math.sqrt(self.__speedofsoundsquare)
        self.t = dt * np.arange(self.__timesteps)

        self.X_2D, self.Y_2D = np.meshgrid(self.X, self.Y)
        self.X_2D = np.transpose(self.X_2D)
        self.Y_2D = np.transpose(self.Y_2D)

        # Select the correct Initial conditions
        if self.__Neta == 1:
            T = 1200 * np.exp(-((self.X_2D - self.__Lxi / 2) ** 2) / 200)
            S = np.ones_like(T)
        else:
            T = 1200 * np.exp(-(((self.X_2D - self.__Lxi / 2) ** 2) / 200 + ((self.Y_2D - self.__Leta / 2) ** 2) / 200))
            S = np.ones_like(T)

        # Arrange the values of T and S in 'q'
        NN = self.__Nxi * self.__Neta
        T = np.reshape(T, newshape=NN, order="F")
        S = np.reshape(S, newshape=NN, order="F")
        q = np.array([T, S]).T

        print(self.t[-1])

        return dx, dy, dt, q

    # Private function for this class
    def __TimeIntegration(self, dx, dy, dt, q):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat = CoefficientMatrix(orderDerivative=self.__firstderivativeOrder, Nxi=self.__Nxi,
                                     Neta=self.__Neta, periodicity='Periodic', dx=dx, dy=dy)

        # Time loop
        for n in range(self.__timesteps):
            # Main Runge-Kutta 4 solver step
            q = self.__RK4(q, dt, 0, t_step=n)

            # Store the values in the 'self.qs' for all the time steps successively
            T = np.reshape(q[:, 0], newshape=[self.__Nxi, self.__Neta], order="F")
            S = np.reshape(q[:, 1], newshape=[self.__Nxi, self.__Neta], order="F")

            if n % self.select_every_n_timestep == 0:
                self.qs.append([T, S])

            print('Time step: ', n)

        pass

    # Private function for this class
    def __RHS(self, q, t, t_step=0):
        T = q[:, 0]
        S = q[:, 1]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = (T > 0).astype(int)
        # This parameter is for preventing division by 0
        epsilon = 0.000001

        # Coefficients for the terms in the equation
        Coeff_diff = self.__thermaldiffusivity
        Coeff_conv_x = self.__windspeed_x[t_step]
        Coeff_conv_y = self.__windspeed_y[t_step]

        Coeff_source = self.__temperaturerisepersecond * self.__scaledheattransfercoefficient
        Coeff_arrhenius = self.__temperaturerisepersecond
        Coeff_massfrac = self.__preexponentialfactor

        DT = Coeff_conv_x * self.Mat.Grad_Xi_kron + Coeff_conv_y * self.Mat.Grad_Eta_kron
        Tdot = Coeff_diff * self.Mat.Laplace.dot(T) - DT.dot(T) - Coeff_source * T + Coeff_arrhenius * arrhenius_activate * S * np.exp(-self.__beta / (T + epsilon))
        Sdot = - Coeff_massfrac * arrhenius_activate * S * np.exp(-self.__beta / (T + epsilon))

        qdot = np.array([Tdot, Sdot]).T

        return qdot

    # Private function for this class
    def __RK4(self, u0, dt, t, t_step=0):
        k1 = self.__RHS(u0, t, t_step)
        k2 = self.__RHS(u0 + dt / 2 * k1, t + dt / 2, t_step)
        k3 = self.__RHS(u0 + dt / 2 * k2, t + dt / 2, t_step)
        k4 = self.__RHS(u0 + dt * k3, t + dt, t_step)

        u1 = u0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u1



