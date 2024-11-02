import numpy as np

class RegulatorModel:
    def __init__(self, N, q, m, n):
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.R = None
        self.P = None  # Terminal cost matrix
        self.N = N
        self.q = q  # output dimension
        self.m = m  # input dimension
        self.n = n  # state dimension

    def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar

        # Compute F
        F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))

        return H, F

    def propagation_model_regulator_fixed_std(self):
        S_bar = np.zeros((self.N * self.q, self.N * self.m))
        T_bar = np.zeros((self.N * self.q, self.n))
        Q_bar = np.zeros((self.N * self.q, self.N * self.q))
        R_bar = np.zeros((self.N * self.m, self.N * self.m))

        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                S_bar[(k - 1) * self.q:k * self.q, (k - j) * self.m:(k - j + 1) * self.m] = np.dot(
                    np.dot(self.C, np.linalg.matrix_power(self.A, j - 1)), self.B)

            T_bar[(k - 1) * self.q:k * self.q, :self.n] = np.dot(self.C, np.linalg.matrix_power(self.A, k))

            Q_bar[(k - 1) * self.q:k * self.q, (k - 1) * self.q:k * self.q] = self.Q
            R_bar[(k - 1) * self.m:k * self.m, (k - 1) * self.m:k * self.m] = self.R

        return S_bar, T_bar, Q_bar, R_bar

    def propagation_model_regulator_fixed_std_terminal_cost(self):
        # Include terminal cost
        N = self.N
        q = self.q
        m = self.m
        n = self.n

        # Initialize matrices with correct dimensions
        S_bar = np.zeros(((N + 1) * q, N * m))
        T_bar = np.zeros(((N + 1) * q, n))
        Q_bar = np.zeros(((N + 1) * q, (N + 1) * q))
        R_bar = np.zeros((N * m, N * m))

        # Compute S_bar, T_bar, Q_bar
        for k in range(N + 1):  # k = 0 to N
            # Compute T_bar[k]
            T_bar[k * q:(k + 1) * q, :] = self.C @ np.linalg.matrix_power(self.A, k)

            # Compute S_bar[k]
            for j in range(N):
                if k - j - 1 >= 0:
                    idx_S_row = k * q
                    idx_S_col = j * m
                    S_bar[idx_S_row:(idx_S_row + q), idx_S_col:(idx_S_col + m)] = \
                        self.C @ np.linalg.matrix_power(self.A, k - j - 1) @ self.B

            # Set Q_bar[k]
            if k < N:
                Q_bar[k * q:(k + 1) * q, k * q:(k + 1) * q] = self.Q
            else:
                # Assign terminal cost P at k = N
                Q_bar[k * q:(k + 1) * q, k * q:(k + 1) * q] = self.P

        # Set R_bar
        for k in range(N):
            R_bar[k * m:(k + 1) * m, k * m:(k + 1) * m] = self.R

        return S_bar, T_bar, Q_bar, R_bar

    def updateSystemMatrices(self, sim, cur_x, cur_u):
        num_states = self.n
        num_controls = self.m
        num_outputs = self.q
        delta_t = sim.GetTimeStep()
        v0 = cur_u[0]
        theta0 = cur_x[2]
        A = np.array([
            [1, 0, -v0 * delta_t * np.sin(theta0)],
            [0, 1, v0 * delta_t * np.cos(theta0)],
            [0, 0, 1]
        ])

        B = np.array([
            [delta_t * np.cos(theta0), 0],
            [delta_t * np.sin(theta0), 0],
            [0, delta_t]
        ])

        self.A = A
        self.B = B
        self.C = np.eye(num_outputs)

    def setCostMatrices(self, Qcoeff, Rcoeff, Pcoeff=None):

        num_states = self.n
        num_controls = self.m

        # Process Qcoeff
        if np.isscalar(Qcoeff):
            Q = Qcoeff * np.eye(num_states)
        else:
            Qcoeff = np.array(Qcoeff)
            if Qcoeff.ndim != 1 or len(Qcoeff) != num_states:
                raise ValueError(f"Qcoeff must be a scalar or a 1D array of length {num_states}")
            Q = np.diag(Qcoeff)

        # Process Rcoeff
        if np.isscalar(Rcoeff):
            R = Rcoeff * np.eye(num_controls)
        else:
            Rcoeff = np.array(Rcoeff)
            if Rcoeff.ndim != 1 or len(Rcoeff) != num_controls:
                raise ValueError(f"Rcoeff must be a scalar or a 1D array of length {num_controls}")
            R = np.diag(Rcoeff)
        
        # Process Pcoeff if provided
        if Pcoeff is not None:
            if np.isscalar(Pcoeff):
                P = Pcoeff * np.eye(num_states)
            else:
                Pcoeff = np.array(Pcoeff)
                if Pcoeff.ndim != 1 or len(Pcoeff) != num_states:
                    raise ValueError(f"Pcoeff must be a scalar or a 1D array of length {num_states}")
                P = np.diag(Pcoeff)
            self.P = P

        self.Q = Q
        self.R = R