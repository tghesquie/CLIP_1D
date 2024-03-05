import numpy as np
import scipy.sparse
from scipy.optimize import approx_fprime

class BulkDamage:
    def __init__(self, lc, Dm, len_mat):
        self.lc = lc
        self.Dm = Dm
        self.len_mat = len_mat

    def get_Bulk_damage0(self, d):
        d_new = np.concatenate([np.zeros(1), d, np.zeros(1)])
        D_values = np.max((d_new - self.len_mat / self.lc) * self.Dm, axis=1)
        D_values[D_values < 0] = 0  
        return D_values

    def get_Bulk_damage(self, d, centeronly=True): 
        D_values = self.get_Bulk_damage0(d)
        D_center = (D_values[:-1] + D_values[1:]) / 2
        if centeronly:
            return D_center
        D_overall = np.empty(2 * len(D_values) - 1)
        D_overall[0::2] = D_values
        D_overall[1::2] = D_center
        return D_values[1:-1], D_center, D_overall

    def get_dBulk_damage0_dd(self, d):
        d_new = np.concatenate([np.zeros(1), d, np.zeros(1)])
        n = len(d_new)
        D_values = (d_new - self.len_mat / self.lc) * self.Dm
        max_indexes = np.argmax(D_values, axis=1)    
        D_values = D_values[np.arange(n), max_indexes]

        dD_values_dd_nz_col = max_indexes[D_values >= 0]
        dD_values_dd = scipy.sparse.csr_matrix((self.Dm * np.ones(len(dD_values_dd_nz_col)), (np.arange(n)[D_values >= 0], dD_values_dd_nz_col)), shape=(n, n))
        return dD_values_dd[:, 1:-1]

    def get_dBulk_damage_dd(self, d):
        dD_values_dd = self.get_dBulk_damage0_dd(d)
        dD_center_dd = (dD_values_dd[:-1, :] + dD_values_dd[1:, :]) / 2
        return dD_center_dd
