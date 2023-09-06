import os
import numpy as np
from scipy import spatial
import pandas as pd
import networkx as nx
from scipy import stats

def calculate_stats(S):
    # S = np.matrix(S) #convert to matrix so that mean works over whole data for all orders
    mean = np.mean(S, axis =0)
    variance = stats.variation(S, axis= 0)
    skew = stats.skew(S, axis= 0)
    kurtosis = stats.kurtosis(S, axis= 0)

    
    return mean, variance, skew, kurtosis 

class graph_scattering(object):
    def __init__(self, 
               A, 
               scales, 
               q, 
               order,
               **args):
        self.A = A
        self.scales = scales 
        self.q = q
        self.order = order + 1
        #self.signals = args.get('signals')
        self.kernel_type = "adaptive"
        self.sigma= 2
        self.N = A.shape[0]
        
    def dirac_signals(self):
        self.signals = np.identity(self.N)

    def diffusion_operators(self):
        #self.W = compute_affinity_matrix(self.data, self.kernel_type, self.sigma, self.k)
        self.D = np.diag(np.sum(self.A, axis= 0))
        self.P = np.matmul(self.A,np.linalg.inv(self.D)) ### WD^{-1} or D^{-1}W
        pscales = np.exp2(self.scales)
        p1scales = np.exp2(np.array([s-1 for s in self.scales]))
        P_1 = np.power.outer(self.P, pscales)
        P_2 = np.power.outer(self.P, p1scales)

        Psi_j_array = P_1 - P_2
        self.Psi_js = np.transpose(Psi_j_array, (2, 0, 1))

    def zeroth_order_transform(self):
        #self.zeroth_order = np.matmul(self.D, self.signals)
        self.zeroth_order = np.matmul(np.identity(self.N), self.signals) #HS 230109

    def first_order_transform(self):
        self.first_order = np.absolute(np.tensordot(self.Psi_js,self.zeroth_order, axes=1))

    def second_order_transform(self):
        #self.second_order = np.absolute(np.matmul(self.Psi_js,self.first_order, axes=1))
        for i in range(0, self.Psi_js.shape[0]):
            c = np.absolute(np.matmul(self.Psi_js[i],self.first_order))
            if i==0:
                self.second_order = c
            else:
                self.second_order = np.vstack([self.second_order,c])

    def third_order_transform(self):
      #self.third_order = np.absolute(np.matmul(self.Psi_js,self.second_order))
      for i in range(0, self.Psi_js.shape[0]):
            c = np.absolute(np.matmul(self.Psi_js[i],self.second_order))
            if i==0:
                self.third_order = c
            else:
                self.third_order = np.vstack([self.third_order,c])

    def statistical_moments(self):
        self.mean= np.array([])
        self.variance= np.array([])
        self.skew = np.array([])
        self.kurtosis = np.array([])
        
        global S

        for i in range(0,self.order):
            if i==0:
                S = self.zeroth_order
                mean, variance, skew, kurtosis = calculate_stats(S)
                self.mean = np.append(self.mean, mean)
                self.variance = np.append(self.variance, variance)
                self.skew = np.append(self.skew, skew)
                self.kurtosis = np.append(self.kurtosis, kurtosis)

            if i==1:
                SO = self.first_order
                for j in range(0, SO.shape[0]):
                    S = SO[j]
                    mean, variance, skew, kurtosis = calculate_stats(S)
                    self.mean = np.append(self.mean, mean)
                    self.variance = np.append(self.variance, variance)
                    self.skew = np.append(self.skew, skew)
                    self.kurtosis = np.append(self.kurtosis, kurtosis)
            if i==2:
                SO = self.second_order
                for j in range(0, SO.shape[0]):
                    S = SO[j]
                    mean, variance, skew, kurtosis = calculate_stats(S)
                    self.mean = np.append(self.mean, mean)
                    self.variance = np.append(self.variance, variance)
                    self.skew = np.append(self.skew, skew)
                    self.kurtosis = np.append(self.kurtosis, kurtosis)
            if i==3:
                SO = self.third_order
                for j in range(0, SO.shape[0]):
                    S = SO[j]
                    mean, variance, skew, kurtosis = calculate_stats(S)
                    self.mean = np.append(self.mean, mean)
                    self.variance = np.append(self.variance, variance)
                    self.skew = np.append(self.skew, skew)
                    self.kurtosis = np.append(self.kurtosis, kurtosis)

        return self.mean, self.variance, self.skew, self.kurtosis

def get_pretrained_undirected_scattering(data, args):
    uds = pd.read_csv(f'results/Undirected_Scattering/Undirected_Scattering_{args.dataset}_train_val_embedding.csv', index_col=0)
    return uds
    
def run_undirected_scattering(data, args):
    
    G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol')
    A = nx.adjacency_matrix(G).toarray()
    N = A.shape[0]
    
    GS = graph_scattering(A, scales=range(2), q = 0.1, order= 2)
    GS.dirac_signals()
    GS.diffusion_operators()
    GS.zeroth_order_transform()
    GS.first_order_transform()
    GS.second_order_transform()

    gz = GS.zeroth_order.reshape((1,GS.zeroth_order.shape[0], GS.zeroth_order.shape[1]))
    g1 = GS.first_order
    g2 = GS.second_order
    z = np.concatenate([gz, g1, g2], axis = 0)

    c = np.transpose(z, (1,0,2)) #to take moments by filter scale
    mean, variance, skew, kurtosis  = calculate_stats(c)
    stats = np.concatenate([mean, variance, skew, kurtosis])
    
    if not os.path.exists(f'results/{args.model}/'):
        os.makedirs(f'results/{args.model}')

    np.save(f'results/{args.model}/Undirected_Scattering_{args.dataset}_embedding.npy', pd.DataFrame(stats.T).fillna(0).values)
