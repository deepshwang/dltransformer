import torch.nn as nn
import pdb
import torch

class RelativeSpatialEncoding(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(RelativeSpatialEncoding, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.bias = bias
        self.encode_relative_position = self.create_linear()

    def forward(self, g):
        g.apply_edges(self.encode, etype='e_RSC_G')
        pdb.set_trace()
        g.apply_nodes(self.reduce, ntype='G')
        return g

    def encode(self, edges):
        rel_RSC_G = edges.src['G'] - edges.dst['RSC']
        rel_RSC_G_enc = self.encode_relative_position(rel_RSC_G)
        return {'e_RSC_G': rel_RSC_G_enc}

    def reduce(self, nodes):
        pdb.set_trace()
        return {'RSC_encoded': torch.cat(torch.max(nodes.mailbox['e_RSC_G']), torch.avg(nodes.mailbox['e_RSC_G']))}

    def create_linear(self):
        return nn.Sequential(nn.Linear(self.c_in, self.c_out, bias=self.bias),
                              nn.ReLU())



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        head_out = g.ndata['wV']/g.ndata['z']
        
        return head_out

class PointGraphTransformerA(nn.Module):
    def __init__(self):
        super(PointGraphTransformerA, self).__init__()
        self.fps_ratio = 32
        self.k = self.fps_ratio * 4
        return

    def forward(self, pos):
        B, N, C = pos.shape
        device= pos.device

        # [1] Sample N/fps_ratio number of farthest points as A_centroid.
        A_centroid_idx = pt_utils.farthest_point_sample(pos, int(N/self.fps_ratio))
        A_centroid = self.gather_by_idx(pos, A_centroid_idx)
        pdb.set_trace()

        # [2] Find k-Nearest-Neighbors of A_centroids
        A_k_neighbors_idx = self.faissKNN(self.k-1, pos, A_centroid)
        
        # [3] Assign a A_centroid and corresponding nearest neighbor as a cluster
        A_cluster_idx = torch.cat((torch.unsqueeze(A_centroid_idx, -1), A_k_neighbors_idx.to(device)), dim=-1).view(B, -1).int()
        A_cluster = self.gather_by_idx(pos, A_cluster_idx).view(B, -1, self.k, 3)

        print("# points assigned to cluster: ", torch.unique(A_cluster_idx).shape[0], " / ", N)
        print("Repeted # of points: ", A_centroid_idx.shape[1] * self.k - torch.unique(A_cluster_idx).shape[0], " / ", N)

        # [4] Re-sample another set of centroids that are located farthest from the previous centroids \theta wise
        # as B_centroid

        # 1) Convert the points into \theta of cylindrical coordinate
        A_cluster_theta = self.cart2theta(A_cluster)

        # 2) Retrieve the \theta wise farthest point for every cluster from the centroid
        A_cluster_theta_l1 = A_cluster_theta - np.expand_dims(A_cluster_theta[:,:,0], axis=-1)
        A_cluster_max_theta_idx = np.argmax(A_cluster_theta_l1, axis=2)
        B_centroid_idx = np.take_along_axis(A_cluster_idx.view(B, -1, self.k), np.expand_dims(A_cluster_max_theta_idx, axis=-1), axis=2)
        
        # 3)
        B_centroid = self.gather_by_idx(pos, B_centroid_idx)
        B_k_neighbors_idx = self.faissKNN(self.k-1, pos, B_centroid)

        # [3] Assign a A_centroid and corresponding nearest neighbor as a cluster
        B_cluster_idx = torch.cat((B_centroid_idx, B_k_neighbors_idx.to(device)), dim=-1).view(B, -1).int()
        B_cluster = self.gather_by_idx(pos, B_cluster_idx).view(B, -1, self.k, 3)



        return A_cluster


    def faissKNN(self, k, index, query, idx_include_query=True):
        B, N, C = index.shape
        index = np.squeeze(index.cpu().numpy().copy(order='C'))

        query = np.squeeze(query.cpu().numpy())
        
        N = query.shape[0]

        faiss_index = faiss.IndexFlatL2(index.shape[1])
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
        faiss_index.add(index)


        _, k_idx = faiss_index.search(query, k+1)
        k_idx = k_idx[:, 1:]


        return torch.tensor(k_idx).view(B, k_idx.shape[0], k_idx.shape[1])

    def cart2theta(self, input_xyz):
        device = input_xyz.device
        input_xyz = input_xyz.cpu().numpy()

        # rho = np.sqrt(input_xyz[:, :, :, 0] ** 2 + input_xyz[:, :,:, 1] ** 2)
        theta = np.arctan2(input_xyz[:, :, :, 1], input_xyz[:, :, :, 0])
        return theta



    def gather_by_idx(self, db, q_idx):
        db_flipped = torch.einsum("ijk->ikj", db).contiguous()
        db = pt_utils.gather_operation(db_flipped, q_idx)
        db = torch.einsum("ijk->ikj", db).contiguous()
        return db

