import os
import io
import sys
import gzip
import sqlite3
from itertools import groupby

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import torch

import esm 

import treeswift

############################

def walk(rootdir):
    """
    Recursively walks thru all files contained within a
    starting directory and all its sub-directories.
    
    Parameters
    ----------
    rootdir : str
        the starting directory for the recursive walk
    
    Yields
    ------
    output : str
        the path of a file
    """
    for subdir, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(subdir, f)

def read_fasta(file):
    """
    Iteratively returns the entries from a fasta file.
    
    Parameters
    ----------
    file : str
        the fasta file
    
    Yields
    ------
    output : tuple
        a single entry in the fasta file (header, sequence)
    """
    is_header = lambda x: x.startswith('>')
    compress  = lambda x: ''.join(_.strip() for _ in x)
    reader    = iter(groupby(open(file), is_header))
    reader    = iter(groupby(open(file), is_header)) if next(reader)[0] else reader
    for key, group in reader:
        if key:
            for header in group:
                header = header[1:].strip()
        else:
            sequence = compress(group)
            if sequence != '':
                yield header, sequence

############################

def gzip_string(data):
    """
    Reversibly (de)compresses a string using gzip.
    
    Parameters
    ----------
    data : str or bytes
        the string to be compressed or the bytes to be decompressed
    
    Yields
    ------
    output : str or bytes
        the decompressed string or the compressed bytes
    """
    if type(data) == str:
        return gzip.compress(data.encode())
    if type(data) == bytes:
        if data.startswith(b'\x1f\x8b'):
            return gzip.decompress(data).decode()
    raise Exception('Invalid data')
    
def gzip_array(data):
    """
    Reversibly (de)compresses a numpy array using gzip.
    
    Parameters
    ----------
    data : np.ndarray or bytes
        the array to be compressed or the bytes to be decompressed
    
    Yields
    ------
    output : np.ndarray or bytes
        the decompressed array or the compressed bytes
    """
    if type(data) == np.ndarray:
        buffer = io.BytesIO()
        np.save(buffer, data)
        return gzip.compress(buffer.getvalue())
    if type(data) == bytes:
        if data.startswith(b'\x1f\x8b'):
            return np.load(io.BytesIO(gzip.decompress(data)))
    raise Exception('Invalid data')

def gzip_tensor(data):
    """
    Reversibly (de)compresses a pytorch tensor using gzip.
    
    Parameters
    ----------
    data : torch.Tensor or bytes
        the tensor to be compressed or the bytes to be decompressed
    
    Yields
    ------
    output : torch.Tensor or bytes
        the decompressed tensor or the compressed bytes
    """
    if type(data) == torch.Tensor:
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return gzip.compress(buffer.getvalue())
    if type(data) == bytes:
        if data.startswith(b'\x1f\x8b'):
            return torch.load(io.BytesIO(gzip.decompress(data)))
    raise Exception('Invalid data')

############################

class ESM_Model:
    """ 
    A basic class for loading and running ESM models.
    For more information, visit: https://github.com/facebookresearch/esm
    
    Example:
    >> esm = ESM_Model('esm1b_t33_650M_UR50S')
    >> sequence = 'MKQEVILVLDCGATNVRAIAVNRQGKIVARASTPNASDIAMENNTWHQWSLDAI'
    >> embedding = esm.encode(sequence, device='cuda', threads=1)
    """
    
    def __init__(self, *args):
        """
        Parameters
        ----------
        model_name : str, optional
            the name of the ESM model to load
        """
        if len(args) == 1:
            self.load(args[0])
    
    def load(self, model_name):
        """
        Load an ESM model. If a model is already loaded, replace it.
        
        Parameters
        ----------
        model_name : str
            the name of the ESM model to load
        """
        self.model_name = model_name
        self.model, alphabet = eval(f'esm.pretrained.{self.model_name}()')
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        self.embed_dim = self.model._modules['layers'][0].embed_dim
        self.layers = sum(1 for i in self.model._modules['layers'])
        
    def encode(self, sequence, device='cuda', threads=1):
        """
        Generates embeddings using a loaded model.
        
        Parameters
        ----------
        sequence : str
            protein sequence
        device : str, optional
            torch device for generating embeddings.
            examples: "cpu", "cuda", "cuda:0", "cuda:1"
        threads : int
            if using cpu, try running with multiple threads
            
        Returns
        -------
        embedding : torch.tensor
            the protein sequence embedding of size
        """
        try:
            torch.cuda.empty_cache()
            torch.set_num_threads(threads)

            batch_labels, batch_strs, batch_tokens = self.batch_converter([['',sequence]])
            batch_tokens = batch_tokens.to(device)
            self.model = self.model.to(device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.layers], return_contacts=False)
                results = results["representations"][self.layers].to('cpu')[0]
            return results
        except:
            if device != 'cpu':
                return self.encode(sequence, device='cpu', threads=threads)
            else:
                return

def validate_fasta(fasta_file, max_length=1022):
    """
    Check if a given fasta file may cause potential problems in the
    embedding tree pipeline.
    
    This function checks the following criteria:
    (1) All sequence accessions must be unique. The sequence accession is the
        first "word" (all characters up until the first space) on each line
        that starts with ">".
    (2) Sequence headers cannot contain parentheses, quotes, colonrs, or
        semicolons. The header refers to all lines that start with ">" which
        includes the accession and everything that comes after it. This is
        enforced to prevent clashes with the newick tree format.
    (3) If you are using the ESM-1b language model, sequences cannot be longer
        than 1022 residues. This may be different for other language models.
    
    Parameters
    ----------
    fasta_file : str
        path to a fasta file
    max_length : int, optional
        maximum allowed length for a single sequence.
        
    Returns
    -------
    valid : bool
        whether or not the fasta file passes the criteria
    """
    valid = True
    illegal_chars = '()"' + "':;" # parenthesis, quotes, colon, semicolon
    accessions, sequences = set(), []
    for n, (h, s) in enumerate(read_fasta(fasta_file)):
        a = h.split(' ')[0]
        if a in accessions:
            valid = False
            sys.stderr.write(f'validate_fasta : "{a}" is a non-unique accession \n')
        else:
            accessions.add(a)
        l = len(s)
        if l > max_length:
            valid = False
            sys.stderr.write(f'validate_fasta : "{a}" has sequence length {l}, max is {max_length})\n')
        if any(i in illegal_chars for i in h):
            valid = False
            sys.stderr.write(f'validate_fasta : sequence header "{h}" has illegal character from set: {illegal_chars}\n')
    sys.stderr.write(f'validate_fasta : found {1+n} sequences in "{fasta_file}"\n')
    sys.stderr.write(f'validate_fasta : {"passed" if valid else "failed"}\n')
    return valid

def validate_database(db_file):

    """
    Ensures that an existing database does not exist.
    If it does, raise an error.
    
    Parameters
    ----------
    db_file : str
        path to a SQLite database
    """
    if os.path.exists(db_file):
        if os.path.getsize(db_file) == 0:
            os.remove(db_file)
            return
        else:
            raise FileExistsError(f'validate_database : existing dababase found at "{db_file}", delete or rename existing file\n')
    return

############################

class Database:
    """
    A basic class for writing and reading SQLite databases. Used for storing
    fasta headers, sequences, and embedding vectors. If no table name is provided
    default to the name "dataset".
    """
    def __init__(self, db_file):
        self.db_file    = db_file
        self.con        = sqlite3.connect(db_file)
    
    def close(self):
        self.con.close()
        
    def execute_iterate(self, *args):
        cursor = self.con.cursor()
        cursor.execute(*args)
        while True:
            ans = cursor.fetchone()
            if ans != None:
                yield ans
            else:
                return 
        
    def version(self):
        return next(self.execute_iterate("select sqlite_version();"))[0]
    
    def get_tables(self):
        return [i[0] for i in self.execute_iterate('SELECT name FROM sqlite_master WHERE type="table";')]
    
    def get_columns(self, table='dataset'):
        return [i[1] for i in self.execute_iterate(f"PRAGMA table_info({table});")]

    def get_max_rowid(self, table='dataset'):
        return next(self.execute_iterate(f'SELECT max(rowid) FROM {table};'))[0]
    
    def create_table(self, table='dataset', columns=[]):
        cursor = self.con.cursor()
        if len(columns)==0:
            cursor.execute(f'CREATE TABLE "{table}" (rowid INTEGER PRIMARY KEY) WITHOUT ROWID;')
        else:
            existing = set(self.get_columns(table=table))
            query = f'CREATE TABLE {table} ('
            for column_name, column_type in columns:
                column_type = column_type.upper()
                assert column_type in {'TEXT', 'INT', 'REAL', 'BLOB', 'NULL'}
                assert column_name not in existing
                query += f' "{column_name}" {column_type},'
            column_name, column_type = zip(*columns)
            assert len(column_name), len(set(column_name))
            query = query[:-1]+' );'
            cursor.execute(query)
        self.con.commit()
    
    def add_columns(self, columns, table='dataset'):
        """
        add new columns to the database
        
        Parameters
        ----------
            columns : list of tuple of str
                each columns is specified by a tuple (name, type)
                allowed SQLite types are TEXT, INT, REAL, BLOB, NULL
            table : str, optional
                name of the SQLite database table
        """
        cursor = self.con.cursor()
        existing = set(self.get_columns(table=table))
        for column_name, column_type in columns:
            column_type = column_type.upper()
            assert column_type in {'TEXT', 'INT', 'REAL', 'BLOB', 'NULL'}
            assert column_name not in existing    
        column_name, column_type = zip(*columns)
        assert len(column_name), len(set(column_name))
        
        cursor = self.con.cursor()
        for column_name, column_type in columns:
            cursor.execute(f'ALTER TABLE {table} ADD "{column_name}" "{column_type}";')
        self.con.commit()
    
    def add_rows(self, columns, iterable, table='dataset'):
        """
        add new rows to the database
        
        Parameters
        ----------
            columns : list of str
                list of existing column names in the database
            iterable : iterable of list
                an iterable containing the data to insert to the database,
                the data in each list should correspond to the columns names
            table : str, optional
                name of the SQLite database table
        """
        # iterable contains list of tuples : (*fields)
        existing = self.get_columns(table=table)
        assert all(i in existing for i in columns)
        cursor = self.con.cursor()
        _expr1 = ','.join(f'"{i}"' for i in columns)
        _expr2 = ','.join('?' for i in columns)
        query  = f"INSERT INTO {table} ({_expr1}) VALUES ({_expr2});"
        for n, fields in enumerate(iterable):
            cursor.execute(query, fields)
            sys.stderr.write(f'ADDING ENTRIES : {1+n}\r')
        sys.stderr.write('\n')
        self.con.commit()

    def retrieve(self, table='dataset', columns=[], rowids=[]):
        """
        retrieve data
        
        Parameters
        ----------
            table : str, optional
                name of the SQLite database table
            columns : list of str, optional
                list of existing column names in the database, if nothing
                is provided, retrieves all columns including rowid
            rowids : list of int, optional
                list containing the rowids to retrieve, if nothing is provided
                retrives all rows
        
        Yields
        ------
            entry : list of dict
                requested columns as a dictionary, column name as key
        """
        existing = self.get_columns(table=table)
        columns = existing if len(columns)==0 else columns
        assert all(i in existing for i in columns)
        columns = ['rowid'] + columns
        
        _expr1 = ','.join(f'"{i}"' for i in columns)
        query  = f'SELECT {_expr1} FROM {table}'
        
        asdict = lambda x: dict(zip(columns,x))
        if len(rowids) != 0:
            for rowid in map(int,rowids):
                yield asdict(next(self.execute_iterate(query+' WHERE ROWID=?;', [rowid])))
        else:
            for entry in self.execute_iterate(query):
                yield asdict(entry)
    
    def update_rows(self, columns, iterable, table='dataset'):
        """
        Update values on existing rows.
        
        Parameters
        ----------
            columns : list of str, optional
                list of existing column names in the database, if nothing
                is provided, retrieves all columns including rowid
            iterable : iterable of tuple
                an iterable containing tuples (int, list) containing the rowid
                and list of data to insert to the database, the data in each 
                list should correspond to the columns names
            table : str, optional
                name of the SQLite database table
        
        """
        # iterable contains list of tuples : (rowid, [*fields])
        existing = self.get_columns(table=table)
        assert all(i in existing for i in columns)
        cursor = self.con.cursor()
        
        _expr1 = ','.join(f'"{i}"' for i in columns)
        _expr2 = ','.join('?' for i in columns)
        #query  = f"INSERT INTO {table} ({_expr1}) VALUES ({_expr2})"
        
        for n, (rowid, fields) in enumerate(iterable):
            cursor.execute(f"UPDATE {table} SET ({_expr1}) = ({_expr2}) WHERE ROWID={rowid}", fields)
            sys.stderr.write(f'UPDATING ENTRIES : {1+n}\r')
        sys.stderr.write('\n')
        self.con.commit()

############################

class ESM_Representations:
    """
    Functions for deriving fixed-size embeddings from full-size embeddings.
    I put them inside of a class to keep them organized in one place.
    
    Compatible with ESM-1b embeddings which has residue tokens flanked by the two
    special tokens: the beginning-of-sequence token and end-of-sequence token.
    
    Reduces an full-size embedding of size (t, e) to a fixed-size embedding 
    of size (e), where (t) is the tokens and (e) is the embedding dimension.
    
    Example:
    >> esm = ESM_Model('esm1b_t33_650M_UR50S')
    >> sequence = 'MKQEVILVLDCGATNVRAIAVNRQGKIVARASTPNASDIAMENNTWHQWSLDAI'
    >> embedding = esm.encode(sequence, device='cuda', threads=1)
    >> 
    >> ESM_Representations.beginning_of_sequence(embedding)
    """
    @staticmethod
    def beginning_of_sequence(embedding):
        return embedding[0]
    
    @staticmethod
    def end_of_sequence(embedding):
        return embedding[-1]

    @staticmethod
    def mean_special_tokens(embedding):
        return embedding[[0, -1]].mean(0)
    
    @staticmethod
    def mean_residue_tokens(embedding):
        return embedding[1:-1].mean(0)

class Metrics:
    """
    Functions for calculating distance matrices. I put them inside of a class to keep
    them organized in one place.
    
    Given two arrays of size (x, e) and (y, e), calculates a distance matrix
    of size (x, y).
    
    Example:
    >> A = np.random.random((100,1028))
    >> B = np.random.random((200,1028))
    >>
    >> Metrics.ts_ss(A, B)
    """
    @staticmethod
    def cosine(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> cdist(x1, x2, metric='cosine')
        """
        return cdist(x1, x2, metric='cosine')
    
    @staticmethod
    def euclidean(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> cdist(x1, x2, metric='euclidean')
        """
        return cdist(x1, x2, metric='euclidean')
    
    @staticmethod
    def manhattan(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> cdist(x1, x2, metric='cityblock')
        """
        return cdist(x1, x2, metric='cityblock')

    @staticmethod
    def jensenshannon(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> from scipy.special import softmax
        >> cdist(softmax(x1), softmax(x2), metric='jensenshannon')
        """
        x1 = softmax(x1, axis=1) # used to remove negative values
        x2 = softmax(x2, axis=1)
        return cdist(x1, x2, metric='jensenshannon')
    
    @staticmethod
    def ts_ss(x1, x2):
        """
        Stands for triangle area similarity (TS) and sector area similarity (SS)
        For more information: https://github.com/taki0112/Vector_Similarity
        """
        x1_norm = np.linalg.norm(x1, axis=-1)[:,np.newaxis]
        x2_norm = np.linalg.norm(x2, axis=-1)[:,np.newaxis]
        x_dot = x1_norm @ x2_norm.T

        ### cosine similarity
        cosine_sim = 1 - cdist(x1, x2, metric='cosine')
        cosine_sim[cosine_sim != cosine_sim] = 0
        cosine_sim = np.clip(cosine_sim, -1, 1, out=cosine_sim)

        ### euclidean_distance
        euclidean_dist = cdist(x1, x2, metric='euclidean')

        ### triangle_area_similarity
        theta = np.arccos(cosine_sim) + np.radians(10)
        triangle_similarity = (x_dot * np.abs(np.sin(theta))) / 2

        ### sectors area similarity
        magnitude_diff = np.abs(x1_norm - x2_norm.T)
        ed_plus_md = euclidean_dist + magnitude_diff
        sector_similarity =  ed_plus_md * ed_plus_md * theta * np.pi / 360

        ### hybridize
        similarity = triangle_similarity * sector_similarity
        return similarity

def silhouette(distmat, labels, ignore=['']):
    """
    Calculates the silhouette score of a distance matrix. 
    
    Parameters
    ----------
    distmat : np.ndarray
        a square, symmetrical distance matrix of size (n, n)
    labels : list of str
        list of size (n) containing labels corresponding to the distance matrix
    ignore : list of str, optional
    list of labels to ignore. data points with these labels will not be
    considered when calculating silhouette score. We automatically ignore any
    label that occurs only once. By default we also ignore blank labels.

    Returns
    -------
    output : float
        the calculated silhouette score, based on the provided labels
    """
    exclude = set(k for k, g in groupby(sorted(labels)) if sum(1 for i in g)<2)
    exclude.update(set(ignore))
    mask = [i not in exclude for i in labels]
    return silhouette_score(distmat[mask,:][:,mask], labels[mask], metric='precomputed')

############################

def upgma(distmat, names):
    """
    Builds a tree from a distance matrix using the UPGMA algorithm.
    
    Parameters
    ----------
    distmat : np.ndarray
        a square, symmetrical distance matrix of size (n, n)
    names : list of str
        list of size (n) containing names corresponding to the distance matrix
    
    Returns
    -------
    tree : str
        a newick-formatted tree
    """
    D = squareform(distmat, checks=True)
    Z = hierarchy.linkage(D, method='average', optimal_ordering=False)
    def to_newick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f'{leaf_names[node.id]}:{parentdist-node.dist:.6f}{newick}'
        else:
            if len(newick) > 0:
                newick = f'):{parentdist - node.dist:.6f}{newick}'
            else:
                newick = ');'
            newick = to_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = to_newick(node.get_right(), f',{newick}', node.dist, leaf_names)
            newick = f'({newick}'
            return newick
    tree = hierarchy.to_tree(Z, False)
    return to_newick(tree, "", tree.dist, names)

def neighbor_joining(distmat, names):
    """
    Builds a tree from a distance matrix using the NJ algorithm using the
    original algorithm published by Saitou and Nei.
    
    Parameters
    ----------
    distmat : np.ndarray
        a square, symmetrical distance matrix of size (n, n)
    names : list of str
        list of size (n) containing names corresponding to the distance matrix
    
    Returns
    -------
    tree : str
        a newick-formatted tree
    """
    def join_ndx(D, n):
        # calculate the Q matrix and find the pair to join
        Q  = np.zeros((n, n))
        Q += D.sum(1)
        Q += Q.T
        Q *= -1.
        Q += (n - 2.) * D
        np.fill_diagonal(Q, 1.) # prevent from choosing the diagonal
        return np.unravel_index(Q.argmin(), Q.shape)
    
    def branch_lengths(D, n, i, j):
        i_to_j = float(D[i, j])
        i_to_u = float((.5 * i_to_j) + ((D[i].sum() - D[j].sum()) / (2. * (n - 2.))))
        if i_to_u < 0.:
            i_to_u = 0.
        j_to_u = i_to_j - i_to_u
        if j_to_u < 0.:
            j_to_u = 0.
        return i_to_u, j_to_u
    
    def update_distance(D, n1, mask, i, j):
        D1 = np.zeros((n1, n1))
        D1[0, 1:] = 0.5 * (D[i,mask] + D[j,mask] - D[i,j])
        D1[0, 1:][D1[0, 1:] < 0] = 0
        D1[1:, 0] = D1[0, 1:]
        D1[1:, 1:] = D[:,mask][mask]
        return D1
        
    t = names
    D = distmat.copy()
    np.fill_diagonal(D, 0.)
    
    while True:
        n = D.shape[0]
        if n == 3:
            break
        ndx1, ndx2 = join_ndx(D, n)
        len1, len2 = branch_lengths(D, n, ndx1, ndx2)
        mask  = np.full(n, True, dtype=bool)
        mask[[ndx1, ndx2]] = False
        t = [f"({t[ndx1]}:{len1:.6f},{t[ndx2]}:{len2:.6f})"] + [i for b, i in zip(mask, t) if b]
        D = update_distance(D, n-1, mask, ndx1, ndx2)
        
    len1, len2 = branch_lengths(D, n, 1, 2)
    len0 = 0.5 * (D[1,0] + D[2,0] - D[1,2])
    if len0 < 0:
        len0 = 0
    newick = f'({t[1]}:{len1:.6f},{t[0]}:{len0:.6f},{t[2]}:{len2:.6f});'
    return newick
     
def cophenetic_distmat(tree, names):
    """
    Calculates the all-versus-all distance matrix of a tree based on the
    cophenetic distances.
    
    Parameters
    ----------
    tree : str
        a newick-formatted tree
    names : list of str
        a list of names contained within the tree. the order of the names provided
        in the list will be used to determine the order of the output.
    
    Returns
    -------
    cophmat : np.ndarray
        a square, symmetrical distance matrix
    """
    tree = treeswift.read_tree_newick(tree) if type(tree) is str else tree
    cophdic = tree.distance_matrix()
    node2name = {i:i.get_label() for i in cophdic.keys()}
    unique_names = set(node2name.values())
    assert len(node2name)==len(set(node2name))
    cophdic = {node2name[k1]:{node2name[k2]:cophdic[k1][k2] for k2 in cophdic[k1]} for k1 in cophdic.keys()}
    assert all(i in unique_names for i in names)
    cophmat = np.zeros([len(names)]*2)
    for ni, i in enumerate(names):
        for nj, j in enumerate(names[:ni]):
            cophmat[ni][nj] = cophmat[nj][ni] = cophdic[i][j]
    return cophmat


