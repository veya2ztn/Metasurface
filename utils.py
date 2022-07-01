import numpy as np
from mltool.fastprogress import isnotebook
from mltool.visualization import *
import os,json,time,math,shutil
import torch
import random
def makerecard(name,successQ,_dir,content=""):
    flag = "s" if successQ else "f"
    file_name = flag + "<--" + name
    if not os.path.exists(_dir):os.makedirs(_dir)
    abs_path  = os.path.join(_dir,file_name)
    datetime  = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    status    = "Success" if successQ else "Fail"
    tjson_dict={}
    tjson_dict["Time"]   = datetime
    tjson_dict["Name"]   = name
    tjson_dict["Status"] = status
    tjson_dict["Content"]= content
    with open(abs_path, 'w') as f:
        json.dump(tjson_dict, f)

class DataType:
    def __init__(self,field,shape):
        self.recard_shape = shape
        self.force_field(field)

    def _generate(self):
        if self.field == 'complex':
            if self.recard_shape[-1]==2:
                self.shape = self.recard_shape[:-1]
                self.data_shape = self.recard_shape
            else:
                self.shape = self.recard_shape
                self.data_shape = tuple(list(self.recard_shape)+[2])
        else:
            self.shape = self.recard_shape
            self.data_shape = self.recard_shape
    def __repr__(self):
        _str = f"field:{self.field} internal data shape:{self.shape}"
        return _str

    def force_field(self,field):
        self.field=field
        self._generate()

    def sample(self):
        return torch.randn([2]+list(self.data_shape))

    def reset(self,field,shape):
        self.recard_shape = shape
        self.force_field(field)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def get_GPU_para(project_config):
    GPU_MEMORY_CONFIG_FILE="projects/GPU_Memory_Config.json"
    if os.path.exists(GPU_MEMORY_CONFIG_FILE):
        with open(GPU_MEMORY_CONFIG_FILE,'r') as f:memory_record = json.load(f)
    else:
        memory_record = {}
    MODEL_TYPE       = project_config.project_name.split('.')[0]
    if MODEL_TYPE in memory_record:
        memory_k,memory_b=memory_record[MODEL_TYPE]
        print("detecte memory file, use detected parameters k={},b={}".format(memory_k,memory_b))
    else:
        print("no auto parameters detected, use configuration parameters k={},b={}".format(memory_k,memory_b))
    if memory_k and memory_b:
        free_memory = query_gpu()[0]['memory.free']*0.9
        BATCH_SIZE   = (free_memory - memory_b)/memory_k
        #BATCH_SIZE   = int(np.round(BATCH_SIZE/100) *100)
        BATCH_SIZE  = int(BATCH_SIZE)
        print("==== use automatice batch size, it will occupy 90% free GPU memory ===")
        print("==== the batch size now set {}".format(BATCH_SIZE))
    else:
        print("please set memory configuration if you use auto mode.")
        print("you can run `python mutli_task_train test` to generate auto memory parametersaa")
        raise
    return BATCH_SIZE

def record_data(datalist,file_name):
    with open(file_name,'a') as log_file:
        ll=["{:.4f}".format(data) for data in datalist]
        printedstring=' '.join(ll)+'\n'
        _=log_file.write(printedstring)

def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    n_aux_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        if "aux" in name:
            n_aux_params += n_elem
        n_total_params += n_elem
    return n_total_params, n_total_params - n_aux_params

def mymovefile(srcfile,dstfile):
    if (not os.path.exists(srcfile)) or  (not os.path.isfile(srcfile)):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile,dstfile)

def show_a_demo(y_test_r,y_test_c,y_test_p,_dir=None,filename=None):
    fig=plt.figure()
    x = np.arange(len(y_test_c))
    if y_test_r is None:
        plt.plot(x,y_test_c,'g',x,y_test_p,'b')
    else:
        plt.plot(x,y_test_r,'r',x,y_test_c,'g',x,y_test_p,'b')
    if _dir is None:
        return
    if not os.path.exists(_dir):os.makedirs(_dir)
    path = os.path.join(_dir,filename)
    plt.savefig(path)
    plt.close()

def occumpy_mem():
    block_mem = query_gpu()[0]['memory.free']-100
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x

def parse(line,qargs):
    '''
    https://zhuanlan.zhihu.com/p/28690706
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    #numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    numberic_args = ['memory.used','memory.free','memory.total']
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu(qargs=['memory.used','memory.free','memory.total']):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    #qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line,qargs) for line in results]
def curver_filte_smoothpeak(tensor0,low_resp=0.1,smooth=0.01):
    if isinstance(tensor0,torch.Tensor):tensor0=tensor0.numpy()
    maxten    = np.max(tensor0,1)
    maxfilter = np.where(maxten>0.1)[0]
    #tensor0   = tensor0[maxfilter]
    tensor=np.pad(tensor0,((0,0),(1,1)),"edge")
    grad_r = tensor[...,2:]-tensor[...,1:-1]
    grad_l = tensor[...,1:-1]-tensor[...,:-2]
    out = np.abs((grad_l - grad_r))
    maxout=np.max(out,1)
    smoothfilter=np.where(maxout<0.01)[0]
    filted_index=np.intersect1d(maxfilter,smoothfilter)
    return filted_index

def random_v_flip(data):
    batch,c,w,h = data.shape
    index=torch.randint(2,(batch,))==1
    data[index]=data[index].flip(2)
def random_h_flip(data):
    batch,c,w,h = data.shape
    index=torch.randint(2,(batch,))==1
    data[index]=data[index].flip(3)


def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r
normer = np.linalg.norm
def _c(ca, i, j, p, q):
    if ca[i, j] > -1:return ca[i, j]
    elif i == 0 and j == 0:ca[i, j] = normer(p[i]-q[j])
    elif i > 0 and j == 0:ca[i, j]  = max(_c(ca, i-1, 0, p, q), normer(p[i]-q[j]))
    elif i == 0 and j > 0:ca[i, j]  = max(_c(ca, 0, j-1, p, q), normer(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            normer(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]

def frdist(p, q):
    """
        Computes the discrete Fréchet distance between
        two curves. The Fréchet distance between two curves in a
        metric space is a measure of the similarity between the curves.
        The discrete Fréchet distance may be used for approximately computing
        the Fréchet distance between two arbitrary curves,
        as an alternative to using the exact Fréchet distance between a polygonal
        approximation of the curves or an approximation of this value.
        This is a Python 3.* implementation of the algorithm produced
        in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance.
        Tech. Report CD-TR 94/64, Information Systems Department, Technical
        University of Vienna.
        http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
        Function dF(P, Q): real;
            input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
            return: δdF (P, Q)
            ca : array [1..p, 1..q] of real;
            function c(i, j): real;
                begin
                    if ca(i, j) > −1 then return ca(i, j)
                    elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                    elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
                    elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj) }
                    elsif i > 1 and j > 1 then ca(i, j) :=
                    max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
                    else ca(i, j) = ∞
                    return ca(i, j);
                end; /* function c */
            begin
                for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
                return c(p, q);
            end.
        Parameters
        ----------
        P : Input curve - two dimensional array of points
        Q : Input curve - two dimensional array of points
        Returns
        -------
        dist: float64
            The discrete Fréchet distance between curves `P` and `Q`.
        Examples
        --------
        >>> from frechetdist import frdist
        >>> P=[[1,1], [2,1], [2,2]]
        >>> Q=[[2,2], [0,1], [2,4]]
        >>> frdist(P,Q)
        >>> 2.0
        >>> P=[[1,1], [2,1], [2,2]]
        >>> Q=[[1,1], [2,1], [2,2]]
        >>> frdist(P,Q)
        >>> 0
    """
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:raise ValueError('Input curves are empty.')
    if len_p != len_q:raise ValueError('Input curves do not have the same dimensions.')

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist

# not right!!!
def BitsToIntAFast(bits):
    m,n = bits.shape # number of columns is needed, not bits.size
    a = 2**np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code
ImageUniqueCode=BitsToIntAFast
