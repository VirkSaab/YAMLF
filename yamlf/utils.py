import torch, os, random, time, datetime
import numpy as np
import torch.nn as nn

__all__ = ["set_seed", "calc_time_taken", "net_stats"]

# SEED
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.random.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(s)
    print('Random state fixed at', s)

def calc_time_taken(start_time):
    t = time.time()-start_time
    if t > 1.: return str(datetime.timedelta(seconds=round(t)))
    else: return f"{t} secs"

def net_stats(net):
    def find_modules(m, cond):
        if cond(m): return [m]
        return sum([find_modules(o,cond) for o in m.children()], [])
    def is_lin_layer(l):
        convs = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        fcs = (nn.Linear)
        bnorm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        embds = (nn.Embedding, nn.EmbeddingBag)
        rnns = (nn.RNN, nn.RNNBase, nn.RNNCell, nn.RNNCellBase)
        if isinstance(l, convs): return 'conv'
        elif isinstance(l, fcs): return 'fc'
        elif isinstance(l, bnorm): return 'bn'
        elif isinstance(l, embds): return 'embd'
        elif isinstance(l, rnns): return 'rnn'
        else: return None
    def count_params():
        total_params, trainable_params = 0, 0
        for ps in net.parameters():
            # total_params
            if ps.flatten().size() == torch.Size([]): total_params += 1
            else: total_params += sum(list(ps.flatten().shape))
            # trainable_params
            if ps.requires_grad:
                if ps.flatten().size() == torch.Size([]): trainable_params += 1
                else: trainable_params += sum(list(ps.flatten().shape))
        return total_params, trainable_params
    mods   = find_modules(net, lambda o: is_lin_layer(o))
    total_params, trainable_params = count_params()
    convs  = len([mod for mod in mods if 'Conv' in str(mod)])
    bnorms = len([mod for mod in mods if 'BatchNorm' in str(mod)])
    fcs    = len([mod for mod in mods if 'Linear' in str(mod)])
    embds  = len([mod for mod in mods if 'Embedding' in str(mod)])
    rnns   = len([mod for mod in mods if 'RNN' in str(mod)])
    lstms  = len([mod for mod in mods if 'LSTM' in str(mod)])
    disp   = "NETWORK STATS:\n"
    if convs  > 0: disp += f"{convs} convs\n"
    if bnorms > 0: disp += f"{bnorms} batchnorms\n"
    if fcs    > 0: disp += f"{fcs} dense\n"
    if rnns   > 0: disp += f"{rnns} RNN\n"
    if embds  > 0: disp += f"{embds} Embedding\n"
    if lstms  > 0: disp += f"{lstms} LSTM\n"
    disp += f"# parameters: {total_params/1e6:.3f}M"
    print(disp)
