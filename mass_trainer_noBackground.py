import numpy as np
run = 0
np.random.seed(run)
import os, glob
import time
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage.transform import rescale
plt.rcParams["figure.figsize"] = (5,5)
from torch.utils.data import *

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-e', '--epochs', default=90, type=int, help='Number of training epochs.')
parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
parser.add_argument('-b', '--resblocks', default=3, type=int, help='Number of residual blocks.')
parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
args = parser.parse_args()

lr_init = args.lr_init
resblocks = args.resblocks
epochs = args.epochs
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

#run_logger = False
run_logger = True 
eb_scale = 25.
m0_scale = 15
mass_bins = np.arange(3600,15000+200,200)/1000. # for histogram in eval()
#n_train = 256*3040
n_all = 256*2900
n_val = 25600
n_train = n_all - n_val

decay = 'HToTauTau_m3p6To15_pT0To200_ctau0To3_eta0To1p4'

expt_name = 'EBtzo%.f_AOD_m0o%.1f_ResNet_blocks%d_seedPos_MAEloss_lr%s_epochs%d_ntrain%d_nval%d_run%d'\
            %(eb_scale, m0_scale, resblocks, str(lr_init), epochs, n_train, n_val, run)

expt_name = '%s_%s'%(decay, expt_name)
if run_logger:
    if not os.path.isdir('LOGS'):
        os.makedirs('LOGS')
    f = open('LOGS/%s.log'%(expt_name), 'w')
    #for d in ['MODELS', 'METRICS','PLOTS']:
    for d in ['MODELS', 'PLOTS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))

def logger(s):
    global f, run_logger
    print(s)
    if run_logger:
        f.write('%s\n'%str(s))

def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs().cuda()
    #loss = wgt*(pred-true).pow(2).cuda()
    return loss.mean()

def transform_y(y):
    return y/m0_scale

def inv_transform(y):
    return y*m0_scale

class ParquetDataset(Dataset):
    def __init__(self, filename, label):
        self.parquet = pq.ParquetFile(filename)
        #self.cols = None # read all columns
        #self.cols = ['Xtz_aod.list.item.list.item.list.item','m','pt','w','iphi','ieta'] 
        self.cols = ['X_jet.list.item.list.item.list.item','am','apt','iphi','ieta'] 
        self.label = label
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jet'] = np.float32(data['X_jet'][0])/eb_scale
        data['am'] = transform_y(np.float32(data['am']))
        data['apt'] = np.float32(data['apt'])
        #data['w'] = np.float32(data['w'])
        data['iphi'] = np.float32(data['iphi'])/360.
        data['ieta'] = np.float32(data['ieta'])/170.
        data['label'] = self.label
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups

logger('>> Experiment: %s'%(expt_name))

#directory = 'IMG/HToTauTau_m3p6To15_pT0To200_ctau0To3_eta0To1p4'
#decays = [
#    'HToTauTau_m3p6To15_pT0To200_ctau0To3_eta0To1p4.parquet.1'
#    ]
decay = 'HToTauTau_m3p6To15_pT0To200_ctau0To3_eta0To1p4'
decays = glob.glob('IMG/%s/*.parquet*'%decay)
#print(">> Input files:",decays)
dset_train = ConcatDataset([ParquetDataset('%s'%d, i) for i,d in enumerate(decays)])

idxs = np.random.permutation(len(dset_train))
idxs_train = idxs[:n_train]
idxs_val = idxs[n_train:]
np.savez('MODELS/%s/idxs_train+val.npz'%(expt_name), idxs_train=idxs_train, idxs_val=idxs_val)
assert len(idxs_train)+len(idxs_val) == len(idxs), '%d vs. %d'%(len(idxs_train)+len(idxs_val), len(idxs))
# Train
train_sampler = sampler.SubsetRandomSampler(idxs_train)
train_loader = DataLoader(dataset=dset_train, batch_size=256, num_workers=10, pin_memory=True, sampler=train_sampler)
# Val
val_sampler = sampler.SubsetRandomSampler(idxs_val)
val_loader = DataLoader(dataset=dset_train, batch_size=256, num_workers=10, pin_memory=True, sampler=val_sampler)
logger('>> N samples: Train: %d + Val: %d'%(len(idxs_train), len(idxs_val)))

# Test sets
dset_sg = ParquetDataset('IMG/HToTauTau_m3p6To15_pT0To200_ctau0To3_eta0To1p4/HToTauTau_m3p6To15_pT0To200_ctau0To3_eta0To1p4.parquet.1', 1)
sg_loader = DataLoader(dataset=dset_sg, batch_size=256, num_workers=10)
#dset_bg = ParquetDataset('IMG/DoublePhotonPt10To100_pythia8_ReAOD_PU2017_MINIAODSIM_wrapfix.tzfixed_m0Neg%dTo0_wgts.val.parquet'%args.neg_mass, 0)
#bg_loader = DataLoader(dataset=dset_bg, batch_size=256, num_workers=10)
#logger('>> N test samples: sg: %d + bg: %d'%(len(dset_sg), len(dset_bg)))
logger('>> N test samples: sg: %d'%(len(dset_sg)))

import torch_resnet_concat as networks
resnet = networks.ResNet(7, resblocks, [16, 32])
resnet.cuda()
optimizer = optim.Adam(resnet.parameters(), lr=lr_init)
#lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.5)

def do_eval(resnet, val_loader, mae_best, epoch, sample, tgt_label):
    global expt_name
    loss_ = 0.
    #m_pred_, m_true_, mae_, pt_, wgts_ = [], [], [], [], []
    m_pred_, m_true_, mae_, pt_, = [], [], [], []
    iphi_, ieta_ = [], []
    label_ = []
    now = time.time()
    for i, data in enumerate(val_loader):
        #X, m0, pt, wgts = data['Xtz_aod'].cuda(), data['m'].cuda(), data['pt'], data['w']
        X, m0, pt = data['X_jet'].cuda(), data['am'].cuda(), data['apt']
        iphi, ieta = data['iphi'].cuda(), data['ieta'].cuda()
        #logits = resnet(X)
        logits = resnet([X, iphi, ieta])
        loss_ += mae_loss_wgtd(logits, m0).item()
        # Undo preproc on mass
        logits, m0 = inv_transform(logits), inv_transform(m0)
        #mae = (logits-m0).abs().mean()
        mae = (logits-m0).abs()
        # Store batch metrics:
        m_pred_.append(logits.tolist())
        m_true_.append(m0.tolist())
        mae_.append(mae.tolist())
        pt_.append(pt.tolist())
        #wgts_.append(wgts.tolist())
        iphi_.append(iphi.tolist())
        ieta_.append(ieta.tolist())
        label_.append(data['label'].tolist())

    now = time.time() - now
    #m_true_ = np.concatenate(m_true_)
    #m_pred_ = np.concatenate(m_pred_)
    #mae_ = np.array(mae_)
    #pt_ = np.concatenate(pt_)
    #wgts_ = np.concatenate(wgts_)
    #iphi_ = np.concatenate(iphi_)
    #ieta_ = np.concatenate(ieta_)
    label_ = np.concatenate(label_)
    m_true_ = np.concatenate(m_true_)[label_==tgt_label]
    m_pred_ = np.concatenate(m_pred_)[label_==tgt_label]
    #mae_ = np.array(mae_)[label_==tgt_label]
    mae_ = np.concatenate(mae_)[label_==tgt_label]
    pt_ = np.concatenate(pt_)[label_==tgt_label]
    #wgts_ = np.concatenate(wgts_)[label_==tgt_label]
    iphi_ = np.concatenate(iphi_)[label_==tgt_label]
    ieta_ = np.concatenate(ieta_)[label_==tgt_label]

    logger('%d: Val m_pred: %s...'%(epoch, str(np.squeeze(m_pred_[:5]))))
    logger('%d: Val m_true: %s...'%(epoch, str(np.squeeze(m_true_[:5]))))
    logger('%d: Val time:%.2fs in %d steps for N=%d'%(epoch, now, len(val_loader), len(m_true_)))
    logger('%d: Val loss:%f, mae:%f'%(epoch, loss_/len(val_loader), np.mean(mae_)))

    score_str = 'epoch%d_%s_mae%.4f'%(epoch, sample, np.mean(mae_))

    if 'pi0' in sample:
        # Check 2D m_true v m_pred
        logger('%d: Val m_true vs. m_pred, [0,1600,200] MeV:'%(epoch))
        sct = np.histogram2d(np.squeeze(m_true_), np.squeeze(m_pred_), bins=mass_bins)[0]
        logger(np.uint(np.fliplr(sct).T))
        # Extended version
        plt.plot(m_true_, m_pred_, ".", color='black', alpha=0.1, label='MAE = %.3f GeV'%np.mean(mae_))
        plt.xlabel(r'$\mathrm{m_{label}}$', size=16)
        plt.ylabel(r'$\mathrm{m_{pred}}$', size=16)
        plt.plot((0., 1.6), (1.2, 1.2), color='r', linestyle='--', alpha=0.5)
        plt.plot((1.2, 1.2), (-0.4, 1.6), color='r', linestyle='--', alpha=0.5)
        plt.plot((0., 1.6), (0., 0.), color='r', linestyle='--', alpha=0.5)
        plt.plot((0., 1.6), (0., 1.6), color='r', linestyle='--', alpha=0.5)
        plt.xlim(0., 1.6)
        plt.ylim(-0.4, 1.6)
        plt.legend(loc='upper left')
        plt.savefig('PLOTS/%s/mtruevpred_%s.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()
        # Truncated version
        plt.plot(m_true_, m_pred_, ".", color='black', alpha=0.125, label='MAE = %.3f GeV'%np.mean(mae_))
        plt.xlabel(r'$\mathrm{m_{label}}$', size=16)
        plt.ylabel(r'$\mathrm{m_{pred}}$', size=16)
        plt.plot((0., 1.2), (0., 1.2), color='r', linestyle='--', alpha=0.5)
        plt.xlim(0., 1.2)
        plt.ylim(0., 1.2)
        plt.legend(loc='upper left')
        plt.savefig('PLOTS/%s/mtruevpred_%s_trunc.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()

    # Check 1D m_pred
    hst = np.histogram(np.squeeze(m_pred_), bins=mass_bins)[0]
    logger('%d: Val m_pred, [0,1600,200] MeV: %s'%(epoch, str(np.uint(hst))))
    mlow = hst[0]
    mrms = np.std(hst)
    logger('%d: Val m_pred, [0,1600,200] MeV: low:%d, rms: %f'%(epoch, mlow, mrms))
    norm = 1.*len(m_pred_)/len(m0)
    plt.hist((m_true_ if 'pi0' in sample else np.zeros_like(m_true_)),\
            range=(3.6,15), bins=20, histtype='step', label=r'$\mathrm{m_{true}}$', linestyle='--', color='grey', alpha=0.6)
    plt.hist(m_pred_, range=(3.6,15), bins=20, histtype='step', label=r'$\mathrm{m_{pred}}$', linestyle='--', color='C0', alpha=0.6)
    plt.hist((m_true_ if 'pi0' in sample else np.zeros_like(m_true_)),\
            #range=(3.6,15), bins=20, histtype='step', label=r'$\mathrm{m_{true,w}}$', color='grey', weights=wgts_*norm)
            range=(3.6,15), bins=20, histtype='step', label=r'$\mathrm{m_{true,w}}$', color='grey', weights=norm)
    #plt.hist(m_pred_, range=(3.6,15), bins=20, histtype='step', label=r'$\mathrm{m_{pred,w}}$', color='C0', weights=wgts_*norm)
    plt.hist(m_pred_, range=(3.6,15), bins=20, histtype='step', label=r'$\mathrm{m_{pred,w}}$', color='C0', weights=norm)
    plt.xlim(3.6,15)
    plt.xlabel(r'$\mathrm{m}$', size=16)
    if 'pi0' in sample:
        plt.legend(loc='lower center')
    else:
        plt.legend(loc='upper right')
    #plt.show()
    plt.savefig('PLOTS/%s/mpred_%s.png'%(expt_name, score_str), bbox_inches='tight')
    plt.close()

    if run_logger:

        if 'pi0' in sample and 'val' in sample:
            filename = 'MODELS/%s/model_%s.pkl'%(expt_name, score_str.replace('pi0_',''))
            model_dict = {'model': resnet.state_dict(), 'optim': optimizer.state_dict()}
            torch.save(model_dict, filename)

    return np.mean(mae_)

# MAIN #
print_step = 2000
#print_step = 10000
mae_best = 1.
logger(">> Training <<<<<<<<")
for e in range(epochs):

    epoch = e+1
    epoch_wgt = 0.
    n_trained = 0
    logger('>> Epoch %d <<<<<<<<'%(epoch))

    # Run training
    #lr_scheduler.step()
    resnet.train()
    now = time.time()
    for i, data in enumerate(train_loader):
        #X, m0, wgts = data['Xtz_aod'].cuda(), data['m'].cuda(), data['w'].cuda()
        X, m0 = data['X_jet'].cuda(), data['am'].cuda()
        iphi, ieta = data['iphi'].cuda(), data['ieta'].cuda()
        optimizer.zero_grad()
        #logits = resnet(X)
        logits = resnet([X, iphi, ieta])
        #loss = mae_loss_wgtd(logits, m0, wgt=wgts)
        loss = mae_loss_wgtd(logits, m0)
        #break
        loss.backward()
        optimizer.step()
        epoch_wgt += len(m0) 
        #epoch_wgt += wgts.sum()
        n_trained += 1
        if i % print_step == 0:
            logits, m0 = inv_transform(logits), inv_transform(m0)
            mae = (logits-m0).abs().mean()
            logger('%d: (%d/%d) m_pred: %s...'%(epoch, i, len(train_loader), str(np.squeeze(logits.tolist()[:5]))))
            logger('%d: (%d/%d) m_true: %s...'%(epoch, i, len(train_loader), str(np.squeeze(m0.tolist()[:5]))))
            logger('%d: (%d/%d) Train loss:%f, mae:%f'%(epoch, i, len(train_loader), loss.item(), mae.item()))

    now = time.time() - now
    logits, m0 = inv_transform(logits), inv_transform(m0)
    mae = (logits-m0).abs().mean()
    logger('%d: Train time:%.2fs in %d steps for N:%d, wgt: %.f'%(epoch, now, len(train_loader), n_trained, epoch_wgt))
    logger('%d: Train loss:%f, mae:%f'%(epoch, loss.item(), mae.item()))

    if epoch > 1 and epoch < 70:
        pass
        continue

    # Run Validation
    resnet.eval()
    _ = do_eval(resnet, val_loader, mae_best, epoch, 'val_pseduscalar', 1)
    _ = do_eval(resnet, val_loader, mae_best, epoch, 'val_tau', 0)

    _ = do_eval(resnet, sg_loader, mae_best, epoch, 'test_pseudoscalar', 1)
    _ = do_eval(resnet, bg_loader, mae_best, epoch, 'test_tau', 0)

if run_logger:
    f.close()
