import os
import copy
import numpy as np
import logging
import shutil
import threading
import gc
import zipfile
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
# from autoaugment import CIFAR10Policy

B=5


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
      

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


# def _data_transforms_cifar10(cutout_size, autoaugment=False):
#     CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
#     CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
#
#     if autoaugment:
#         train_transform = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             CIFAR10Policy(),
#             transforms.ToTensor(),
#             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#         ])
#     else:
#         train_transform = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#         ])
#     if cutout_size is not None:
#         train_transform.transforms.append(Cutout(cutout_size))
#
#     valid_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#         ])
#     return train_transform, valid_transform


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def convert_to_pil(bytes_obj):
    img = Image.open(BytesIO(bytes_obj))
    return img.convert('RGB')


class ReadImageThread(threading.Thread):
    def __init__(self, root, fnames, class_id, target_list):
        threading.Thread.__init__(self)
        self.root = root
        self.fnames = fnames
        self.class_id = class_id
        self.target_list = target_list
        
    def run(self):
        for fname in self.fnames:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(self.root, fname)
                with open(path, 'rb') as f:
                    image = f.read()
                item = (image, self.class_id)
                self.target_list.append(item)


class InMemoryDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.samples = []
        classes, class_to_idx = self.find_classes(self.path)
        dir = os.path.expanduser(self.path)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if num_workers == 1:
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            path = os.path.join(root, fname)
                            with open(path, 'rb') as f:
                                image = f.read()
                            item = (image, class_to_idx[target])
                            self.samples.append(item)
                else:
                    fnames = sorted(fnames)
                    num_files = len(fnames)
                    threads = []
                    res = [[] for i in range(num_workers)]
                    num_per_worker = num_files // num_workers
                    for i in range(num_workers):
                        start_index = num_per_worker * i
                        end_index = num_files if i == num_workers - 1 else num_per_worker * (i+1)
                        thread = ReadImageThread(root, fnames[start_index:end_index], class_to_idx[target], res[i])
                        threads.append(thread)
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    for item in res:
                        self.samples += item
                    del res, threads
                    gc.collect()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def find_classes(root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    

class ZipDataset(data.Dataset):
    def __init__(self, path, transform=None):
        super(ZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        with zipfile.ZipFile(self.path, 'r') as reader:
            classes, class_to_idx = self.find_classes(reader)
            fnames = sorted(reader.namelist())
        for fname in fnames:
            if self.is_directory(fname):
                continue
            target = self.get_target(fname)
            item = (fname, class_to_idx[target])
            self.samples.append(item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        with zipfile.ZipFile(self.path, 'r') as reader:
            sample = reader.read(sample)
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]
    
    @staticmethod
    def find_classes(reader):
        classes = [ZipDataset.get_target(name) for name in reader.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ReadZipImageThread(threading.Thread):
    def __init__(self, reader, fnames, class_to_idx, target_list):
        threading.Thread.__init__(self)
        self.reader = reader
        self.fnames = fnames
        self.target_list = target_list
        self.class_to_idx = class_to_idx
    
    def run(self):
        for fname in self.fnames:
            if InMemoryZipDataset.is_directory(fname):
                continue
            image = self.reader.read(fname)
            class_id = self.class_to_idx[InMemoryZipDataset.get_target(fname)]
            item = (image, class_id)
            self.target_list.append(item)


class InMemoryZipDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        reader = zipfile.ZipFile(self.path, 'r')
        classes, class_to_idx = self.find_classes(reader)
        fnames = sorted(reader.namelist())
        if num_workers == 1:
            for fname in fnames:
                if self.is_directory(fname):
                    continue
                target = self.get_target(fname)
                image = reader.read(fname)
                item = (image, class_to_idx[target])
                self.samples.append(item)
        else:
            num_files = len(fnames)
            threads = []
            res = [[] for i in range(num_workers)]
            num_per_worker = num_files // num_workers
            for i in range(num_workers):
                start_index = num_per_worker * i
                end_index = num_files if i == num_workers - 1 else (i+1) * num_per_worker
                thread = ReadZipImageThread(reader, fnames[start_index:end_index], class_to_idx, res[i])
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for item in res:
                self.samples += item
            del res, threads
            gc.collect()
        reader.close()
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]

    @staticmethod
    def find_classes(fname):
        classes = [ZipDataset.get_target(name) for name in fname.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class FSDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, redundancy, targets=None, train=True, sos_id=-1, eos_id=-1):
        super(FSDataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = copy.deepcopy(inputs)
        self.redundancy = copy.deepcopy(redundancy)
        self.targets = copy.deepcopy(targets)
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id
        # self.swap = swap
    
    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_redundancy = self.redundancy[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = self.targets[index]
        encoder_input[encoder_input==-1] = self.eos_id
        # if self.swap:
        #     a = np.random.randint(0, 5)
        #     b = np.random.randint(0, 5)
        #     encoder_input = encoder_input[:4 * a] + encoder_input[4 * a + 2:4 * a + 4] + \
        #                     encoder_input[4 * a:4 * a + 2] + encoder_input[4 * (a + 1):20 + 4 * b] + \
        #                     encoder_input[20 + 4 * b + 2:20 + 4 * b + 4] + encoder_input[20 + 4 * b:20 + 4 * b + 2] + \
        #                     encoder_input[20 + 4 * (b + 1):]

        if self.train:
            decoder_input = torch.cat((torch.tensor([self.sos_id]), encoder_input[:-1]))
            sample = {
                'encoder_input': encoder_input.long(),
                'encoder_target': encoder_target,
                'encoder_redundancy':encoder_redundancy,
                'decoder_input': decoder_input.long(),
                'decoder_target': encoder_input.long(),
            }
        else:
            sample = {
                'encoder_input': encoder_input.long(),
                'encoder_redundancy':encoder_redundancy,
                'decoder_target': encoder_input.long(),
            }
            if encoder_target is not None:
                sample['encoder_target'] = encoder_target
        return sample
    
    def __len__(self):
        return len(self.inputs)


class TripletFSDataset(torch.utils.data.Dataset):
    def __init__(self, h_inputs, t_inputs, targets=None, train=True, sos_id=-1, eos_id=-1):
        super().__init__()
        if targets is not None:
            assert len(h_inputs) == len(targets) and len(t_inputs) == len(targets)
        self.h_inputs = copy.deepcopy(h_inputs)
        self.t_input = copy.deepcopy(t_inputs)
        self.targets = copy.deepcopy(targets)
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id

    def __getitem__(self, index):
        encoder_input = self.h_inputs[index]
        decoder_input = self.t_input[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = self.targets[index]
        encoder_input[encoder_input==-1] = self.eos_id
        decoder_input[decoder_input==-1] = self.eos_id

        if self.train:
            decoder_input = torch.cat((torch.tensor([self.sos_id]), decoder_input[:-1]))
            sample = {
                'encoder_input': encoder_input.long(),
                'encoder_target': encoder_target,
                'decoder_input': decoder_input.long(),
                'decoder_target': decoder_input.long(),
            }
        else:
            sample = {
                'encoder_input': encoder_input.long(),
                'decoder_target': decoder_input.long(),
            }
            if encoder_target is not None:
                sample['encoder_target'] = encoder_target
        return sample

    def __len__(self):
        return len(self.targets)

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
      

def save(model_path, args, model, epoch, step, optimizer, best_acc_top1, is_best=True):
    if hasattr(model, 'module'):
        model = model.module
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'best_acc_top1': best_acc_top1,
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    torch.save(state_dict, filename)
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    shutil.copyfile(filename, newest_filename)
    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)
  

def load(model_path):
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    if not os.path.exists(newest_filename):
        return None, None, 0, 0, None, 0
    state_dict = torch.load(newest_filename)
    args = state_dict['args']
    model_state_dict = state_dict['model']
    epoch = state_dict['epoch']
    step = state_dict['step']
    optimizer_state_dict = state_dict['optimizer']
    best_acc_top1 = state_dict.get('best_acc_top1')
    return args, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1

  
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


# def sample_arch(arch_pool, prob=None):
#     N = len(arch_pool)
#     indices = [i for i in range(N)]
#     if prob is not None:
#         prob = np.array(prob, dtype=np.float32)
#         prob = prob / prob.sum()
#         index = np.random.choice(indices, p=prob)
#     else:
#         index = np.random.choice(indices)
#     arch = arch_pool[index]
#     return arch
#
#
# def generate_arch(n, num_nodes, num_ops=7):
#     def _get_arch():
#         arch = []
#         for i in range(2, num_nodes+2):
#             p1 = np.random.randint(0, i)
#             op1 = np.random.randint(0, num_ops)
#             p2 = np.random.randint(0, i)
#             op2 = np.random.randint(0 ,num_ops)
#             arch.extend([p1, op1, p2, op2])
#         return arch
#     archs = [[_get_arch(), _get_arch()] for i in range(n)] #[[[conv],[reduc]]]
#     return archs
#
#
# def build_dag(arch):
#     if arch is None:
#         return None, None
#     # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
#     arch = list(map(int, arch.strip().split()))
#     length = len(arch)
#     conv_dag = arch[:length//2]
#     reduc_dag = arch[length//2:]
#     return conv_dag, reduc_dag
#
#
# def parse_arch_to_seq(cell, nodes=B):
#     seq = []
#     for i in range(nodes):
#         prev_node1 = cell[4*i] + 1
#         prev_node2 = cell[4*i+2] + 1
#         op1 = cell[4*i+1] + 7
#         op2 = cell[4*i+3] + 7
#         seq.extend([prev_node1, op1, prev_node2, op2])
#     return seq
#
#
# def parse_seq_to_arch(seq, nodes=B):
#     n = len(seq)
#
#     def _parse_cell(cell_seq):
#         cell_arch = []
#         for i in range(nodes):
#             p1 = cell_seq[4*i] - 1
#             op1 = cell_seq[4*i+1] - 7
#             p2 = cell_seq[4*i+2] - 1
#             op2 = cell_seq[4*i+3] - 7
#             cell_arch.extend([p1, op1, p2, op2])
#         return cell_arch
#     conv_seq = seq[:n//2]
#     reduc_seq = seq[n//2:]
#     conv_arch = _parse_cell(conv_seq)
#     reduc_arch = _parse_cell(reduc_seq)
#     arch = [conv_arch, reduc_arch]
#     return arch


def pairwise_accuracy(la, lb):
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    N = len(la)
    assert N == len(lb)
  
    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c
  
    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N


def generate_eval_points(eval_epochs, stand_alone_epoch, total_epochs):
    if isinstance(eval_epochs, list):
        return eval_epochs
    assert isinstance(eval_epochs, int)
    res = []
    eval_point = eval_epochs - stand_alone_epoch
    while eval_point + stand_alone_epoch <= total_epochs:
        res.append(eval_point)
        eval_point += eval_epochs
    return res
