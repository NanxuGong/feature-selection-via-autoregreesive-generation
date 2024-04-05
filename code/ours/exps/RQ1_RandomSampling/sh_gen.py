def gen_code(name):
    print(f'/home/xiaomeng/miniconda3/envs/shaow/bin/python3 -u /home/xiaomeng/jupyter_base/AutoFS/code/ours/exps/RQ1:RandomSampling/train_controller_random.py --task_name {name}')

if __name__ == '__main__':
    for i in ['spectf', 'svmguide3', 'spam_base',
                  'ionosphere', 'megawatt1', 'uci_credit_card',
                                             'openml_618', 'openml_589', 'openml_616', 'openml_607', 'openml_620',
                  'openml_637', 'german_credit',
                  'openml_586', 'higgs']:
        gen_code(i)