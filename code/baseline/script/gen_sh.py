for task_name in ['spectf', 'svmguide3', 'german_credit', 'spam_base',
                  'ionosphere', 'megawatt1', 'uci_credit_card',
                                             'openml_618', 'openml_589', 'openml_616', 'openml_607', 'openml_620',
                  'openml_637',
                  'openml_586', 'arrhythmia', 'uci_credit_card', 'higgs', 'ap_omentum_ovary', 'activity', 'mice_protein'
                  , 'coil-20', 'isolet', 'minist', 'minist_fashion']:
    #  'uci_credit_card''higgs',
    strs = f'nohup /home/xiaomeng/miniconda3/envs/shaow/bin/python -u ' \
           f'/home/xiaomeng/jupyter_base/AutoFS/code/baseline/automatic_feature_selection_gen.py --name {task_name} > /home/xiaomeng/jupyter_base/AutoFS/code/baseline/script/{task_name}.log & '
    print(strs)
