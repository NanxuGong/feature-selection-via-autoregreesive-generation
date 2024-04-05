# nohup python train_controller.py --method_name rnn --task_name semeion --batch_size 1024 --epochs 300 --encoder_layers 2 --mlp_layers 3 --lr 0.001 > trainlog_test_in_big_set/semeion/semeion_rnn.log &
# wait

# nohup python train_controller.py --method_name transformer --task_name semeion --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/semeion/semeion_transformer.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name semeion --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/semeion/semeion_transformerVae.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name semeion --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/semeion/semeion_transformerVae.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name activity --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/activity/activity_transformerVae.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name activity --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/activity/activity_transformerVae.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name minist --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/minist/minist_transformerVae.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name minist --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/minist/minist_transformerVae.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name minist_fashion --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/minist_fashion/minist_fashion_transformerVae.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name minist_fashion --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/minist_fashion/minist_fashion_transformerVae.log &

# data augmentation experiments
# german credit | acctivity | openml_586
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 0 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/german_credit/german_credit_transformerVae_0.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 5 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/german_credit/german_credit_transformerVae_5.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 10 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/german_credit/german_credit_transformerVae_10.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 15 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/german_credit/german_credit_transformerVae_15.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 20 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/german_credit/german_credit_transformerVae_20.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 0 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/german_credit/german_credit_transformerVae_0.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 5 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/german_credit/german_credit_transformerVae_5.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 10 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/german_credit/german_credit_transformerVae_10.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 15 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/german_credit/german_credit_transformerVae_15.log &
# nohup python train_controller.py --method_name transformerVae --task_name german_credit --gen_num 20 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/german_credit/german_credit_transformerVae_20.log &
# wait

# # openml_586
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 0 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/openml_586/openml_586_transformerVae_0.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 5 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/openml_586/openml_586_transformerVae_5.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 10 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/openml_586/openml_586_transformerVae_10.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 15 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/openml_586/openml_586_transformerVae_15.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 20 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/openml_586/openml_586_transformerVae_20.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 0 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/openml_586/openml_586_transformerVae_0.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 5 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/openml_586/openml_586_transformerVae_5.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 10 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/openml_586/openml_586_transformerVae_10.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 15 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/openml_586/openml_586_transformerVae_15.log &
# nohup python train_controller.py --method_name transformerVae --task_name openml_586 --gen_num 20 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/openml_586/openml_586_transformerVae_20.log &
# wait

# # activity
# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 0 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/activity/activity_transformerVae_0.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 0 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/activity/activity_transformerVae_0.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 5 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/activity/activity_transformerVae_5.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 5 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/activity/activity_transformerVae_5.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 10 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/activity/activity_transformerVae_10.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 10 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/activity/activity_transformerVae_10.log &
# wait

# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 15 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/activity/activity_transformerVae_15.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 15 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/activity/activity_transformerVae_15.log &
# wait

# nohup python train_controller.py 1--method_name transformerVae --task_name activity --gen_num 20 --batch_size 64 --epochs 1000 --lr 0.0001 > trainlog_test_in_big_set/activity/activity_transformerVae_20.log &
# wait
# nohup python train_controller.py --method_name transformerVae --task_name activity --gen_num 20 --batch_size 64 --epochs 500 --lr 0.0001 --pre_train False >> trainlog_test_in_big_set/activity/activity_transformerVae_20.log &
# bash /root/NIPSAutoFS/code/ours/experiment.sh
nohup python3 /root/NIPSAutoFS/code/baseline/automatic_feature_selection_gen.py  --name svmguide3 --> result/activity/gen_feature.log &

# nohup python3 /root/NIPSAutoFS/code/ours/train_controller.py --method_name transformerVae --task_name activity --epochs 300 --pre_train True > result/activity/transformerVae_tf.log &
# nohup python3 /root/NIPSAutoFS/code/ours/train_controller.py --method_name rnn --task_name activity --pre_train True > result/activity/rnn_tf.log &
# nohup python3 /root/NIPSAutoFS/code/ours/train_controller.py --method_name transformer --task_name activity --epochs 300 --pre_train True > result/activity/transformer_tf.log &


# nohup python3 /root/NIPSAutoFS/code/ours/train_controller.py --method_name transformerVae --task_name svmguide3 --epochs 300 --pre_train True > result/svmguide3/transformerVae_ft.log &
# nohup python3 /root/NIPSAutoFS/code/ours/train_controller.py --method_name rnn --task_name svmguide3 --pre_train True > result/svmguide3/rnn_ft.log &
# nohup python3 /root/NIPSAutoFS/code/ours/train_controller.py --method_name transformer --task_name svmguide3 --epochs 300 --pre_train True > result/svmguide3/transformer_ft.log &
