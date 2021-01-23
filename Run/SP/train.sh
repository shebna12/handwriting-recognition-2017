



$1 train --solver=$2"/solver.prototxt" 2>&1 | 
tee -a $2"/logs/train.log"



# RESUME TRAINING
 # --snapshot=models/sp_alexnet/caffe_alexnet_train_iter_738.solverstate 2>&1 | tee a- logs/log_80.log