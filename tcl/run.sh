foo() {
     nohup python main.py --models=sourceonly --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --n_epochs=100  --lr=1e-4 --gpu_id=0 
     nohup python main.py --models=deepcoral --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --mmd_gamma=1.0 --lr=1e-4 --n_epochs=100  --gpu_id=0
     nohup python main.py --models=ddc --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --mmd_gamma=1.0 --lr=1e-4 --n_epochs=100  --gpu_id=0
     nohup python main.py --models=DAN_Linear --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --mmd_gamma=1.0 --n_epochs=100 --lr=1e-4 --gpu_id=0
     nohup python main.py --models=dann --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4

     nohup python main.py --models=wasserstein --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --triplet_type=none --w_weight=1.0 --gpu_id=0
     nohup python main.py --models=wasserstein --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --triplet_type=all --w_weight=1.0 --t_weight=0.05 --t_confidence=0.5 --gpu_id=0
     nohup python main.py --models=wasserstein --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --triplet_type=src --w_weight=1.0 --t_weight=0.05 --t_confidence=0.5 --gpu_id=0
     nohup python main.py --models=wasserstein --dataroot=$1 --src=$2 --dest=$3 --n_flattens=$4 --triplet_type=tgt --w_weight=1.0 --t_weight=0.05 --t_confidence=0.5 --gpu_id=0

     sleep 30s
}


foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/test/noload dec4no-fill fec4no-fill 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/test/noload fec4no-fill dec4no-fill 96

foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 0HP 1HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 0HP 2HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 0HP 3HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 1HP 0HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 1HP 2HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 1HP 3HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 2HP 0HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 2HP 1HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 2HP 3HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 3HP 0HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 3HP 1HP 96
foo /nas/code/1025/wdcnn_bearning_fault_diagnosis/output/deonly-fft 3HP 2HP 96

foo /nas/code/0919/data/paderborn_fft_p2560_c15 paderborn_fft_p2560_c15_l1 paderborn_fft_p2560_c15_l2 288
foo /nas/code/0919/data/paderborn_fft_p2560_c15 paderborn_fft_p2560_c15_l1 paderborn_fft_p2560_c15_l3 288
foo /nas/code/0919/data/paderborn_fft_p2560_c15 paderborn_fft_p2560_c15_l2 paderborn_fft_p2560_c15_l1 288
foo /nas/code/0919/data/paderborn_fft_p2560_c15 paderborn_fft_p2560_c15_l2 paderborn_fft_p2560_c15_l3 288
foo /nas/code/0919/data/paderborn_fft_p2560_c15 paderborn_fft_p2560_c15_l3 paderborn_fft_p2560_c15_l1 288
foo /nas/code/0919/data/paderborn_fft_p2560_c15 paderborn_fft_p2560_c15_l3 paderborn_fft_p2560_c15_l2 288

foo /nas/code/0919/data/paderborn_fft_p2560_c3 paderborn_fft_p2560_c3_l1 paderborn_fft_p2560_c3_l2 288 
foo /nas/code/0919/data/paderborn_fft_p2560_c3 paderborn_fft_p2560_c3_l1 paderborn_fft_p2560_c3_l3 288
foo /nas/code/0919/data/paderborn_fft_p2560_c3 paderborn_fft_p2560_c3_l2 paderborn_fft_p2560_c3_l1 288 
foo /nas/code/0919/data/paderborn_fft_p2560_c3 paderborn_fft_p2560_c3_l2 paderborn_fft_p2560_c3_l3 288
foo /nas/code/0919/data/paderborn_fft_p2560_c3 paderborn_fft_p2560_c3_l3 paderborn_fft_p2560_c3_l1 288 
foo /nas/code/0919/data/paderborn_fft_p2560_c3 paderborn_fft_p2560_c3_l3 paderborn_fft_p2560_c3_l2 288
