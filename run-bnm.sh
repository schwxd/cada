foo() {
    nohup python main.py --models=$1 --dataroot=data/machines-c4 --src=DEload2 --dest=ims-severe-fft1-fs1024-num1000 --n_flattens=1920 --slim=$2 --target_labeling=$3 --bnm_sw=$4 --bnm_tw=$5 --bnm_ew=$6
    sleep 30s
    nohup python main.py --models=$1 --dataroot=data/machines-c4 --src=DEload2 --dest=jnu10_1024 --n_flattens=1920 --slim=$2 --target_labeling=$3 --bnm_sw=$4 --bnm_tw=$5 --bnm_ew=$6
    sleep 30s
    nohup python main.py --models=$1 --dataroot=data/machines-c4 --src=ims-severe-fft1-fs1024-num1000 --dest=DEload2 --n_flattens=1920 --slim=$2 --target_labeling=$3 --bnm_sw=$4 --bnm_tw=$5 --bnm_ew=$6
    sleep 30s
    nohup python main.py --models=$1 --dataroot=data/machines-c4 --src=ims-severe-fft1-fs1024-num1000 --dest=jnu10_1024 --n_flattens=1920 --slim=$2 --target_labeling=$3 --bnm_sw=$4 --bnm_tw=$5 --bnm_ew=$6
    sleep 30s
    nohup python main.py --models=$1 --dataroot=data/machines-c4 --src=jnu10_1024 --dest=DEload2 --n_flattens=1920 --slim=$2 --target_labeling=$3 --bnm_sw=$4 --bnm_tw=$5 --bnm_ew=$6
    sleep 30s
    nohup python main.py --models=$1 --dataroot=data/machines-c4 --src=jnu10_1024 --dest=ims-severe-fft1-fs1024-num1000 --n_flattens=1920 --slim=$2 --target_labeling=$3 --bnm_sw=$4 --bnm_tw=$5 --bnm_ew=$6
    sleep 30s
}

foo dann_vat 1 1 0 0 0
foo dann_vat 1 1 0 0 1
foo dann_vat 1 1 0 1 0
foo dann_vat 1 1 0 1 1
foo dann_vat 1 1 1 0 0
foo dann_vat 1 1 1 0 1
foo dann_vat 1 1 1 1 0
foo dann_vat 1 1 1 1 1

