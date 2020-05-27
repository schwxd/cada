set sleep_count=90
set slim_count=5
set MODEL=DCTLN


nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=DEload2 --dest=ims-severe-fft1-fs1024-num1000 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=DEload2 --dest=jnu10_1024 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=DEload2 --dest=seu30-fft1-fs1024-num1000 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%


nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=ims-severe-fft1-fs1024-num1000 --dest=DEload2 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=ims-severe-fft1-fs1024-num1000 --dest=jnu10_1024 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=ims-severe-fft1-fs1024-num1000 --dest=seu30-fft1-fs1024-num1000 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%


nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=jnu10_1024 --dest=ims-severe-fft1-fs1024-num1000 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=jnu10_1024 --dest=DEload2 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=jnu10_1024 --dest=seu30-fft1-fs1024-num1000 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%

nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=seu30-fft1-fs1024-num1000 --dest=ims-severe-fft1-fs1024-num1000 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=seu30-fft1-fs1024-num1000 --dest=DEload2 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/machines-c4 --src=seu30-fft1-fs1024-num1000 --dest=jnu10_1024 --n_flattens=1920 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%


