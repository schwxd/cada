set sleep_count=120
set slim_count=0
set MODEL=DCTLN

nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load0 --dest=load1 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load0 --dest=load2 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load0 --dest=load3 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%

nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load1 --dest=load0 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load1 --dest=load2 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load1 --dest=load3 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%

nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load2 --dest=load1 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load2 --dest=load0 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load2 --dest=load3 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%

nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load3 --dest=load1 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load3 --dest=load0 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%
nohup python main.py --models=%MODEL% --dataroot=data/motor_train1_fft0_normal0_frame1200 --src=load3 --dest=load2 --n_flattens=256 --slim=%slim_count% --mmd_gamma=1 --bn=1
timeout /T %sleep_count%


set sleep_count=120
set slim_count=0
set MODEL=DCTLN
