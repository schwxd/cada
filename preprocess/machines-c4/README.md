
跨数据集：
    Windows：
        python 1-cwru.py --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --testnumber=1 --dataroot=D:\fault\cdan-machines\matdata\cwru-de-c4-dataset

        python 2-ims.py  --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --dataroot=F:/ds/IMS
        python 3-seu.py  --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --dataroot=D:/fault/cdan-machines/matdata/seu-bearingset
        python 4-jnu.py --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --dataroot=D:/fault/cdan-machines/matdata/jnu-bearingset
        python 6-simulink.py --normal=0 --fft=0 --framesize=1200  --dataroot=D:/fault/cada/matdata 
        # 读取inner.mat, normal.mat, outer.mat三个文件，trainnumber写在了文件中

    c4:
python 1-cwru.py --fft=1 --framesize=1024 --trainnumber=1000 --dataroot=D:\fault\cada\matdata\de-only\c4\load0

python 2-ims.py  --normal=0 --fft=1 --framesize=1024 --trainnumber=1000 --dataroot=F:/ds/IMS

python 3-seu.py --fft=1 --framesize=1024 --trainnumber=1000 --dataroot=D:/fault/cdan-machines/matdata/seu-bearingset

python 4-jnu.py --fft=1 --framesize=1024 --trainnumber=1000 --dataroot=D:/fault/cdan-machines/matdata/jnu-bearingset

    服务器：
        python 1-cwru.py --normal=0 --fft=1 --framesize=1200 --trainnumber=2000 --testnumber=1 --dataroot=/nas/data/cwru/de-only/c10

跨故障深度：
    Windows：
        python 5-pb-c3.py --dataroot=D:\fault\paderborn_dataset --fft=0 --train=0 --normal=0 
        python 5-pb-c3.py --dataroot=D:\fault\paderborn_dataset --fft=0 --train=1 --normal=0 

    服务器：
