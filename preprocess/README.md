
跨数据集：
    Windows：
        python 1-cwru.py --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --testnumber=1 --dataroot=D:\fault\cdan-machines\matdata\cwru-de-c4-dataset

        python 2-ims.py  --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --dataroot=F:/ds/IMS
        python 3-seu.py  --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --dataroot=D:/fault/cdan-machines/matdata/seu-bearingset
        python 4-jnu.py --normal=0 --fft=0 --framesize=1200 --trainnumber=2000 --dataroot=D:/fault/cdan-machines/matdata/jnu-bearingset


    服务器：


跨故障深度：
    Windows：
        python 5-pb-c3.py --dataroot=D:\fault\paderborn_dataset --fft=0 --train=0 --normal=0 
        python 5-pb-c3.py --dataroot=D:\fault\paderborn_dataset --fft=0 --train=1 --normal=0 

    服务器：