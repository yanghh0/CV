
### My Environment
    System          : windows10  
    Python Version  : Python3.6


### How To Run
    python main.py

### 程序说明
    1. 程序运行后默认先通过摄像头录制 10s 的视频, 保存为 video.avi.
    2. 录制结束后 OpenCV 再读取 video.avi 进行播放.
    3. 播放过程中, 按空格触发角点检测算法, 此时窗口会被 destroy.
    4. 角点检测程序跑完后会依次显示(需要按任意键才能显示下一张图片): 最大特征值图，最小特征值图，R图，叠加检测结果的图片。
       然后再按任意键继续播放视频.
    5. 每次触发检测都会保存 4 个文件: 
       - i-e-max.jpg (最大特征值图)  
       - i-e-min.jpg (最小特征值图)
       - i-R.jpg (R图)  
       - i-final.jpg (叠加检测结果的图片)
       最前面的数 i 代表第 i 次触发检测的结果
    6. TestImage() 函数是用来测试单张图像的。
    7. TestVideo() 函数是调用摄像头先录后播，程序默认执行此函数。

    
