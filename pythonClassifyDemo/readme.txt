classify Demo 说明
1 安装部署
  1.1 demo运行 依赖于OpenCV，所以需要先安装python-opencv。建议安装步骤如下：
       a ssh登录到Atlas 200DK，并切换到root账号 
       b 设置正确的apt安装源，请注意Atals200DK OS为基于arm架构的ubuntu 16.04.4
       c apt-get install python-opencv
  1.2 将classifyDemo.rar copy到Atlas200 DK的 /home/HwHiAiUser/ 路径下，并解压缩
  1.3 执行 python classifyDemo.py  

2 demo 功能说明
    a 从ImageNetRaw 目录下读取jpeg图片 （总共100张）
    b 将读取的jpeg图片调用opencv resize到256*224,并转换成YUV420 sp
    c 将转换后的YUV420 SP图片数据 送入 Matrix进行推理。demo采用resnet18网络，
      推理结果是1000个分类的置信度。
    d 后处理阶段，将1000个分类置信度排序，选取最高置信度及其分类标签，
      在图片上进行标注。标注后图片存放在resnet18Result 目录下。

3 demo 文件说明
      ImageNetRaw/              -- 存放输入图片
      classifyDemo.py*          -- 主程序
      imageNetClasses.py*       -- imageNet 1000种分类标签
      jpegHandler.py*           -- jpeg图片处理，如resize、色域转换、文字标注等
      models/                   -- 存放网络模型
      resnet18Result/           -- 存放标注能够的图片文件
      
4 网络模型
    本demo采用resnet18，demo中包含转换后的Davinci模型（模型文件后缀名为.om）。
    由于Davinci模型和Atals200 DK的软件版本存在配套关系,如果demo中附带的Resnet18.om
    和Atlas 200DK 软件版本不匹配，可将与Atlas 200 DK配套的
    mind studio 安装路径/model-zoo/built-in-model/.resnet18/ 路径下的resnet18.om 
    copy到本demo的models 路径下替换原有的resnet18.om