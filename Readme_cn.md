中文|[English](Readme.md)

# 分类网络应用（Python）<a name="ZH-CN_TOPIC_0232337297"></a>

本Application支持运行在Atlas 200 DK上，实现了Resnet18网络的推理功能并输出带有推理结果标签和置信度的图片。

当前分支中的应用适配**1.3.0.0**与**1.32.0.0及以上**版本的[DDK&RunTime](https://ascend.huawei.com/resources)。

## 前提条件<a name="zh-cn_topic_0228752399_zh-cn_topic_0198304761_section137245294533"></a>

部署此Sample前，需要准备好以下环境：

-   已完成Mind Studio的安装。

-   已完成Atlas 200 DK开发者板与Mind Studio的连接，SD卡的制作、编译环境的配置等。
-   由于需要配置开发板联网，默认设置为USB连接，开发板地址为192.168.1.2。

## 样例部署<a name="zh-cn_topic_0228752399_section412811285117"></a>

可以选择如下快速部署或者常规方法部署，二选一即可：

1.  快速部署，请参考：  [https://github.com/Atlas200dk/faster-deploy](https://github.com/Atlas200dk/faster-deploy)  。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >-   该快速部署脚本可以快速部署多个案例，请选择classification-python案例部署即可。  
    >-   该快速部署脚本自动完成了代码下载、模型转换、环境变量配置等流程，如果需要了解详细的部署过程请选择常规部署方式。转：**[2. 常规部署](#zh-cn_topic_0228752399_li3208251440)**  

2.  <a name="zh-cn_topic_0228752399_li3208251440"></a>常规部署，请参考：  [https://github.com/Atlas200dk/sample-README/tree/master/sample-classification-python](https://github.com/Atlas200dk/sample-README/tree/master/sample-classification-python)  。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >
    >-   该部署方式，需要手动完成代码下载、模型转换、环境变量配置等过程。完成后，会对其中的过程更加了解。  


## 环境部署<a name="zh-cn_topic_0228752399_zh-cn_topic_0198304761_section1759513564117"></a>

1.  应用代码拷贝到开发板。

    以Mind Studio安装用户进入分类网络应用\(python\)代码所在根目录，如：$HOME/sample-classification-python，执行以下命令将应用代码拷贝到开发板。

    **scp -r ../sample-classification-python/ HwHiAiUser@192.168.1.2:/home/HwHiAiUser/HIAI\_PROJECTS**

    提示password时输入开发板密码，开发板默认密码为**Mind@123**，如[图 应用代码拷贝](#zh-cn_topic_0228752399_zh-cn_topic_0228757083_zh-cn_topic_0198304761_fig1660453512014)。

    **图 1** **应用代码拷贝**<a name="zh-cn_topic_0228752399_zh-cn_topic_0228757083_zh-cn_topic_0198304761_fig1660453512014"></a>  
    

    ![](figures/zh-cn_image_0228832431.png)

    在Mind Studio所在Ubuntu服务器中，以HwHiAiUser用户SSH登录到Host侧。

    **ssh HwHiAiUser@192.168.1.2**

    切换到root用户，开发板中root用户默认密码为**Mind@123**。

    **su root**

2.  配置开发板联网。

    请参考[https://github.com/Atlas200dk/sample-README/tree/master/DK\_NetworkConnect](https://github.com/Atlas200dk/sample-README/tree/master/DK_NetworkConnect)  ，进行开发板网络连接配置。

3.  安装环境依赖。

    请参考[https://github.com/Atlas200dk/sample-README/tree/master/DK\_Environment](https://github.com/Atlas200dk/sample-README/tree/master/DK_Environment)  ，进行环境依赖配置。


## 程序运行<a name="zh-cn_topic_0228752399_zh-cn_topic_0198304761_section6245151616426"></a>

1.  切换HwHiAiUser用户，并进入分类网络应用代码所在目录。

    **su HwHiAiUser**

    **cd \~/HIAI\_PROJECTS/sample-classification-python/classificationapp**

2.  执行应用程序。

    执行**classify.py**脚本会将推理结果在执行终端直接打印显示。

    命令示例如下所示：

    **python classify.py**

    执行成功后效果如[图 推理成功示意图](#zh-cn_topic_0228752399_zh-cn_topic_0228757083_fig1863053617417)所示。

    **图 2**  推理成功示意图<a name="zh-cn_topic_0228752399_zh-cn_topic_0228757083_fig1863053617417"></a>  
    

    ![](figures/zh-cn_image_0228757215.png)

3.  执行结果查看。

    执行结果保存在当前目录下的resent18Result目录下，需要在Atlas200DK中用以下命令将结果拷贝到Ubuntu服务器中查看推理结果图片。

    **scp -r username@host\_ip:/home/username/HIAI\_PROJECTS/sample-classification-python/classificationapp/resent18Result \~**

    -   username：开发板用户﻿名，默认为HwHiAiUser。
    -   host\_ip：开发板ip，USB连接一般为192.168.1.2.网线连接时一般为192.168.0.2。

    **命令示例：**

    **scp -r HwHiAiUser@192.168.1.2:/home/HwHiAiUser/HIAI\_PROJECTS/sample-classification-python/classificationapp/resent18Result \~**

    该命令会把推理结果拷贝到Mindstudio安装用户的家目录中，可以直接查看。


## 相关说明<a name="zh-cn_topic_0228752399_zh-cn_topic_0198304761_section1092612277429"></a>

-   **分类网络应用（Python）的流程说明如下**：
    1.  从ImageNetRaw目录下读取jpeg图片。
    2.  将读取的jpeg图片调用opencv resize到256\*224，并转换成YUV420SP。
    3.  将转换后的YUV420SP图片数据送入Matrix进行推理。demo采用的是resnet18网络，推理结果是1000个分类的置信度
    4.  后处理阶段，将1000个分类置信度排序，选取最高置信度及其分类标签，再图片上进行标注。标注后图片存放在resnet18Result目录下。

-   **分类网络应用（Python）的文件架构说明如下**：
    -   ImageNetRaw：存放输入图片
    -   classify.py：主程序
    -   imageNetClasses.py：imageNet 1000种分类标签
    -   jpegHandler.py：jpeg图片处理，如resize、色域转换、文字标注等
    -   models：存放模型网络
    -   resnet18Result：存放标注后的图片


