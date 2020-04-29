English|[中文](Readme_cn.md)

# Classification Network Application \(Python\)<a name="EN-US_TOPIC_0232649529"></a>

This application can run on the Atlas 200 DK to implement the inference function of the ResNet-18 network and output images with inference result labels and confidence.

The current application adapts to  [DDK&RunTime](https://ascend.huawei.com/resources)  of 1.3.0.0 as well as 1.32.0.0 and later versions.

## Prerequisites<a name="en-us_topic_0228752399_en-us_topic_0198304761_section137245294533"></a>

Before deploying this sample, ensure that:

-   Mind Studio  has been installed.

-   The Atlas 200 DK developer board has been connected to  Mind Studio, the SD card has been created, and the build environment has been configured.
-   The developer board is connected to the Internet over the USB port by default. The IP address of the developer board is  **192.168.1.2**.

## Sample Deployment<a name="en-us_topic_0228752399_section412811285117"></a>

You can use either of the following methods:

1.  Quick deployment: visit  [https://github.com/Atlas200dk/faster-deploy](https://github.com/Atlas200dk/faster-deploy).

    >![](public_sys-resources/icon-note.gif) **NOTE:**   
    >-   The quick deployment script can be used to quickly deploy multiple cases. Select  **classification-python**.  
    >-   The quick deployment script automatically completes code download, model conversion, and environment variable configuration. To learn about the detailed deployment process, select the common deployment mode. Go to  [2. Common deployment](#en-us_topic_0228752399_li3208251440).  

2.  <a name="en-us_topic_0228752399_li3208251440"></a>Common deployment: visit  [https://github.com/Atlas200dk/sample-README/tree/master/sample-classification-python](https://github.com/Atlas200dk/sample-README/tree/master/sample-classification-python).

    >![](public_sys-resources/icon-note.gif) **NOTE:**   
    >-   In this deployment mode, you need to manually download code, convert models, and configure environment variables. After that, you will have a better understanding of the process.  


## Environment Deployment<a name="en-us_topic_0228752399_en-us_topic_0198304761_section1759513564117"></a>

1.  Copy the application code to the developer board.

    Go to the root directory of the classification network application \(python\) code as the  Mind Studio  installation user, for example,  **$HOME/sample-classification-python**, and run the following command to copy the application code to the developer board:

    **scp -r ../sample-classification-python/ HwHiAiUser@192.168.1.2:/home/HwHiAiUser/HIAI\_PROJECTS**

    Enter the password of the developer board as prompted. The default password is  **Mind@123**, as shown in  [Figure 1](#en-us_topic_0228752399_en-us_topic_0228757083_en-us_topic_0198304761_fig1660453512014).

    **Figure  1**  Copying application code<a name="en-us_topic_0228752399_en-us_topic_0228757083_en-us_topic_0198304761_fig1660453512014"></a>  
    

    ![](figures/en-us_image_0228832431.png)

    Log in to the host side as the  **HwHiAiUser**  user in SSH mode on Ubuntu Server where  Mind Studio  is located.

    **ssh HwHiAiUser@192.168.1.2**

    Switch to the  **root**  user. The default password of the  **root**  user on the Atlas DK developer board is  **Mind@123**.

    **su root**

2.  Configure the network connection of the developer board.

    Configure the network connection of the Atlas DK developer board by referring to  [https://github.com/Atlas200dk/sample-README/tree/master/DK\_NetworkConnect](https://github.com/Atlas200dk/sample-README/tree/master/DK_NetworkConnect).

3.  Install the environment dependency.

    Configure the environment dependency by referring to  [https://github.com/Atlas200dk/sample-README/tree/master/DK\_Environment](https://github.com/Atlas200dk/sample-README/tree/master/DK_Environment).


## Application Running<a name="en-us_topic_0228752399_en-us_topic_0198304761_section6245151616426"></a>

1.  Switch to the  **HwHiAiUser**  user and go to the directory where the classification network application code is located.

    **su HwHiAiUser**

    **cd \~/HIAI\_PROJECTS/sample-classification-python/classificationapp**

2.  Run the application.

    Run the  **classify.py**  script to print the inference result on the execution terminal.

    Command example:

    **python classify.py**

    [Figure 2](#en-us_topic_0228752399_en-us_topic_0228757083_fig1863053617417)  shows the inference result after the execution is successful.

    **Figure  2**  Successful inference<a name="en-us_topic_0228752399_en-us_topic_0228757083_fig1863053617417"></a>  
    

    ![](figures/en-us_image_0228757215.png)

3.  Query the execution result.

    The execution result is stored in  **resent18Result**  of the current directory. You need to run the following command on the Atlas200 DK to copy the result to the Ubuntu server to view the inference result image:

    **scp -r username@host\_ip:/home/username/HIAI\_PROJECTS/sample-classification-python/resent18Result \~**

    -   **username**: user name of the developer board. The default value is  **HwHiAiUser**.
    -   **host\_ip**: IP address of the developer board. Generally, the IP address is  **192.168.1.2**  for USB connection and  **192.168.0.2**  for network cable connection.

    Command example:

    **scp -r HwHiAiUser@192.168.1.2:/home/HwHiAiUser/HIAI\_PROJECTS/sample-classification-python/resent18Result \~**

    This command copies the inference result to the home directory of the Mind Studio installation user. You can view the inference result directly.


## Remarks<a name="en-us_topic_0228752399_en-us_topic_0198304761_section1092612277429"></a>

-   The process of the classification network application \(Python\) is as follows:
    1.  Read a JPEG image from the  **ImageNetRaw**  directory.
    2.  Call OpenCV to resize the read JPEG image to 256 x 224 and convert it to YUV420SP.
    3.  Send the converted YUV420SP image data to Matrix for inference. The demo uses the ResNet-18 network, and the inference result is the confidence of 1000 classes.
    4.  Post-process the image. 1000 classification confidence levels are sorted, the highest confidence level and its classification label are selected, and the image is labeled. A labeled image is stored in the  **resnet18Result**  directory.

-   The directory structure of the classification network application \(Python\) is described as follows:
    -   **ImageNetRaw**: storing input images
    -   **classify.py**: main program
    -   **imageNetClasses.py**: labeling for 1000 classes of ImageNet
    -   **jpegHandler.py**: JPEG image processing, such as resizing, color space conversion \(CSC\), and text labeling
    -   **models**: storing model networks
    -   **resnet18Result**: storing labeled images


