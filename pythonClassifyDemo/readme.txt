classify Demo ˵��
1 ��װ����
  1.1 demo���� ������OpenCV��������Ҫ�Ȱ�װpython-opencv�����鰲װ�������£�
       a ssh��¼��Atlas 200DK�����л���root�˺� 
       b ������ȷ��apt��װԴ����ע��Atals200DK OSΪ����arm�ܹ���ubuntu 16.04.4
       c apt-get install python-opencv
  1.2 ��classifyDemo.rar copy��Atlas200 DK�� /home/HwHiAiUser/ ·���£�����ѹ��
  1.3 ִ�� python classifyDemo.py  

2 demo ����˵��
    a ��ImageNetRaw Ŀ¼�¶�ȡjpegͼƬ ���ܹ�100�ţ�
    b ����ȡ��jpegͼƬ����opencv resize��256*224,��ת����YUV420 sp
    c ��ת�����YUV420 SPͼƬ���� ���� Matrix��������demo����resnet18���磬
      ��������1000����������Ŷȡ�
    d ����׶Σ���1000���������Ŷ�����ѡȡ������Ŷȼ�������ǩ��
      ��ͼƬ�Ͻ��б�ע����ע��ͼƬ�����resnet18Result Ŀ¼�¡�

3 demo �ļ�˵��
      ImageNetRaw/              -- �������ͼƬ
      classifyDemo.py*          -- ������
      imageNetClasses.py*       -- imageNet 1000�ַ����ǩ
      jpegHandler.py*           -- jpegͼƬ������resize��ɫ��ת�������ֱ�ע��
      models/                   -- �������ģ��
      resnet18Result/           -- ��ű�ע�ܹ���ͼƬ�ļ�
      
4 ����ģ��
    ��demo����resnet18��demo�а���ת�����Davinciģ�ͣ�ģ���ļ���׺��Ϊ.om����
    ����Davinciģ�ͺ�Atals200 DK������汾�������׹�ϵ,���demo�и�����Resnet18.om
    ��Atlas 200DK ����汾��ƥ�䣬�ɽ���Atlas 200 DK���׵�
    mind studio ��װ·��/model-zoo/built-in-model/.resnet18/ ·���µ�resnet18.om 
    copy����demo��models ·�����滻ԭ�е�resnet18.om