Follow given steps to get the results:

1. Open this folder in VS STUDIO
2. Ctrl + `
3. conda activate RESTORMER
4. $env:KMP_DUPLICATE_LIB_OK="TRUE"
5. > python setup.py develop --no_cuda_ext
5. For training > python basicsr/train.py -opt Deraining/Options/Deraining_Video_Restormer.yml --launcher none
6. For testing > python basicsr/test.py -opt Deraining/Options/Deraining_Test_Video_Restormer.yml --launcher none

7. You can change the checkpoint in the file Deraining_Test_Video_Restormer.yml which is located at D:\NAVEEN\WorkDone\ICME\Code\Deraining\Options\Deraining_Test_Video_Restormer.yml from
pretrain_network_g: ./experiments/Deraining_Restormer_Video_Clip_Train/models/net_g_576200.pth

