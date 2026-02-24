
To test our model, do following steps:

1. python setup.py develop --no_cuda_ext

2. Put the test data in Test_Dataset folder

Test_Dataset/
├── gt/
│   ├── 001/ (contains 10 images)
│   ├── 002/ (contains 10 images)
│   └── ...
└── input/
    ├── 001/ (contains 10 images)
    ├── 002/ (contains 10 images)
    └── ...

3. python basicsr/test.py -opt All_Weather/Options/Deraining_Test_Video_Restormer.yml --launcher none

4. Results will saved in results folder