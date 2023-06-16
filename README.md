# session7
Session 7 assignment

This repository is for the DNN arhitecture change steps on MNIST dataset to achieve 99.4 accuracy in *K parameters and within 15 epochs
I have achived the target in 3 steps below

**Step1 of architecture changes:**

**Target**: In this step I have choosen vanilla architecture of 6 convolution layer and 2 transtion blocks (maxpool) and used GAP in the the last layer. My target is if I can achieve around 99 validation accuracy then i can refine the mode further in later steps. I will run it for 20 epochs which is more than 15) just to study how the accuracy changes in vanila architecture

**Result**: I have got Train accuracy: 99.62 validation accuracy:99.09 Number of parameters: 40,202

**Analysis**: I could see that validation accuracy is steadily increasing over epochs, and finally got vlidation accuracy of 99.09 which is a good architecture to explore further. Also noticed that train accuracy 99.62 is much higher than validation accuracy 99.09, so model could be overfitting. But as number of parameters is 40,202 which is around 4 times the target parameters, I will try to reduce the parameters in next step


**Step2 of architecture changes:**

**Target**: In this step I will try to change the vanilla architecture by changing number of parameters within 10,000 . I will reduce number of kernels also remove all the bias parameters by setting bias value to False. In this step I will change the architecture from step2 by introducing Batchnormalization and Dropout. I will run for 15 epochs. My expectation is that it should increase validation accuracy from Vanilla architecture

**Result**: I have got Train accuracy: 98.73   validation accuracy: 99.22  Number of parameters: 8,582

**Analysis**:As expected validation accuracy increased to 99.22  from the step1 model, which has accuracy 99.09. I also observe that validation accuracy 99.22  is much higher than training accuracy 98.73. These could be  because of regularization effect of batch normalization and droupout introudced in this step

**Step3 of architecture changes:**

**Target**: In this step I will change the architecture from step2 by introducing Image augmentation of random rotation between -7 to +7 degrees. Also I used StepLR with step size 6 and gamma value 0.1. Also, I will change the batch size to 64 and check if the validation accuracy has stabilized or not

**Result**: I have got Train accuracy: 98.56 validation accuracy: 99.41 Number of parameters: 8,582

**Analysis**: In this step, I could see the validation accuracy is 99.41. However, when I check the last 5 epochs validation accuracies are: 99.38,99,41,99.39,99,39,99.41 which is more steady with less variance so model has stabilized. I also observe that validation accuracy 99.41 is much higher than training accuracy 98.56. These are because with image augmentation, as CNN could learn from more images. With this I have achieved all the requirements as mentioned in the assignment
