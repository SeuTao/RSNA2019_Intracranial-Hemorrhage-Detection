# RSNA Intracranial Hemorrhage Detection
This is the source code for the first place solution to the [RSNA2019 Intracranial Hemorrhage Detection Challenge](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection). 
Video with quick overview: 

## Solutuoin Overview
![image](https://github.com/SeuTao/RSNA2019_1st_place_solution/blob/master/png/overview.png)

#### Dependencies
- opencv-python==3.4.2
- scikit-image==0.14.0
- scikit-learn==0.19.1
- scipy==1.1.0
- torch==1.1.0
- torchvision==0.2.1

### code
- 2dCNN
- SequenceModel

After single models training,  the oof files will be saved in ./SingleModelOutput(three folders for three pipelines). For the final sequence model training:
After training the sequence model, the final submission will be ./FinalSubmission/final_version/submission_tta.csv

## Final Submission
### Private Leaderboard:
- 0.04383