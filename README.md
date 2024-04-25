# Hair_Care_Recommendation_Chatbot
## Git clone
```
git clone https://github.com/Gnort1002/Hair_Care_Recommendation_Chatbot.git
cd Hair_Care_Recommendation_Chatbot
git checkout test
cd HairFastGan
```
## Download weight
Download weight for HairFastGan here
[Link](https://drive.google.com/drive/folders/1MTTik9uSUc6JJOcykn2za3CtU3mA3c9G?usp=sharing)

Download weight for SwinFace here
[Link](https://drive.google.com/file/d/1-iTuA7gaepNT94gADqwotfJNP35F6Lpp/view?usp=sharing)

Download weight for alignment here

## Format weight use in project
```
|-HairFastGAN
             |--alignment
                         |---shape_predictor_68_face_landmarks.dat
             |--swinface_project
                         |---checkpoint_step_79999_gpu_0.pt
             |--pretrained_models (weight for HairFastGan)

```
## Run
See file demo.ipynb to run
