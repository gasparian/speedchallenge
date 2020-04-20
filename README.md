Welcome to the comma.ai Programming Challenge!
======

Your goal is to predict the speed of a car from a video.

- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.

Deliverable
-----

Your deliverable is test.txt. E-mail it to givemeajob@comma.ai, or if you think you did particularly well, e-mail it to George.

Evaluation
-----

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

Twitter
------

<a href="https://twitter.com/comma_ai">Follow us!</a>  

#### Approach  

Data for pretraining:  
https://github.com/commaai/comma2k19  

Download torrent-file from [academictorrents.com](http://academictorrents.com):  
```
wget --no-check-certificate 'http://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent' -P ./data  
```  

Install torrent cli:  https://help.ubuntu.com/community/TransmissionHowTo  


