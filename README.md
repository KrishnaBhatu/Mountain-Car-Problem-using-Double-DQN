# Double Deep Q Learning

## Dependencies Required
 - Tensorflow version 1.4.0
 - Keras version 2.2.5 

## Running Instructions
 1. To run the code execute command 
 `$ python problem1_sol.py` 
 2. To train the model from scratch uncomment line 34 and 35 and comment line 39 and 40. In this case it is recommended to set NUM_EPISODES at least 10000.

## File structure and outputs
 1. problem1_sol.py is the script to run the code
 2. best_one.h5 is the model we trained
 3. /Graph\ Outputs/ directory contains the output graphs of rewards and running averages
 4. /Video\ Outputs/ directory contains the output videos recorded at three different instances:
	1. Start of learning.mp4 - recorded when the training was initiated
	2. Middle of learning.mp4 - recorded from around 5501th episode during the training
	3. End of learning.mp4 - recorded from around 9844th episode during the training  

*Note: Refer [Report](https://github.com/KrishnaBhatu/Mountain-Car-Problem-using-Double-DQN/blob/master/ReportHW3.pdf) for deeper understanding of the project*
