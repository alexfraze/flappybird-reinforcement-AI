import numpy as np
import random

random.seed(803)
np.random.seed(803)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop


"""
Here are two values you can use to tune your Qnet
You may choose not to use them, but the training time
would be significantly longer.
Other than the inputs of each function, this is the only information
about the nature of the game itself that you can use.
"""
PIPEGAPSIZE  = 100
BIRDHEIGHT = 24

class QNet(object):

	def __init__(self):
		"""
		Initialize neural net here.
		You may change the values.

		Args:
			num_inputs: Number of nodes in input layer
			num_hidden1: Number of nodes in the first hidden layer
			num_hidden2: Number of nodes in the second hidden layer
			num_output: Number of nodes in the output layer
			lr: learning rate
		"""
		self.num_inputs = 2
		self.num_hidden1 = 10
		self.num_hidden2 = 10
		self.num_hidden3 = 10
		self.num_output = 2
		self.lr = 0.01 #default 0.001
		self.build()

		self.scored = False
		self.crashed = False
		self.mini_batch_training = []	
		self.mini_batch_targets = []

	def build(self):
		"""
		Builds the neural network using keras, and stores the model in self.model.
		Uses shape parameters from init and the learning rate self.lr.
		You may change this, though what is given should be a good start.
		"""
		model = Sequential()
		model.add(Dense(self.num_hidden1, init='lecun_uniform', input_shape=(self.num_inputs,)))
		model.add(Activation('relu'))

		model.add(Dense(self.num_hidden2, init='lecun_uniform'))
		model.add(Activation('relu'))

		model.add(Dense(self.num_hidden3, init='lecun_uniform'))
		model.add(Activation('relu'))

		model.add(Dense(self.num_output, init='lecun_uniform'))
		model.add(Activation('linear'))

		rms = RMSprop(lr=self.lr)
		model.compile(loss='mse', optimizer=rms)
		self.model = model


	def flap(self, input_data):
		"""
		Use the neural net as a Q function to act.
		Use self.model.predict to do the prediction.

		Args:
			input_data (Input object): contains information you may use about the 
			current state.

		Returns:
			(choice, prediction, debug_str): 
				choice (int) is 1 if bird flaps, 0 otherwise. Will be passed
					into the update function below.
				prediction (array-like) is the raw output of your neural network,
					returned by self.model.predict. Will be passed into the update function below.
				debug_str (str) will be printed on the bottom of the game
		"""

		# state = your state in numpy array
		state = np.array([input_data.distX, input_data.distY])
		prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size=1)[0]
		#choice = make choice based on prediction
		choice = np.argmax(prediction)
		# print(prediction)
		debug_str = str(-abs(float(input_data.distY)-70))
		return (choice, prediction, debug_str)
		

	def update(self, last_input, last_choice, last_prediction, crash, scored, playerY, pipVelX):
		"""
		Use Q-learning to update the neural net here
		Use self.model.fit to back propagate

		Args:
			last_input (Input object): contains information you may use about the
				input used by the most recent flap() 
			last_choice: the choice made by the most recent flap()
			last_prediction: the prediction made by the most recent flap()
			crash: boolean value whether the bird crashed
			scored: boolean value whether the bird scored
			playerY: y position of the bird, used for calculating new state
			pipVelX: velocity of pipe, used for calculating new state

		Returns:
			None
		"""
		# This is how you calculate the new (x,y) distances
		new_distX = last_input.distX + pipVelX
		new_distY = last_input.pipeY - playerY

		# state = compute new state in numpy array
		state = np.array([new_distX, new_distY])
		
		# reward = compute your reward
		alive = not(crash)
		bird_dist_above_pipe = 55 #give the bird some room to pass over pipe
		reward_alive = 0#3*(not(crash))
		reward_dead = -300*crash
		reward_dead_y = 0#-30*crash*abs(float(new_distY)-bird_dist_above_pipe)*(new_distX+10)/10
		reward_score = 500*scored
		reward_y_distance = -0.01*((float(new_distY)-bird_dist_above_pipe)**2)
		reward_y_positive = 0 #500/(abs(float(new_distY)-bird_dist_above_pipe)+1) 	
		reward_yx_interaction = 3*reward_y_distance*(float(1)/(new_distX+1))
		reward_not_flap = 100 if last_choice==0 and new_distY > bird_dist_above_pipe else 0
		reward_flap = 100 if last_choice==1 and new_distY < bird_dist_above_pipe  else 0
		penalty_not_flap = -100 if last_choice==0 and new_distY < bird_dist_above_pipe else 0
		penalty_flap = -100 if last_choice==1 and new_distY > bird_dist_above_pipe  else 0
		
		# print(reward_dead, reward_alive, reward_dead_y,  reward_y_distance,  reward_yx_interaction,
		# reward_score, reward_y_positive, reward_flap, reward_not_flap, penalty_flap, penalty_not_flap)
		reward = (reward_dead + reward_alive + reward_dead_y + reward_y_distance + reward_yx_interaction + 
			reward_score + reward_y_positive + reward_flap + reward_not_flap + 
			penalty_not_flap + penalty_flap)

		# reward = reward/5 #scale it smaller
		new_state_prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size = 1)

		# update old prediction from flap() with reward + gamma * np.max(prediction)
		gamma = 0.7 #value later rewards fairly significantly. 
		updated_state_prediction = reward+gamma*np.max(new_state_prediction)
		learning_rate = 0.2 #do i need this second learning rate or is the NN arleady doing this for me?
		
		last_prediction[last_choice] = ((1-learning_rate)*last_prediction[last_choice] + 
		learning_rate*updated_state_prediction)
		# record updated prediction and old state in your mini-batch
		old_state = np.array([last_input.distX, last_input.distY])
		updated_prediction = last_prediction

		# self.mini_batch_training = np.append(self.mini_batch_training, old_state)
		# self.mini_batch_targets = np.append(self.mini_batch_targets,updated_prediction)

		self.mini_batch_targets.append(updated_prediction)
		self.mini_batch_training.append(old_state)

		# if(crash): self.crashed = True
		# if(scored): self.scored = True
		# if batch size is large enough, back propagate
		batch_size = len(self.mini_batch_targets)
		sub_batch_size = 150
		if(batch_size >= 500):
			epochs = 10
			# if(self.scored): 
			# 	epochs += 500
			# if(self.crashed):
			# 	epochs -= 100 
			print("back_propagating")
			indexes = xrange(sub_batch_size)
			indexes = np.random.choice(batch_size, sub_batch_size, replace=False)
			# indexes = xrange(batch_size)
			# print(self.mini_batch_training)
			# print(np.array(self.mini_batch_targets)[indexes])
			# print(np.array(self.mini_batch_targets))
			self.model.fit(np.array(self.mini_batch_training)[indexes], 
				np.array(self.mini_batch_targets)[indexes], batch_size=sub_batch_size, epochs=epochs)
			#drop off the first element in training samples.

			#pick 1 element to delete from training sample. 
			sample_to_forget = indexes[0] #TA told other kids to forget one of used samples.
			del self.mini_batch_targets[sample_to_forget]
			del self.mini_batch_training[sample_to_forget]
			# self.crashed = False
			# self.scored = False
				  
		
class Input:
	def __init__(self, playerX, playerY, pipeX, pipeY,
				distX, distY):
		"""
		playerX: x position of the bird
		playerY: y position of the bird
		pipeX: x position of the next pipe
		pipeY: y position of the next pipe
		distX: x distance between the bird and the next pipe
		distY: y distance between the bird and the next pipe
		"""
		self.playerX = playerX
		self.playerY = playerY
		self.pipeX = pipeX
		self.pipeY = pipeY
		self.distX = distX
		self.distY = distY

