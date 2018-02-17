import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the environment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.resetCount = 0

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        self.resetCount +=1
        
        if testing:
            self.epsilon = 0.0
            self.alpha = 0.0
        else:
            if True:
                #self.epsilon_decay_pw()
                self.epsilon_decay_e()
                #self.epsilon_decay_recip()
                #self.alpha_decay_e()
                self.alpha_decay_constant()
                #self.alpha_decay_lin()
                #self.alpha_decay_epsilon()
            else:
                self.epsilon_decay_noOptimization()
                self.alpha_decay_constant()
        
        if self.epsilon < 0.0:
            self.epsilon = 0.0
        elif self.epsilon > 1.0:
            self.epsilon = 1.0

        return None
        
        
    def epsilon_decay_noOptimization(self):
        self.epsilon -= 0.050
    
    def epsilon_decay_sawtooth(self):
        self.epsilon -= 0.020
        if self.resetCount%20 == 0:
            self.epsilon += 0.35
        


    def epsilon_decay_pw(self):
        if self.resetCount < 10:
            self.epsilon = 0.95
        elif self.resetCount < 20:
            self.epsilon = 0.5
        elif self.resetCount < 30:
            self.epsilon = 0.5
        elif self.resetCount < 40:
            self.epsilon = 0.5
        elif self.resetCount < 50:
            self.epsilon = 0.5
        elif self.resetCount < 60:
            self.epsilon = 0.5
        elif self.resetCount < 70:
            self.epsilon = 0.5
        elif self.resetCount < 80:
            self.epsilon = 0.5
        elif self.resetCount < 90:
            self.epsilon = 0.5
        elif self.resetCount < 100:
            self.epsilon = 0.5
        elif self.resetCount < 110:
            self.epsilon = 0.5
        elif self.resetCount < 120:
            self.epsilon = 0.1
        elif self.resetCount < 130:
            self.epsilon = 0.1
        elif self.resetCount < 140:
            self.epsilon = 0.1
        elif self.resetCount < 150:
            self.epsilon = 0.1
        else:
            self.epsilon = 0.0
        
            
    def epsilon_decay_e(self):
        self.epsilon = math.e**(-0.0015 * self.resetCount) # -0.02 is 150 trials, -0.015 is 200 trials

    def epsilon_decay_recip(self):
        self.epsilon = 10.0/(self.resetCount + 10.0);
        
        
    def alpha_decay_constant(self):
        self.alpha = 0.5

    def alpha_decay_e(self):
        self.alpha = math.e**(-0.025 * self.resetCount)
    
    def alpha_decay(self):
        #alpha formula
        self.alpha -=0.01
        if self.alpha < 0.3:
            self.alpha = 0.3

    def alpha_decay_lin(self):
        a0 = 0.8
        m = -0.6/200 # <total decrease> / <total distance>
        self.alpha = m*self.resetCount + a0
        
    def alpha_decay_epsilon(self):
        self.alpha = self.epsilon
            
    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        state = (inputs['light'], inputs['left'], inputs['right'], inputs['oncoming'], waypoint)

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """
        
        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        # maxKey = max(self.Q[state], key=self.Q[state].get)
        maxVal = max(self.Q[state].values())
        maxKeyList = [k for k,v in self.Q[state].items() if v==maxVal]
        maxKeyRandom = random.choice(maxKeyList)
        # (max key, max val)
        # key: action, random if same Q value as other action
        # val: Q value
        maxQ = (maxKeyRandom, maxVal)
        
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        
        if state in self.Q:
            pass
        else:
            self.Q[state] = {None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0}
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action, Qval = None, None
        
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        
        #valid_actions = [None, 'forward', 'left', 'right']
        
        if self.learning == True :
            p = random.uniform(0.0, 1.0)
            if(p < self.epsilon):
                action = random.choice(self.valid_actions)	
            else :
                action, Qval = self.get_maxQ(self.state)
        else :
            action = random.choice(self.valid_actions)		
        
        
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        
        #Q(s, a) = (1-alpha)*Q(s, a) + alpha*(reward + gamma*maxQ(nextState, nextAction))
        
        self.Q[state][action] = (1-self.alpha) * self.Q[state][action] + self.alpha*reward
        
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.01, log_metrics=True, display=False, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=50, tolerance=0.05)


if __name__ == '__main__':
    run()
