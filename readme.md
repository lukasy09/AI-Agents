<h1>1.Introduction</h1>
<p>This repository contains all my projects related to Reinforcement Learning and creating agents' AI</p>
<p>For those, I'm using openAI environments and creating my own solutions for the problems


<h1>2.Projects</h1>

<h2>2.1 Cartpole</h2>

<p>Simple control agent which is to keep "a wooden bar" in a vertical position</p>
<p>The agent can take 2 possible actions. It is capable of putting a unit force to the left and the right (-1, 1)</p>
<p>The agent's task is to navigate the object to the left or the right side keeping it vertically(maximal possible angle is 15 degrees)</p>

<h3>2.1.1 This agent taking some random decisions(actions)</h3>
<p align = "center" >
	<img src = "/assets/cartpole_random.gif"/>
</p>

<h3>2.1.2 The following agent is trained and tries to keep taking good decisions</h3>
<p align = "center" > <img src = "/assets/cartpole_trained.gif"/></p>


<h2>2.2 Classic pendulum </h2>

<p>An agent which task is to try to `stay upright`. The agent is capable of giving itself angular velocity in range [-2, 2] (rad/s).  </p>
<p>The starting state is defined by 3 numbers - sine, cosine in range [-1, 1] and `theta dot` (dθ/dt - angular velicty [rad/s])</p>

<h3>2.2.1 Untrained agent, doing random things</h3>
<p align = "center">
	<img src = "/assets/pendulum_random.gif"/>
</p>



<h3>2.2.2 Trained agent, trying to do something reasonable</h3>
<p align = "center">
	<img src = "/assets/pendulum_trained.gif"/>
</p>