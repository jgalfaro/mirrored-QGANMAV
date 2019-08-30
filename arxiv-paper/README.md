Faking and Discriminating the Navigation Data of a Micro Aerial Vehicle Using Quantum Generative Adversarial Networks
===

### Michel Barbeau, Carleton University, School of Computer Science, Canada.

### Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, CNRS UMR 5157 SAMOVAR, Télécom SudParis, France.

<a href="https://arxiv.org/abs/1907.03038">https://arxiv.org/abs/1907.03038</a>

## Abstract
We show that the Quantum Generative Adversarial Network (QGAN)
paradigm can be employed by an adversary to learn generating data that
deceives the monitoring of a Cyber-Physical System (CPS) and to
perpetrate a covert attack. As a test case, the ideas are elaborated
considering the navigation data of a Micro Aerial Vehicle (MAV). A
concrete QGAN design is proposed to generate fake MAC navigation data.
Initially, the adversary is entirely ignorant about the dynamics of
the CPS, the strength of the approach from the point of view of the
bad guy. A design is also proposed to discriminate between genuine and
fake MAV navigation data. The designs combine classical optimization,
qubit quantum computing and photonic quantum computing. Using the
PennyLane software simulation, they are evaluated over a classical
computing platform. We assess the learning time and accuracy of the
navigation data generator and discriminator versus space complexity,
i.e., the amount of quantum memory needed to solve the problem.   

*Keywords:* Autonomous Aerial Vehicle, Micro Aerial Vehicle,
Cyber-Physical Security, Covert Attack, Photonic Quantum Computing,
Quantum Computing, Quantum Generative Adversarial Network, Quantum
Machine Learning.

## I. Introduction


Cyber-Physical Systems (CPSs) comprise physical processes monitored
and controlled through embedded computing and networked resources.
Signals to actuators and feedback from sensors are exchanged with
controllers using, e.g., wireless communication. The advantages of
such architectures include flexibility and relatively low deployment
costs. Nevertheless, the perpetration of cyber-physical attacks must
be addressed. The problem is particularly challenging when the CPS
consists of disruptive technologies such as MAVs, Unmanned Aerial
Vehicles (UAV), Unmanned Underwater Vehicles (UUV) and MAV swarming.

Today's cybersecurity solutions, from in-depth defense techniques
(e.g., firewalls) to intrusion detection and cryptographic techniques,
aim to prevent system breaches from happening. However, several
stories of attacks and disruption of CPS exist (e.g., from the
[Stuxnet worm](http://j.mp/2jaM6uM) incident affecting a Iran's
*atomic program* [7] to recent incidents in Saudi Arabia affecting
[Houthi drones](http://bit.ly/2LMqR3H) [9]). CPS protection solutions
must manage and take control over adversarial actions. Protection must
be built taking on the adversary mindset, predicting its intentions
and adequately mitigating the effects of its actions.

In this paper, we explore the use of the QGAN paradigm to address
cyber-physical security issues in the domain of MAVs. A concrete QGAN
design is proposed to generate fake MAV navigation data. Initially,
the adversary is entirely ignorant about the dynamics of the CPS. From
the point of view of the adversary, it is the strength of the
approach. A design is also proposed to discriminate between real and
fake MAV navigation data. The designs combine classical optimization,
qubit quantum computing and photonic quantum computing. We build upon
the PennyLane quantum machine learning software platform [3]. In
particular, we reuse and adapt ideas from the variational classifier
[21] and QGAN [22] examples.

We evaluate our approach using the simulation capabilities of
PennyLane. We measure the learning time and accuracy of the navigation
data generator and discriminator with respect to the space complexity,
i.e., the amount of quantum memory used to solve the problem. At the
outset, we acknowledge that the exponentially growing time complexity
in the number of qubits of our solution is a barrier to its
application on a large scale. In particular, when the calculations are
all done in simulation over a classical computing platform.
Nevertheless, we show the feasibility of the approach on a small scale
and identify hurdles that are likely to be overcome by the upcoming
evolution of quantum machine learning.

This paper has been accepted for publication in IEEE GLOBECOM 2019
Workshop on Quantum Communications and Information Technology 2019
(fifth QCIT workshop of the Emerging Technical Committee on Quantum
Communications and Information Technology, QCIT-ETC, cf.
[http://qcit.committees.comsoc.org/qcit19-workshop/](http://qcit.committees.comsoc.org/qcit19-workshop/)).

Section II elaborates further on our problem domain and related work.
Section III presents our solution. Section IV provides experimental
work. Section V concludes the paper.


## II. PROBLEM DOMAIN

The problem domain encompasses CPS controllers, playing the role of
defenders, and adversaries. We conceptualize the situation in terms of
activities consisting of gathering and hiding knowledge about both
defensive and adversarial strategies. We envision the use of new
learning theories, in which defenders and adversaries conceal their
actions to avoid being profiled for the purpose of thwarting their
cyberphysical battle weapons. Defenders equipped with Artificial
Intelligence (AI) tools, such as machine learning, can identify
adversarial actions trying to collect as much knowledge as possible
about their targets. The defense starts by learning the weaknesses of
the adversaries and offensively mislead their intentions, thwarting
their actions in the end. Once the defender knows the adversary, e.g.,
the behavior performed to identify and disrupt the services, the
defender starts offering assets sacrificed to coax the adversary and
to manage a potential security breach.

We are particularly interested in a type of CPS which function is air
space surveillance and coastal water monitoring. The application
domain of interest includes Micro Aerial Vehicles (MAVs) and related
technologies such as UAV, UUV, formations of MAVs and collaborating
MAVs. We focus on scenarios where an adversary targets the components
of the CPS and perpetrates covert cyber-physical attacks [27], [26].
The adversary is to operate a stealthy disruption of services. The
purpose is disrupting the navigation data of the MAVs and deceive the
defender. The role of the defender is to recognize the activities
performed by the adversary, i.e., identify the intentions of the
adversary and correct the adversarial actions.

### A. Covert Attack and Feedback Truthfulness

A covert attack is an aggression on the state of a CPS where the
adversary attempts to be invisible [27]. It is assumed that the
adversary knows or can learn the dynamics of the CPS. While the attack
is being carried out, the perpetrator compensates the impact of the
attack over the system by providing fake information to the system
operators (e.g., by concealing the effect of the spoofed inputs).
Hence, from the point of view of an observer, responsible for
detecting the attack, the execution of the CPS looks normal. Assume
the scenario shown in Figure 1. It depicts the disruption of the
navigation data of a series of MAVs. The manipulation is conducted by
a remote adversary via, e.g., GPS jamming and spoofing attacks [1],
[12]. The goal of the adversary is to conduct navigation data
modifications (e.g., swapping the x;y coordinates of the navigation
traces) and hide the disruption to the defender, with additional
cyber-physical covert attacks [27], [26].

![Fig. 1. MAV navigation data trace disruptions.](https://github.com/jgalfaro/mirrored-QGANMAV/blob/master/arxiv-paper/figures/fig1.png?raw=true)

## III. FAKING AND DISCRIMINATING NAVIGATION DATA

Using the Generative Adversarial Network (GAN) framework,
we validate that a covert attack can be perpetrated using
adversarial learning. A GAN consists of two main entities: a
discriminator and a generator [10]. The discriminator is the
defender’s tool. The generator is the adversary tool. There are
genuine (real) data and generated (fake) data. The generator
aims at generating data to deceive the discriminator. The
discriminator is trained with genuine and generated data.
The training process aims to a discriminator able to label
genuine or generated data correctly, with high probability
of correctness. The adversary wins the game when this
probability is at least 50%. To this end, the generator is
trained, assuming it can challenge the discriminator with data
and access to the verdict. Training is an iterative process.
Training iterates until the production of fake data is accepted
by the discriminator with high probability.

In a QGAN [5], [15] the data can be quantum. Using a
Parrot Mambo MAV, we generate genuine navigation data.
The navigation is classical and in continuous domains. Using
probability amplitude encoding, the genuine (classical) data
is mapped to quantum data and used to train a discriminator,
defined as a qubit-quantum circuit. Using a photonic-quantum
circuit, we validate that the adversary can learn to generate
fake data resembling genuine data, assuming access to nothing
else but the verdict of the discriminator.

### A. Discriminator Design

We build upon the PennyLane [3] variational classifier [21] and QGAN
[22] examples. The elementary circuit design <img
src="https://latex.codecogs.com/gif.latex?\mathcal{E}(\omega)" /> of
Farhi and Neven [8] is used, pictured in Figure 3. Every elementary
circuit processes qubits. In Figure 3, <img
src="https://latex.codecogs.com/gif.latex?$n$" /> is three. The
circuit formal parameter <img
src="https://latex.codecogs.com/gif.latex?$w$"/> is a <img
src="https://latex.codecogs.com/gif.latex?$n$"/> by three matrix of
rotation angles. For <img
src="https://latex.codecogs.com/gif.latex?$i=0,1,\ldots,n-1$"/>, the
gate <img
src="https://latex.codecogs.com/gif.latex?$Rot(\omega_{i,0},\omega_{i,1},\omega_{i,2})$"/>
applies the <img src="https://latex.codecogs.com/gif.latex?$x$"/>,
<img src="https://latex.codecogs.com/gif.latex?$y$"/> and <img
src="https://latex.codecogs.com/gif.latex?$z$"/>-axis rotations <img
src="https://latex.codecogs.com/gif.latex?$\omega_{i,0}$"/>,
<img src="https://latex.codecogs.com/gif.latex?$\omega_{i,1}$"/>,
<img src="https://latex.codecogs.com/gif.latex?$\omega_{i,2}$"/> to qubit
<img src="https://latex.codecogs.com/gif.latex?$\left\vert{\psi_i}\right\rangle$"/>. The
three rotations can take a qubit from any state to any state. For
entanglement purposes, qubit <img
src="https://latex.codecogs.com/gif.latex?$i$"/> is connected to qubit
<img src="https://latex.codecogs.com/gif.latex?$i+1$"/> modulo <img
src="https://latex.codecogs.com/gif.latex?$n$"/> using a CNOT gate.

![Fig. 3. Three-qubit elementary circuit layer.](https://github.com/jgalfaro/mirrored-QGANMAV/blob/master/arxiv-paper/figures/fig3.png?raw=true)

### B. Generator Design

*Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
minim veniam, quis nostrud exercitation ullamco laboris nisi ut
aliquip ex ea commodo consequat. Duis aute irure dolor in
reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
pariatur. Excepteur sint occaecat cupidatat non proident, sunt in
culpa qui officia deserunt mollit anim id est laborum.*


## IV. PERFORMANCE

The performance of the photonic-circuit design described in Section
III has been validated through simulation on a classical computing
platform. Simulations were conducted using an Intel Xeon 32-core 2.70
GHz server, with 256 GB of memory. We generated genuine navigation for
a Parrot Mambo MAV. In the scenario, the MAV takes off one meter, does
two circles on the horizontal plane, then lands. The navigation data
consists of x, y and z velocity triples. The whole scenario generates
less than 64 real number values.

Figure 8 plots the discriminator and generator learning time (ms)
versus the number of qubits available. The x axis represents the
number of qubits. The left y axis refers to the learning time (ms).
The right y axis shows the corresponding probability of real true, for
the discriminator, and probability of fake true, for the generator.
Hundred optimization iterations were done for each case. Negligible
error margins are included, but not visible since they are very tiny.
The discriminator is trained with six different genuine navigation
data sets. A navigation data set is picked at random at every
optimization iteration. The discriminator optimization time grows
exponentially. Due to the exponential complexity of the generator
circuit (in <img
src="https://latex.codecogs.com/gif.latex?$\mathcal{O}(2^{n})$"/>),
the optimization time also grows exponentially. On our simulation
platform, it becomes unpractical from six qubits. The learning time
becomes in the order of days. Amplitude encoding has also <img
src="https://latex.codecogs.com/gif.latex?$\mathcal{O}(2^{n})$"/> time
complexity, but it is only executed once at the start of the
optimization process.

V. CONCLUSION
============

We have investigated the use of QGAN designs to generate fake MAV
navigation data. We assume the same approach to discriminate between
genuine and fake MAV navigation data. The goal pursued by the
adversary is to generate fake data that is accepted as true by a
trained discriminator. On the other hand, the discriminator must
accept with high probabilities true navigation data and reject fake
one. The elaborated quantum circuits have been evaluated running on a
a classical computing platform. As demonstrated in Figure 8, the
exponentially growing time complexity in the number of qubits is an
obstacle to scalability. We identified hurdles that must be overcome
by the upcoming evolution of quantum machine learning. The main hurdle
for the adversary is the generation of navigation data in classical
continuous domains, i.e., real numbers, and the cost of the
transformation into the quantum format at every optimization
iteration. Further research is needed to improve and find alternatives
to the design depicted in Figure 7.



REFERENCES
============

[1] Michel Barbeau, Joaquin Garcia-Alfaro, and Evangelos Kranakis.
Geocaching-inspired resilient path planning for drone swarms. In IEEE
MiSARN 2019, co-located with IEEE INFOCOM 2019 – IEEE Conference on
Computer Communications, France, 2019.

[2] Marco Barreno, Blaine Nelson, Anthony D Joseph, and J Doug Tygar.
The security of machine learning. Machine Learning, 81(2):121–148,
2010.

[3] Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin,
Carsten Blank, Keri McKiernan, and Nathan Killoran. Pennylane:
Automatic differentiation of hybrid quantum-classical computations,
2018.

[4] Samuel Rota Bulo, Battista Biggio, Ignazio Pillai, Marcello
Pelillo, and Fabio Roli. Randomized prediction games for adversarial
machine learning. IEEE transactions on neural networks and learning
systems, 28(11):2466–2478, 2016.

[5] Pierre-Luc Dallaire-Demers and Nathan Killoran. Quantum generative
adversarial networks. Phys. Rev. A, 98:012324, Jul 2018.

[6] Claudio De Stefano, Carlo Sansone, and Mario Vento. To reject or
not to reject: that is the question-an answer in case of neural
classifiers. IEEE Transactions on Systems, Man, and Cybernetics, Part
C (Applications and Reviews), 30(1):84–94, 2000.

[7] Nicolas Falliere, Liam Murchu, and Eric Chien. W32.Stuxnet
dossier, symantec security response.
[http://j.mp/2jaM6uM](http://j.mp/2jaM6uM), 2011.

[8] Edward Farhi and Hartmut Neven. Classification with quantum neural
networks on near term processors, 2018.

[9] Alex Gatopoulos. Houthi drone attacks in saudi show new level of
sophistication. [http://j.mp/2LMqR3H](http://j.mp/2LMqR3H), May 2019.

[10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David
Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio.
Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes,
N. D. Lawrence, and K. Q. Weinberger, editors, Advances in Neural
Information Processing Systems 27, pages 2672–2680. Curran Associates,
Inc., 2014.

[11] Vojtech Havlıcek, Antonio D. C´orcoles, Kristan Temme, Aram W.
Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta.
Supervised learning with quantum-enhanced feature spaces. Nature,
567(7747):209–212, 2019.

[12] Ahmad Y Javaid, Farha Jahan, and Weiqing Sun. Analysis of global
positioning system-based attacks and a novel global positioning system
spoofing detection/mitigation algorithm for unmanned aerial vehicle
simulation. Simulation, 93(5):427–441, 2017.

[13] Murat Kantarcıoglu, Bowei Xi, and Chris Clifton. Classifier
evaluation and attribute selection against active adversaries. Data
Mining and Knowledge Discovery, 22(1-2):291–335, 2011.

[14] Nathan Killoran, Josh Izaac, Nicolas Quesada, Ville Bergholm,
Matthew Amy, and Christian Weedbrook. Strawberry Fields: A
Software Platform for Photonic Quantum Computing. Quantum, 3:129,
March 2019.

[15] Seth Lloyd and Christian Weedbrook. Quantum generative adversarial
learning. Phys. Rev. Lett., 121:040502, Jul 2018.

[16] MathWorks. Quadcopter Project.
[mathworks-quadcopter-project.html](https://www.mathworks.com/help/aeroblks/quadcopter-project.html).
Accessed: 2019-06-20.

[17] Nicolas Papernot, Patrick McDaniel, Arunesh Sinha, and Michael
Wellman. Towards the science of security and privacy in machine
learning. arXiv preprint [arXiv:1611.03814](https://arxiv.org/abs/1611.03814), 2016.

[18] Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram
Swami. Distillation as a defense to adversarial perturbations
against deep neural networks. In 2016 IEEE Symposium on Security
and Privacy (SP), pages 582–597. IEEE, 2016.

[19] Gianluigi Pillonetto, Francesco Dinuzzo, Tianshi Chen, Giuseppe
De Nicolao, and Lennart Ljung. Kernel methods in system identification,
machine learning and function estimation: A survey. Automatica,
50(3):657–682, 2014.

[20] Gianluigi Pillonetto and Giuseppe De Nicolao. A new kernel-based
approach for linear system identification. Automatica, 46(1):81–93,
2010.

[21] Maria Schuld. Example - Variational classifier.
[https://github.com/XanaduAI/pennylane/blob/master/examples/](https://github.com/XanaduAI/pennylane/blob/master/examples/).
Accessed: 2019-06-11.

[22] Maria Schuld. Example - Quantum Generative Adversarial Network.
[https://github.com/XanaduAI/pennylane/blob/master/examples/](https://github.com/XanaduAI/pennylane/blob/master/examples/).
Accessed: 2019-06-11.

[23] Maria Schuld and Nathan Killoran. Quantum machine learning in
feature hilbert spaces. Phys. Rev. Lett., 122:040504, Feb 2019.

[24] Maria Schuld and Francesco Petruccione. Supervised Learning with
Quantum Computers. Quantum science and technology. Springer,
2018.

[25] John Shawe-Taylor and Nello Cristianini. Kernel Methods for Pattern
Analysis. Kernel Methods for Pattern Analysis. Cambridge University
Press, 2004.

[26] Roy Smith. Covert Misappropriation of Networked Control Systems:
Presenting a Feedback Structure. IEEE Control Systems, 35(1):82–92,
Feb 2015.

[27] Andre Teixeira, Iman Shames, Henrik Sandberg, and Karl Henrik
Johansson. A secure control framework for resource-limited
adversaries. Automatica, 51:135–148, 2015.
