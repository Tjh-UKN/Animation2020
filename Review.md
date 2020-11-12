# 基于神经网络的人物动作捕捉和角色动作生成文献综述
## 1 背景




## 2 里程碑

### 2.1 相位函数神经网络(PFNN)

Daniel Holden等人提出的PFNN(Phase-Functional Neural Network, 相位函数神经网络)，主要针对于虚拟角色的实时动作控制。由于这个问题有太多需求需要满足，因而这个问题还是比较具有挑战性的。而当情况更复杂时，例如环境由不平坦的地形和大的障碍物组成时，该问题甚至更具有挑战性。

在Holden的论文中，主要贡献是提供了一种基于神经网络的实时运动合成框架PFNN和为PFNN准备训练数据的过程。PFNN的工作原理是在每帧生成一个回归网络的权重，以作为相位的函数。生成以后，该权重将会作为该帧的控制参数以控制角色运动。PFNN的设计避免了明确地混合几个相位的数据，而是通过建立了一个随时间平滑演变的回归函数来实现。与不同的神经模型相比，PFNN具有以下优势：

1. 不同于CNN这样的离线模型，PFNN是在线实时运动生成的模型。
2. 不同于RNN模型，PFNN能够生成更加稳定且在与用户交互的复杂环境下更高质量的运动。
3. PFNN运行速度很快而且所需内存很小，只需要几毫秒的执行时间和几MB的内存。

同时在论文中，笔者也提出了PFNN模型的局限性。

1. 系统为了实现实时性能忽略了一些高分辨率细节，例如地形中尖锐细小的障碍物。
2. 本模型无法很好地处理对环境与地形的复杂交互，尤其是精细的手部动作，例如爬上墙壁或与其他对象交互。
3. 由于minibatch中地每个元素产生不同的权值，从而导致训练过程中计算的代价更大，所有PFNN的训练速度相对较慢。
4. 如果用户提供的输入在给定环境中无法实现（例如地形太陡），系统会产生不合需求的结果。
5. 模型结果可能会难以预测，因此难以修改或编辑。

### 2.2 模式自适应神经网络(MANN)

四足动物动画是计算机动画中一个未解决的关键问题。它不仅同计算机游戏和电影的实现有关，也是机器人学中一个具有挑战性的主题。在控制数据获取中，四足动物不能像人类一样被引导，这是一大难题。（就意味着动作的数据样本不是按人为意愿控制的）所以获得的数据经常是无序的，连续动作也是随机的。
混合专家模式是一个传统的机器学习方法，大量的experts被用来处理不同地带的输入。一个门网络决定了对于给定的输入，哪些专家会被使用。训练完成后，这些专家会专用于被门网络分配的下降维度。它和深度学习结构的组合显示出极好的前景。基于传统机器学习技术的动作合成：如K-Nearnest Neighbours（KNN），主成分分析(PCA)，径向基数函数(RBF)，强化学习，高斯处理(GP) 等深度学习技术用于改善不同动作的过渡以及提升输出运动的普适性，最好是生成全新的动作而不是仅仅模仿已有的动作数据。2018年HE ZHANG†， SEBASTIAN STARKE基于神经网络学习和混合专家模型提出了MANN神经网络（四足动物运动控制中的模态自适应神经网络），用于四足动物的动作生成和预测。
该系统由门控网络和动作预测网络组成，门控网络负责接收动作的特征x和混合expert weights动态更新预测网络权重，（混合专家模式是一个传统的机器学习方法，大量的experts被用来处理不同地带的输入。一个门网络决定了对于给定的输入，哪些专家会被使用。训练完成后，这些专家会专用于被门网络分配的下降维度。它和深度学习结构的组合显示出极好的前景。）动作预测网络负责计算当前帧的状态（即上一个动作的下一个状态），所用到的权重由门控网络计算。

MANN的贡献在以下几个方面：
1.第一个系统性的数据驱动的生成高质量复杂四足动作模式以及转换的角色控制器。
2.灵活性高，这个系统可以通过无结构的动作捕获数据，持续学习一系列周期和非周期的动作的expert weight，同时它不需要相位标记和步态标记。


### 2.3 神经状态机(NSM)

基于场景理解的精确控制能够很好地帮助人物角色实现自我定位和导航，从而到达设定的目标位置。Holden等人提出了神经状态机(NSM)，一种引导角色通过精确场景交互实现目标驱动行为的框架。Holden等人在2017年提出的相位函数神经网络(PFNN)通过监督学习进行建模，生成高质量的人物动作并控制角色在有限区域内的运动。神经状态机将高质量的动作生成应用于支持人物角色与环境的交互，基于高阶的场景理解控制角色实现一系列复杂的人物动作。

神经状态机的贡献主要体现在以下几点：
1. 神经状态机构造出一种信号，将动作的相位编码成为高级动作描述标签和目标位置，通过数据集的深度学习，神经网络能够根据高级的指令生成高质量的人物角色动作。神经状态机以一种端到端的方式从数据集中分解权重，改良了传统的使用固定相位函数方法所带来的局限性，即只能应用于周期性的人物动作。
2. 神经状态机实现了一种双向控制框架，该框架综合了第一人称视角和目标视角对人物角色动作的预测，预测结果被实时地反馈给神经网络，从而产生连续的、高精确度的角色运动轨迹。
3. 神经状态机将体积表示方法应用于环境理解，改良了传统等高线表示方法所带来的局限性。该方法增强了了角色与凹形物体的交互效果，
4. 作者设计了一种增强数据集的方案，通过在数据集的每一帧中随机地切换环境变量中的几何物体，且不改变动作和交互的连贯性，使得神经网络获得更大的学习量，而不需要增大数据集的数据量。



## 3 相关工作

### 1. 数据驱动的运动合成技术

**1.1 基于线性模型的方法**

1. 在Howe等人1999年的论文和Safonova等人2004年的论文中都提到了主成分分析(PCA)用于降低运动数据的维数并用于从较少数量的输入来预测全身运动。
2. 关于PCA的应用。Chai和Hodgins在2005年采用局部PCA来合成具有稀疏标记集的全身运动；Tautges等人在2011年利用稀疏惯性传感器产生的类似局部结构来预测全身运动。
这些采用局部或全局PCA的方法在训练和运行时需要大量的计算和预处理。

**1.2 基于内核的方法**可以用来克服线性方法的局限性并考虑数据的非线性。基于径向函数(Radial Basis Functions, RBF)和高斯过程(Gaussian Process, GP)是混合不同类型运动的常用方法。

1. Grochow等人于2004年应用了高斯过程潜在变量模型(Gaussian Process Latent Variable Models, GPLVM)来计算数据的低维度潜在空间以帮助解决反向运动的冗余问题。之后Levine等人在2012年运用GPLVM来提升计划移动的效率。
2. Wang等人于2008年提出了高斯过程动态模型(Gaussian Process Dynamic Model, GPDM),GPDM的工作原理是学习潜在空间的动力学并将动作通过另一个GP投影到整个空间中。
3. 上面这些基于内核的方法在存储和求协方差矩阵的逆矩阵时需要较高的内存成本，二者存储代价规模分别为$N^2$和$N^3$(其中N为数据点数)。Rasmussen和Ghahramani在2002年提出了局部GP方法来限制插值的样本数来解决这个问题。

**1.3 基于神经网络的方法**由于其高可扩展性和高运行效率而成为一个热点。

1. 其中一类模型是基于角色的先前运动姿态来预测角色的当前姿态，被称为自回归模型。Taylor等人于2009年提出的cRBM模型、Fragkiadaki等人于2015年提出的应用潜在空间的LSTM模型的RNN模型都属于自回归模型。这些自回归模型相较于线性方法和内核方法具有更大的规模和更高的运行效率。但是由于噪声和低拟合度的数据积累，运动会逐渐偏离运动流形，无法保证其稳定性。
2. Holden等人于2016年改为使用CNN模型，将低维用户信号映射到全身的动作中，这种算法稳定性很好但是它是一个离线框架，需要在合成运动之前指定沿时间线的完整控制信号。

### 2. 与环境的交互的自动角色控制方法

这些控制方法可以分为两类，基于优化的方法和形状匹配。

**2.1 基于优化的方法**

1. 基于优化的方法[Lau和Kuffner于2005年，Safonova和Hodgins于2007年],基于抽样的方法[Coros等人于2008年，Liu等人于2010年],最大后验概率(MAP)估计[Chai和Hodgins于207年，Min和Chai于2012你年],强化学习方法[Lee和Lee于2004年，Lee等人于2010年，Levine等人于2012年，Lo和Zwicker于2008年，彭等人于2016年]都可以根据角色的当前状态（包括姿势）和它与环境集合的关系来预测运动。它们需要cost/value函数来给估计在不同情况下的每一种动作。其局限性在于计算代价是动作数的指数级的因此可扩展性不是很好。
2. 根据Clavet2016年的论文和Lee等人2010年的论文，在样本中进行k近邻搜索推动运动是可扩展性方面的限制因素。Levine等人与2012年通过在潜在空间中使用GPLVM进行强化学习的方式解决了这个问题，但该方法需要将动作分类，在每一个类中进行搜索。
3. Peng等人在2016年将强化学习应用于基于物理的动画控制空间，以处理高维状态空间。但该系统只在简单的2D环境中测试。

**2.2 形状匹配方法**是对环境进行集合分析，以使姿态或动作适应新的几何环境。

1. Lee等人于2006年进行刚性形状匹配，以适应接触丰富的运动，如坐在椅子上或躺在不同的几何形状。Grabner等人于2011年进行了强力搜索，以发现场景几何中角色可以进行特定动作的位置。古普塔等人于2011年制作了一个以各种姿势占据角色的体积，并将其放入由照片构建的虚拟曼哈顿世界。Kim等人于2014年提出利用物体的各种几何特征来使角色姿势适合不同的几何形状。Kang等人于2014年分析了人体的开放体积和身体舒适度，以预测角色在每种姿势下的静态位置。Savva等人于2016年使用微软Kinect捕捉人类与对象的交互，并生成可用于合成新场景的统计模型。然而上述这些方法只能处理静态姿势。
2. Kapadia等人于2016年估计了动态运动过程中身体与环境之间的接触，并利用这些接触将人体运动融入虚拟环境中。这种方法需要对图形作预先深入分析，不允许在角色第一次面对的新环境中进行实时交互。

### 3. 实现将用户参数映射到潜在变量中

事实上人们更喜欢将场景参数如视点和光照条件映射到潜在变量中，这样用户就可以在合成过程中控制它们。然而Kulkarni等人于2015年提出了一种方法，将视点和人脸图像的光照条件映射到可变自动编码器的隐藏单元中；Memisavic于2013年提出了一种乘法网络，直接将潜在变量（视点）参数化为神经网络的权重，这样当潜在参数对整体输出有全局影响时，这种网络非常有效。这些方法可以被用在动作预测中并采用相位作为所有运动类型的公共参数。

### 4. 基于物理的动作生成技术
**4.1 基于轨迹的方法**

吉拉德和Maciejewski早期工作[1985]建议使用的步态模式，脚的位置样条，逆运动学，和身体的位置，简化体动力学的制约。
布隆伯格和Galyean[1995]培养作为模拟一个多层的运动方法电机系统的狗，重点支持更高层次的行为。
[海克尔等，2008]游戏的发展产生任意足动物，包括运动模式的程序动画的方法。
Torkos和VAN+DE+Panne[1998]轨迹优化技术应用于一个抽象的quadrupedmodel获得与脚位置和计时模式兼容的议案。
Wampler和Popovic通过最优化扭转力和步态循环中约束条件的组合的方式计算不同类型的的动物的动作。
W扩展了这一方法，并用来预测外表已知而动作未知的动物的行动。这些方法不能应用到在线性应用上。
Levine和Popovic以类似物理的方式让狗的运动适应动态改变的地形。这种方法对使少量的典型动作适应不同环境很有用，但需要一系列动作作为先导，这个方法发展出了由数据驱动的方法。

**4.2 基于扭转力的方法**

Raibert和Hodgins设计出了小跑，反射和飞奔步态的控制策略。
Van和Panne用速度作为矩阵最优化控制图的参数来控制一个猫的模型。
Coros【Locomotion skills for simulated quadrupeds. ACM Transactions on Graphics 30, 4 (2011), 】为四足动物设计出一个细节控制器，（有一个灵活的抽象脊柱）通过最优化PD控制器在满足约束的同时来模仿视频中的动作。从而模拟出大量的动作包括走，小跑，慢跑，飞奔。
Peng在控制四足动物模型时应用了强化学习以适应不同的2d地面状态【2015】。
在高维状态空间设计一个稳定的控制器是一个很难的问题，因此低维空间特征需要被手动调控来保持控制器的稳定。Peng通过对自动计算出的特征空间应用深度强化学习解决了这一问题。这个方法又被提升于3d环境和应用到二足动物角色上【2016】。

**4.3 补充**

与基于物理的动画相对的是数据驱动的动画技术，使用捕获的运动数据来进行交互式的角色控制。数据结构比如动作图(motion graphs)，被用来在无序的动作捕获数据中生成连续角色动作。动作图的连通性影响了被控制的角色的反应度，计算机游戏和其他交互应用经常使用更简单的结构，比如有限状态机(finite state machine)它的连通性更加明显，并且序列性动作可以预测。

### 5. 实现角色与环境动态交互的技术

**5.1 基于运动学的动作序列合成**

**基于模板的方法**将经过剪辑的运动片段插入场景中，并依据场景进行适应性调节。这些方法易于实现，但无法为大规模动画生成连续的运动数据。

1. Kang Hoon Lee等人于2006年提出了使用积木块作为“动作补丁”，构建大面积的虚拟环境，引导角色寻找达到目标位置的方法。积木块被赋予动作数据，包含了角色所被允许进行交互的动作集合。动作补丁模型实现了在较为复杂的环境中生成角色的动作。
2. Shailen Agrawal等人于2016年提出了一种目标驱动的运动模型，该法将人物角色的几种不同类型的脚步作为模板，通过调用和优化角色的脚步计划，并以此为基础生成人物角色全身的运动。该法实现了人物角色与环境之间更密切的交互。

**基于内核的方法**解决了基于模板的方法留下的问题。

1. Tomohiko Mukai等人为了解决地质统计学的相关问题，提出了将插入的动作片段视为参数空间内的数据预测的方法。
2. Jack M. Wang等人于2008年提出了高斯过程的动力学模型(GPDMs)，并将其应用于通过高维运动捕捉数据学习人类角色的姿势和动作。该类模型包含与动力学相关的低维隐式空间，以及从隐式空间到显式空间的映射。

**5.2 通过搜索的运动合成**

1. Min Gyu Choi等人于2003年提出了环境中的概率路线图，并且使用动捕数据和概率路线图生成二足动物角色的动作。这套方案以概率路径规划和多层次位移映射为基础，包括路线图构建、路线图搜索和运动合成三个部分。
2. Wan-Yen Lo和Matthias Zwicker 于2003年将强化学习应用于引导人物角色穿过门，于2012年实现引导人物角色绕开障碍物到达目的地。
3. Safonova和Hodgins于2007年引入A*搜索算法，用于混合动作插值的权重从而产生一个新的运动。此模型相当于一个参数化的图结构。Jianyuan Min等人使用最大后验估计(MAP)优化了动作权重的混合过程。

**5.3 通过局部优化或随机优化合成运动**。Igor Mordatch等人以一种局部优化策略——接触不变的优化方法为内核，提出了一种动作合成框架。Wenping Zhao等人使用一种随机优化策略——微粒群优化算法生成人物抓取的动作。

### 6. 应用于合成密切交互的深度学习技术

**6.1 深度监督学习技术**

1. Katerina Fragkiadaki等人在2015年将长短期记忆神经网络( Long Short Term Memory Networks, LSTMs)应用于角色移动的动画模拟；Zimo Li、Ruben Villegas等人的团队将该技术应用于更为复杂的人物动作的刻画。
2. Zimo Li等人于2017年通过自适应的递归神经网络(auto-conditioned Recurrent Neural Network, RNN)的Teacher Forcing训练机制，改善了LSTMs随着错误的积累模拟精确度逐渐下降的问题。
3. Shaojie Bai等人于2018年指出使用简单的卷积神经网络结构进行模拟的效果优于RNN和LSTMs。
4. Daniel Holden等人于2016年将时间卷积应用于这一领域的研究，并于2017年提出相位函数神经网络(PFNN)，用以进行人物角色运动的生成。
5. Robert A. Jacobs在1991年提出的多专家模型(the mixture of experts)是一种监督学习模型。以此为基础，He Zhang等人于2018年提出模式自适应神经网络(mode-adaptive neural network, MANN)，用以模拟四足动物的运动。该模型由预测神经网络和一个门网络组成。

**6.2 深度强化学习技术**

1. 深度强化学习被广泛应用于人物动作的模拟。Xue Bin Peng等人于2018年前后引入参考数据(reference data)和视频数据进行训练，弥补了由单一的数据反馈带来的不足。
2. Wenhao Yu等人于2018年提出了一种训练对称和节能的运动模式的方法。通过为损失函数引入不对称的惩罚项，以及使用额外的辅助学习(locomotion curriculum learning)，生成了更符合人类认知的行走动作。



## 4 应用和展望
## 5 参考文献
 
0.DANIEL HOLDEN, University of Edinburgh、TAKU KOMURA, University of Edinburgh、JUN SAITO, Method Studios，SIGGRAPH2017《Phase-Functioned Neural Networks for Character Control》
1. HE ZHANG † , University of Edinburgh、SEBASTIAN STARKE , University of Edinburgh
TAKU KOMURA, University of Edinburgh、JUN SAITO, Adobe Research, SIGGRAPH2018《Mode-Adaptive Neural Networks for Quadruped Motion Control》

2.SEBASTIAN STARKE† , University of Edinburgh, United Kingdom HE ZHANG† , University of Edinburgh TAKU KOMURA, University of Edinburgh JUN SAITO, Adobe Research, USA ，SIGGRAPH2019《Neural State Machine for Character-Scene Interactions》

3. Nicholas R Howe, Michael E Leventon, and William T Freeman. 1999. Bayesian Reconstruction of 3D Human Motion from Single-Camera Video.. In Proc. NIPS

4. Alla Safonova, Jessica K Hodgins, and Nancy S Pollard. 2004. ACM Trans on Graph 23, 3 (2004),《Synthesizing physically realistic human motion in low-dimensional, behavior-specific spaces. 》//PCA

5. Jinxiang Chai and Jessica K. Hodgins. 2005. Performance Animation from Lowdimensional Control Signals. ACM Trans on Graph 24, 3 (2005)

6. Jochen Tautges, Arno Zinke, Björn Krüger, Jan Baumann, Andreas Weber, Thomas Helten, Meinard Müller, Hans-Peter Seidel, and Bernd Eberhardt. 2011. Motion reconstruction using sparse accelerometer data. ACM Trans on Graph 30, 3 (2011)

7. Keith Grochow, Steven L Martin, Aaron Hertzmann, and Zoran Popović. 2004. Stylebased inverse kinematics. ACM Trans on Graph 23, 3 (2004), 522–531.

8. Sergey Levine, Jack M Wang, Alexis Haraux, Zoran Popović, and Vladlen Koltun. 2012. Continuous character control with low-dimensional embeddings. ACM Trans on Graph 31, 4 (2012), 28.

9. Jack M. Wang, David J. Fleet, and Aaron Hertzmann. 2008. Gaussian Process Dynamical Models for Human Motion. IEEE PAMI 30, 2 (2008), 283–298.

10. Carl Edward Rasmussen and Zoubin Ghahramani. 2002. Infinite mixtures of Gaussian process experts. In Proc. NIPS. 881–888.

11. Graham W Taylor and Geoffrey E Hinton. 2009. Factored conditional restricted Boltzmann machines for modeling motion style. In Proc. ICML. ACM, 1025–1032.

12. Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik. 2015. Recurrent network models for human dynamics. In Proc. ICCV. 4346–4354.

13. Daniel Holden, Jun Saito, and Taku Komura. 2016. A deep learning framework for character motion synthesis and editing. ACM Trans on Graph 35, 4 (2016)

14. Manfred Lau and James J Kuffner. 2005. Behavior planning for character animation. In
Proc. SCA.

15. Alla Safonova and Jessica K Hodgins. 2007. Construction and optimal search of interpolated motion graphs. ACM Trans on Graph 26, 3 (2007), 106

16. Stelian Coros, Philippe Beaudoin, Kang Kang Yin, and Michiel van de Pann. 2008. Synthesis of constrained walking skills. ACM Trans on Graph 27, 5 (2008), 113

17.. Libin Liu, KangKang Yin, Michiel van de Panne, Tianjia Shao, and Weiwei Xu. 2010. Sampling-based contact-rich motion control. ACM Trans on Graph 29, 4 (2010),

18. Jinxiang Chai and Jessica K. Hodgins. 2007. Constraint-based motion optimization using a statistical dynamic model. ACM Trans on Graph 26, 3 (2007).

19. Jianyuan Min and Jinxiang Chai. 2012. Motion graphs++: a compact generative model for semantic motion analysis and synthesis. ACM Trans on Graph 31, 6 (2012),

20. Jehee Lee and Kang Hoon Lee. 2004. Precomputing avatar behavior from human motion data. Proc. SCA (2004), 79–87

21. Yongjoon Lee, Kevin Wampler, Gilbert Bernstein, Jovan Popović, and Zoran Popović. 2010. Motion fields for interactive character locomotion. ACM Trans on Graph 29, 6 (2010), 
138.

22. Sergey Levine, Jack M Wang, Alexis Haraux, Zoran Popović, and Vladlen Koltun. 2012. Continuous character control with low-dimensional embeddings. ACM Trans on Graph 31, 4 
(2012), 28.

23. Wan-Yen Lo and Matthias Zwicker. 2008. Real-time planning for parameterized human motion. In Proc. I3D. 29–38.

24. Xue Bin Peng, Glen Berseth, and Michiel van de Panne. 2016. Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning. ACM Trans on Graph 35, 4 (2016).

25. Simon Clavet. 2016. Motion Matching and The Road to Next-Gen Animation. In Proc. of GDC 2016.

26. Kang Hoon Lee, Myung Geol Choi, and Jehee Lee. 2006. Motion patches: building blocks for virtual environments annotated with motion data. ACM Trans on Graph 25, 3 (2006), 
898–906.

27. Helmut Grabner, Juergen Gall, and Luc Van Gool. 2011. What makes a chair a chair?. In Proc. IEEE CVPR. 1529–1536.

28. Abhinav Gupta, Scott Satkin, Alexei A Efros, and Martial Hebert. 2011. From 3d scene geometry to human workspace. In Proc. IEEE CVPR. 1961–1968.

29. Vladimir G. Kim, Siddhartha Chaudhuri, Leonidas Guibas, and Thomas Funkhouser. 2014. Shape2pose: Human-centric shape analysis. ACM Trans on Graph 33, 4 (2014), 120

30. Changgu Kang and Sung-Hee Lee. 2014. Environment-Adaptive Contact Poses for Virtual Characters. In Computer Graphics Forum, Vol. 33. Wiley Online Library, 1–10.

31. Manolis Savva, Angel X. Chang, Pat Hanrahan, Matthew Fisher, and Matthias Nießner. 2016. PiGraphs: Learning Interaction Snapshots from Observations. ACM Trans on Graph 35, 4 
(2016)

32. Mubbasir Kapadia, Xu Xianghao, Maurizio Nitti, Marcelo Kallmann, Stelian Coros, Robert W Sumner, and Markus Gross. 2016. Precision: precomputing environment semantics for contact-rich character animation. In Proc. I3D. 29–37

33. Tejas D Kulkarni, William F Whitney, Pushmeet Kohli, and Josh Tenenbaum. 2015. Deep convolutional inverse graphics network. In Proc. NIPS. 2539–2547.

34. Roland Memisevic. 2013. Learning to relate images. IEEE PAMI 35, 8 (2013),

35.  In Proceedings of SIGGRAPH’85, 1985, 255-262. Michael Girard, Anthony A Maciejewski. Computational Modeling for the Computer Animation of Legged Figures.

36. Blumberg, B., and Galyean, T. 1995. Multi-level direction of autonomous creatures for real-time virtual environments. In Proc. ACM SIGGRAPH, ACM, 47--54.

37. Hecker, C., Raabe, B., Enslow, R. W., DeWeese, J., Maynard, J., and van Prooijen, K. 2008. Real-time motion retargeting to highly varied user-created morphologies. ACM 
Trans. on Graphics (Proc. SIGGRAPH) 27, 3.

38. Torkos, N., and van de Panne, M. 1998. Footprint-based quadruped motion synthesis. In Proc. Graphics Interface, 151--160

39. Kevin Wampler and Zoran Popović. 2009. Optimal gait and form for animal locomotion. In ACM Transactions on Graphics (TOG), Vol. 28. ACM, 60. 40.

41. Kevin Wampler, Zoran Popović, and Jovan Popović. 2014. Generalizing locomotion style to new animals with inverse optimal regression. ACM Transactions on Graphics (TOG) 33, 4 
(2014), 49.

42. Sergey Levine and Jovan Popović. 2012. Physically Plausible Simulation for Character Animation. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA ’12). Eurographics Association, Goslar Germany, Germany, 221–230.

43. Marc H. Raibert and Jessica K. Hodgins. 1991. Animation of dynamic legged locomotion. In Proceedings of the 18th Annual Conference on Computer Graphics and Interactive 
Techniques, SIGGRAPH 1991, Providence, RI, USA, April 27-30, 1991. 349–358.

44. Michiel van de Panne. 1996. Parameterized gait synthesis. IEEE Computer Graphics and Applications 16, 2 (1996), 40–4

45. Stelian Coros, Andrej Karpathy, Ben Jones, Lionel Reveret, and Michiel Van De Panne. 2011. Locomotion skills for simulated quadrupeds. In ACM Transactions on Graphics (TOG), 
Vol. 30. ACM, 59.

46. Xue Bin Peng, Glen Berseth, and Michiel van de Panne. 2015. Dynamic Terrain Traversal Skills Using Reinforcement Learning. ACM Trans. Graph. 34, 4, Article 80 (July 2015), 11 pages.

47. Xue Bin Peng, Glen Berseth, and Michiel van de Panne. 2016. Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning. ACM Transactions on Graphics (Proc. 
SIGGRAPH 2016) 35, 4 (2016).

48. Kang Hoon Lee, Myung Geol Choi, and Jehee Lee. 2006. Motion patches: building blocks for virtual environments annotated with motion data. ACM Trans on Graph 25, 3 (2006).

49. Shailen Agrawal and Michiel van de Panne. 2016. Task-based Locomotion. ACM Trans on Graph 35, 4 (2016).

50. Tomohiko Mukai and Shigeru Kuriyama. 2005. Geostatistical motion interpolation. ACM Trans on Graph 24, 3 (2005).

51. J.M. Wang, D.J. Fleet, and A. Hertzmann. 2008. Gaussian Process Dynamical Models for Human Motion. IEEE PAMI 30, 2 (Feb 2008

52. Min Gyu Choi, Jehee Lee, and Sung Yong Shin. 2003. Planning biped locomotion using motion capture data and probabilistic roadmaps. ACM Trans on Graph 22, 2 (2003)

53. Wan-Yen Lo and Matthias Zwicker. 2008. Real-time planning for parameterized human motion. In Proc. SCA. 29–38.

54. Wan-Yen Lo, Claude Knaus, and Matthias Zwicker. 2012. Learning motion controllers with adaptive depth perception. In Proc. SCA. 145–154.

55. Alla Safonova and Jessica K Hodgins. 2007. Construction and optimal search of interpolated motion graphs. ACM Trans on Graph 26, 3 (2007).

56. Jianyuan Min and Jinxiang Chai. 2012. Motion Graphs++: A Compact Generative Model for Semantic Motion Analysis and Synthesis. ACM Trans on Graph 31, 6 (2012), 153:1–153:12

57. Igor Mordatch, Emanuel Todorov, and Zoran Popović. 2012. Discovery of complex behaviors through contact-invariant optimization. ACM Trans on Graph 31, 4 (2012).

58. Wenping Zhao, Jianjie Zhang, Jianyuan Min, and Jinxiang Chai. 2013. Robust realtime physics-based motion control for human grasping. ACM Trans on Graph 32, 6 (2013)

59. Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik. 2015. Recurrent network models for human dynamics. In Proc. ICCV. 4346–4354.

60. Zimo Li, Yi Zhou, Shuangjiu Xiao, Chong He, Zeng Huang, and Hao Li. 2017. Autoconditioned recurrent networks for extended complex human motion synthesis. arXiv preprint 
arXiv:1707.05363 (2017

61. Ruben Villegas, Jimei Yang, Duygu Ceylan, and Honglak Lee. 2018. Neural Kinematic Networks for Unsupervised Motion Retargetting. Proceedings of CVPR 2018.

62. Shaojie Bai, J Zico Kolter, and Vladlen Koltun. 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint 
arXiv:1803.01271 (2018).

63. Daniel Holden, Jun Saito, and Taku Komura. 2016. A deep learning framework for character motion synthesis and editing. ACM Trans on Graph 35, 4 (2016).

64. Daniel Holden, Taku Komura, and Jun Saito. 2017. Phase-functioned neural networks for character control. ACM Trans on Graph 36, 4 (2017), 42.

65. Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. Neural Computation 3, 1 (1991), 79–87

66. Xue Bin Peng, Angjoo Kanazawa, Jitendra Malik, Pieter Abbeel, and Sergey Levine. 2018b. SFV: reinforcement learning of physical skills from videos. ACM Trans on Graph 37, 6 
(2018).

67. Wenhao Yu, Greg Turk, and C Karen Liu. 2018. Learning symmetric and low-energy locomotion. ACM Trans on Graph 37, 4 (2018).

