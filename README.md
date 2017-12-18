# AdaLearn
Setup Adabot from [Adabot](https://github.com/anthonyjclark/adabot)

To train Adabot agent:
1. Run Simulation by  **roslaunch adabot_gazebo adabot.world.launch more details** [AdabotGazebo](https://github.com/anthonyjclark/adabot/wiki/adabot-gazebo-Package)
2. Run one of the following to train agent:
- DumbRos.py : Handcrafted controller
- DeepDeterministic.py: Continuous Deep Deterministic Control controller. 
- SimplyPPO.py: Continuous Deep Proximial PolicyControl controller. 
- MLPActorCritic.py: Discrete State MLP controller.
