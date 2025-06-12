---
layout: default
title: Nerf Weight Diffusion
date: 2024-03-21
---
<div class="project-page">
  <div class="project-content">
    <h1>Robot Navigation With Nerf</h1>
    
    <h2>Overview</h2>
    <p>This project assumes static 3D scene and consistency of robot's hardware configuration. Assuming that the scene is static and the hardware configuration does not change over time, also assume that there's a pre-trained Nerf model for the corresponding scene obtained from the robot's hardware. Once the robot is deployed, the initial pose with respect to the Nerf model can be obtained using methods like iNerf (<a href="https://arxiv.org/pdf/2012.05877">link</a>). Then I want to develop a navigation model for the robot that is coupled with the Nerf model. Since the weights of the Nerf model hold 3D information of the scene in question, my hope is that we can use it for conditioning. For example, we can condition a diffusion model with the linearized weights of the Nerf model along with additional information like the current state, currently viewing image, and some form of information that indicates a goal of the navigation. Then the output of the diffusion model would be the next relative movement from the current state. Afterwards, such output can be "added" to the current state and fed into the Nerf model, which results in a rendered image of what the robot system should be viewing once the system has moved. Then the rendered image could be compared against what the robotic system actually sees in order to assist in the learning of the navigation model. Of course, this should be coupled with other conventional frameworks like Reinforcement Learning. This fully utilizes the 3D information of the scene in which the robot is navigating through via the weights of the Nerf model. As a side note for myself, Nerf can be used with multi-robot perception system as shown in Distributed Nerf Learning for Collaborative Multi-Robot Perception (<a href="https://arxiv.org/pdf/2409.20289">link</a>).</p>

    <br>
    <br>

    <h2>Considerations</h2>
    <p>From Section 4 of Learning A Diffusion Prior For Nerfs (<a href="https://arxiv.org/pdf/2304.14473">link</a>), the obfuscation of 3D information when using Nerf representation is evident since there can be multiple different Nerfs that represent the same scene. The paper uses regularization in order to limit the region of representation so as to limit the one-to-many relationship between the scene and the representation, thereby mitigating the confusion for the diffusion model.</p>
  </div>
</div> 