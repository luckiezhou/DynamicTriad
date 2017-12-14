## Dynamic Network Embedding

The goal of so-called "network embedding" is to project each vertex in a graph to a vector in a low-dimensional space. The task attracts considerable research efforts recently, and often serves as a basic feature extraction method for more complex problems. Existing work on network embedding focus mainly on static networks, however, most real-world networks regards evolution as their natural characteristics, that is, real-world networks are usually dynamic. As far as we know, there is no widely-accepted dynamic network embedding algorithm that is proved to work well in practice.

Most existing dynamic network embedding algorithms hold two common assumptions, namely social homophily and temporal smoothness. The social homophily assumption depicts all kinds of structural proximities that is well studied in the static networks, and temporal smoothness assumes the projection of vertices change smoothly over time. However, these assumptions considers spatial and temporal relations separately, and can hardly capture complex network evolution patterns (i.e. the patterns of structural change).

## Triadic Closure Process and Social Strategy

As an effort to take evolution patterns directly into account, we try to model some basic patterns in our dynamic network embedding algorithm. Triangles are known to be a common component of a social network, and the closing of a triangles is considered one of the most important factors for a new edge to emerge. As a result, *the Triadic Closure Process* is directly modeled in the dynamic network embedding algorithm proposed in our new AAAI 18 paper [1].

In a social network, *the Triadic Closure Process* describes the scenario when users are introduced to each other by their common friend. Obviously, the probability of two users to be acquainted with each other depends on the eagerness of their common friends to introduce them to each other, and we call such eagerness the *social strategy* of the user (indeed, the name is not precise as what we discuss here reflects only a part of the literal meaning of social strategy). It is natural to assume that the *social strategy* varies for each user (vertex) in the network. As shown in the figure below, user A tends to introduce new links between his/her friends while user B tends to keep the relations unchanged. 

<div align="center">
    <img src="https://raw.githubusercontent.com/luckiezhou/DynamicTriad/master/docs/image.png"><br><br>
</div>

## Dynamic Network Embedding by Modeling Triadic Closure Process

The core idea of paper [1] is to model the willingness of a user to introduce his/her friends to each other, which we call *social strategy*, according to their relation strengths, and integrate this information into the embedding vector of this user. 

The proposed algorithm defines a triadic loss for each open triangle (two edges among three vertices),  computed according to the relative positions of the three vertices in the latent space, the weight of edges between them and whether the open triangle closes in the next time step.

Together with the basic assumptions of social homophily and temporal smoothness, we define a loss function for each assumption and convert the embedding task into an optimization problem. The optimization problem can be effectively solved with EM algorithm.

## Reference

[1] Zhou, L; Yang, Y; Ren, X; Wu, F and Zhuang, Y, 2018, Dynamic Network Embedding by Modelling Triadic Closure Process, In AAAI, 2018 
