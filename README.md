# e-safe

An online learning algorithm which chooses the safest update with probability (1-epsilon), otherwise chooses from the other updates uniformly at random.

## Application

The goal is to predict the distribution of next requests given the current request sequence.
The dataset is from the 1998 World Cup Website and contains ~1 billion requests, it can be found here:
http://ita.ee.lbl.gov/html/contrib/WorldCup.html

For comparison the algorithm will be evaluated against an HMM trained offline over the same evaluation set.