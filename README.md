# Epistemic Advantage on the Margin: A Network Standpoint Epistemology
A bandit model by Professor [Jingyi Wu](https://www.jingyiwu.org/) that depicts the emergence of several epistemic advantages for marginalized groups due to testimonial ignoration and devaluation. View the full paper [here](http://doi.org/10.1111/phpr.12895).

## Abstract
> I use network models to simulate social learning situations in which the dominant group ignores or devalues testimony from the marginalized group. I find that the marginalized group ends up with several epistemic advantages due to testimonial ignoration and devaluation. The results provide one possible explanation for a key claim of standpoint epistemology, the inversion thesis, by casting it as a consequence of another key claim of the theory, the unidirectional failure of testimonial reciprocity. Moreover, the results complicate the understanding and application of previously discovered network epistemology effects, notably the Zollman effect.

## Model Details
Implemented here is Wu's second variation with one-sided testimonial devaluation.

### Variation 2: One-sided testimonial devaluation
Simulates testimonial devaluation using Jeffrey conditionalization. Differs from the base model in the way that agents beliefs are updated.
* Size of network (3, 6, 12, 18)
* Number of pulls per round (1, 5, 10, 20)
* Probability of B (0.51, 0.55, 0.6, 0.7, 0.8)
* Proportion of marginalized group in population (1/6, 1/3, 1/2, 2/3)
* Degree of devaluation(0.2, 0.5, 0.8)
* Network structure: complete

## About the author
Jingyi Wu is an Assistant Professor in the Department of Philosophy, Logic and Scientific Method at the [London School of Economics](https://www.lse.ac.uk/). She primarily works on social epistemology and philosophy of physics. She also has interests in general philosophy of science, feminist philosophy, philosophy of race, Asian/American philosophy, and mathematical physics.