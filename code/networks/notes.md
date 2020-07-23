# Notes

# vae testing
- 20 epochs seemed to be a sweet spot
- The VAE achieved lower losses with msle
- VD appeared to decrease loss in vae, but results were skewed.  VD appearded to reward circles and perpendicular lines.
  - 0, 9 and 6 were distinuishable
  - 1 was opposite 0, and 7 was rotated from both, but elongated
  - By contrast regular Sampling Layer has troubles discerning loops
  - msle improved loss and prod
  - Loss
    - loss greater with msle
    - loss greater with vd, but reconstruction is garbled
  - A trained vae improved the stack model in just a single episode of training. 


## Stack 