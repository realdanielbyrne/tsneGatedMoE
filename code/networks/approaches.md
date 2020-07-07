# Sampling prior models to produce posterier.  Probabilistic approach to Transfer Learning.

1. Slap pre-trained Encoder onto top of any other network
2. Sample from prior for discrete classes and average to derive expected means and variances.
   Use those parameters to initialize layers in another model which act as gating networks in a mixture of experts model.
  1. Optionally turn off or turn down the learning rate on this layer to keep the initialization values longer into the training cycle.
  2. The sampling prior model opens up the possibility of using this type of transfer learning across many differnt types of priors.  For instance, what if we could erive a set of 5 parameters that describes a trained VGG16, or a Resnet30?  We could then use these parameters as priors in more advanced models.

