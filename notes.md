**why we are using SAR log scale**

Radar comes out of the file as big numbers. Optical comes out small.

```
S1_VV     203
NDVI        0.55
```

The network has no idea one is radar and one is reflectance. It just sees numbers,
and a channel 400x bigger drowns out the rest.

**Step 1: log10.** This squashes big numbers much harder than small ones:

```
log10(30)  = 1.48
log10(200) = 2.30
log10(900) = 2.95
```

A 30x spread becomes a 2x spread.It means the gap between your smallest and biggest radar value shrinks.

Before
Your radar values run from about 30 to about 900.

```
smallest    30
biggest    900
```

900 ÷ 30 = 30      the biggest is 30 times the smallest

After log10
```
log10(30)  = 1.48
log10(900) = 2.95

2.95 ÷ 1.48 = 2.0   the biggest is only 2 times the smallest
```

**Step 2: divide by 3.**

```
2.30 / 3 = 0.77
```

Now radar sits at 0.77, right next to NDVI at 0.55. Measured across 200 real patches:

```
NDVI   0.5454
S1_VV  0.7588
S1_VH  0.6608
```

**Step 3: keep the gaps as gaps.** Some pixels are 0 because the radar swath did not
reach them. `log10(0)` is minus infinity, so those are forced back to 0:

```python
np.where(v > 0, np.log10(np.maximum(v, 1e-6)) / 3.0, 0.0)
```

Otherwise "no data here" would become a large negative number the network reads as a
real, very low radar return.
<!-- its not true 
## Why log10 and not just dividing by 300

Dividing changes the size of the numbers but not their relationship to anything else.
Checked on 3,951 real shots:

```
raw VH      corr with biomass  0.1834
VH / 300    corr with biomass  0.1834   <- identical, dividing changes nothing
log10 VH    corr with biomass  0.2100   <- genuinely better
```

Shrinking every value by the same factor does not change which pixels are bright
relative to each other, so the correlation cannot move. A log changes their relative
spacing, so it can. -->

### What the log actually does

Nothing gets brighter. The top gets squashed harder than the bottom.

```
raw      after log10
 30         1.48
100         2.00
900         2.95

gaps:   30 -> 100 : raw +70    log +0.52
       100 -> 900 : raw +800   log +0.95
```

In raw numbers the second gap is 11x the first. After the log it is only 1.8x. The low
values did not move up, the high values were pulled down toward them.

### Why that helps, reason 1: speckle

Radar sends a wave and listens for the echo. Echoes bouncing off different leaves within
one pixel arrive slightly out of step and interfere, so a pixel can come back very bright
or very dark purely by chance, with no difference in the vegetation. That random grain is
called speckle and every radar image has it.

Five neighbouring pixels of the same forest, one of them speckle:

```
raw    100  100  900  100  100      <- the 900 is noise
log   2.00 2.00 2.95 2.00 2.00
```

Raw, the fake pixel is 9x its neighbours and the network reacts to it as an enormous
signal. After the log it is 1.5x: still the brightest, no longer dominating.

### Why that helps, reason 2: C-band saturates

Mean VH per biomass band, 3,951 shots:

```
   biomass      n    mean raw VH
     0-15     925        82.5
    15-30     530       104.5     <- +22, nearly all the change is here
    30-60     754       108.6     <- +4
   60-120     924       108.9     <- +0.3
  120-240     607       112.4     <- +3.5
 240-1100     211       117.0     <- +4.6
```

**Saturation means the radar stops responding once there is enough vegetation. More
biomass, same number back.**

Read the right column above. Between 0 and 30 Mg/ha it jumps by 22. After that it
crawls. Concretely: a shot at 30 Mg/ha reads about 108.6, a shot at 500 Mg/ha reads
about 117. Seventeen times the biomass, eight percent more radar.

Like a bathroom scale that only goes to 100 kg. Someone who weighs 90 reads 90. Someone
who weighs 140 reads 100, and so does someone who weighs 200. The scale is not broken,
it is maxed out, and above its limit everyone looks identical.

**Why radar does this.** The wave has to reach the biomass to detect it. C-band is short,
about 5.6 cm. It bounces off leaves and small twigs in the TOP of the canopy and never
gets down to the trunks and big branches, which is where most of the mass actually is.

```
     /\/\/\    <- C-band bounces off here and comes back
    /  \  /\
   |  |  |  |    <- most of the biomass is down here, unseen
  ============
```

Once the canopy is dense enough to intercept everything, adding more forest underneath
changes nothing that comes back.

**Why this matters for us.** Published C-band saturation is 61 to 74 Mg/ha. Our data
shows it flattening even earlier, around 30. Our dataset mean is 77 Mg/ha, so radar is
saturated across most of our range: it can separate bare ground from shrub, but not
100 Mg/ha forest from 400 Mg/ha forest. That is a measured explanation for why
Sentinel-1 has contributed so little here, and it is stronger than simply reporting
that it did not help.

It is also the answer to the PALSAR question. L-band is a ~24 cm wave, long enough to
penetrate the canopy and reach trunks, so it saturates around 100-170 Mg/ha instead.
That is why L-band is preferred in forested areas, and why the coarse-resolution
objection to PALSAR is a tradeoff rather than a simple no.
    
---

# Winsorization

**Pick a maximum. Anything above it gets changed to that maximum. Nothing is deleted.**

```
before:   10   20   30   40   1000
after:    10   20   30   40     50     <- cutoff was 50
```

The big one is still the biggest, it just cannot be extreme any more.

## How the cutoff is chosen

Line up all 153,715 biomass values smallest to largest and walk 99.5% of the way along.
The value there is **429.4 Mg/ha**. Everything above it, the top 0.5%, gets set to 429.4.

```
 908.5  ->  429.4
1024.3  ->  429.4
  52.0  ->   52.0     unchanged, already below the cutoff
```

769 shots affected out of 153,715.

## What it does to the labels

```
              before     after
mean           76.97     76.51
std            83.07     80.52
max          1024.32    429.43
skew            2.108     1.763
```

## Why anyone does it

The model is punished by the SQUARE of its mistakes.

```
off by 50   ->  penalty   2,500
off by 900  ->  penalty 810,000      324x bigger
```

So one redwood stand the model gets wrong counts for more than three hundred ordinary
shots it gets right. The model starts bending itself to fit a few giants instead of the
153,000 normal shots. Capping them stops that.

## What it costs

The model can then never predict above 429. Our ROI genuinely contains forest up to
1,024 Mg/ha, so we would be guaranteeing wrong answers on the biggest trees.

## Why it is probably not needed here

Two things already handle this:

* Training on `agbd_sqrt` cut the skew from 2.108 to 0.583. Winsorizing only reaches
  1.763, which is WORSE than what we already have.
* `HuberLoss(delta=2.5)` already caps how much any single shot can contribute.

So it would be a third fix for a problem already solved twice. Still worth one run to
confirm, and a null result is a legitimate methods-section sentence.
