# Neighborhood thinning as a cosmological mechanism

## Status

**Restart checkpoint: 2026-08-21.** This is a falsifiable toy model and
numerical calibration, not a derivation of dark energy. All calculations
completed so far are recorded below and implemented in the accompanying
scripts.

The present verdict is:

1. The paper's combinatorial result remains exact: adding independent,
   nonnegative distinguishing coordinates exponentially thins an exact-match
   neighborhood.
2. Mapping that thinning directly to three-dimensional physical volume gives
   the exact dictionary $H=(\ln2/3)\dot N$, but the dictionary does not supply
   dynamics or establish causation.
3. The simple age law $N=3\log_2(t/t_*)$ predicts the correct few-percent scale
   of $H_0$, but gives a coasting universe and is strongly rejected by the DESI
   distance--redshift shape.
4. Treating accumulated history literally as another positive dimension does
   not repair the model. A relativistic causal-past formulation also fails with
   a constant coupling for both raw and logarithmic causal four-volume.
5. A two-channel rate fits DESI as well as flat $\Lambda$CDM, but is
   algebraically the flat-$\Lambda$CDM background equation and is not yet an
   independent explanation.
6. CAMELS simulations show the opposite of the simplest local causal claim:
   neighborhood change is greatest in dense, gravitationally bound regions
   where expansion is suppressed. Most of that change is predicted by ordinary
   peculiar velocity, and the pattern persists in gravity-only simulations.
7. The surviving possibility is therefore narrower: relational complexity may
   describe gravitational dependency or binding, but the tested neighborhood
   measures do not generate or explain dark-energy expansion.

This is enough for a short exploratory or no-go paper. It is not enough for a
paper claiming a new theory of dark energy. The defensible contribution is the
sequence of explicit bridges, cosmological shape tests, relativistic checks,
and simulation controls that sharply delimit what a neighborhood-complexity
interpretation can mean. A five-page, two-column manuscript implementing this
framing was completed as `neighborhood_thinning_cosmology.tex` and compiled to
`neighborhood_thinning_cosmology.pdf` on 2026-08-21.

A second, deliberately narrower note was completed on the same date in
`../cosmology_time_note/`. It isolates the accumulated-time construction, shows
why $N=3\log_2(t/t_*)$ gives the attractive $70.85$ Hubble-scale estimate,
distinguishes it from literal $3+1$ counting, and gives the explicit power-law
and DESI reasons both constructions fail as expansion histories.

## 1. The exact bridge

For independent binary distinguishing coordinates and an exact-match
neighborhood ($r=0$), the neighborhood density is

$$
\rho_N=2^{-N}.
$$

Define effective relational volume and a three-dimensional scale factor by

$$
V_{\rm rel}=\rho_N^{-1},
\qquad
a_{\rm rel}=V_{\rm rel}^{1/3}=2^{N/3}.
$$

Then the relational Hubble rate is exactly

$$
H_{\rm rel}=\frac{\dot a_{\rm rel}}{a_{\rm rel}}
=\frac{\ln 2}{3}\dot N.
\tag{1}
$$

For larger fixed radii the paper's polynomial correction appears, but the
leading exponential is unchanged.  Equation (1) is a dictionary: it becomes a
physical claim only after $N$ is independently defined and its dynamics are
specified.

## 2. A minimal history-complexity hypothesis

Suppose a history needs one new independent binary distinction in each spatial
dimension whenever the available temporal baseline doubles.  Then

$$
N(t)=3\log_2(t/t_*).
\tag{2}
$$

Equations (1) and (2) give

$$
a(t)=t/t_*,\qquad H(t)=1/t.
\tag{3}
$$

Using the measured cosmic age $t_0=13.80\,\mathrm{Gyr}$, equation (3) predicts

$$
H_0^{\rm age}=70.85\ {\rm km\,s^{-1}\,Mpc^{-1}}.
$$

That is 5.1% above Planck's $67.4$, 3.9% above the DESI DR2+CMB value
$68.17$, and 3.0% below the 2022 SH0ES value $73.04$.  Equivalently, the
observed dimensionless products $H_0t_0$ are approximately 0.95, 0.96, and
1.03.  Thus the ansatz gets the correct order of magnitude—and more than that,
the correct few-percent scale—without putting $H_0$ into equation (2).

There is an important qualification: the CMB-derived age and CMB-derived Hubble
constant are correlated and model-dependent, so the Planck comparison is not a
fully independent prediction.

## 3. Why this is not yet dark energy

The scale factor in (3) is a coasting cosmology:

$$
\ddot a=0,\qquad q=-\frac{a\ddot a}{\dot a^2}=0.
$$

Using the DESI DR2+CMB flat-$\Lambda$CDM matter fraction
$\Omega_m=0.3027$ gives

$$
q_0=\tfrac12\Omega_m-\Omega_\Lambda\simeq-0.546.
$$

So the simplest history-bit law correctly estimates today's Hubble scale but
fails the acceleration test.  Supernova and BAO expansion histories can
distinguish these models; agreement at one time is insufficient.

A constant late-time production rate gives instead

$$
N(t)=N_*+\dot N_\Lambda t,
\qquad
a(t)\propto e^{H_\Lambda t},
\qquad
H_\Lambda=\frac{\ln2}{3}\dot N_\Lambda,
$$

which is de Sitter-like.  With DESI DR2+CMB parameters, the required rate is
about $0.250$ effective bits/Gyr, or one independent distinguishing bit every
four billion years.  This reconstructs the measured dark-energy scale but does
not predict it unless that bit rate follows from an independent microscopic
law.

The corresponding effective cosmological constant is

$$
\Lambda_{\rm eff}
=\frac{3H_\Lambda^2}{c^2}
=\frac{(\ln2)^2}{3c^2}\dot N_\Lambda^2.
\tag{4}
$$

Equation (4) is the concrete target: derive $\dot N_\Lambda$ without using
cosmological expansion data.

## 4. Bound neighborhoods

The intuition that internally dependent neighborhoods resist relational
thinning has a clean rate formulation.  For a spherical region of mass $M$
and radius $R$, define

$$
\Gamma_g^2=\frac{GM}{R^3}.
$$

The late-time expansion rate is

$$
\Gamma_\Lambda^2=H_0^2\Omega_\Lambda.
$$

The balance condition $\Gamma_g=\Gamma_\Lambda$ gives

$$
R_{\rm TA,max}
=\left(\frac{GM}{H_0^2\Omega_\Lambda}\right)^{1/3}
=\left(\frac{3GM}{\Lambda c^2}\right)^{1/3},
\tag{5}
$$

the standard maximum turnaround radius.  In density language it is

$$
\bar\rho>2\rho_\Lambda.
$$

This supports the qualitative picture: below the turnaround scale, relational
constraints can remain locked and physical separations need not follow the
global scale factor.  But equation (5) currently imports ordinary gravity; it
does not derive gravity from information consistency.  A successful theory
would have to derive the dependency rate $\Gamma_g$, including its dependence on
mass and radius, rather than rename it.

## 5. The non-circular empirical program

The cosmological data themselves reconstruct

$$
N_{\rm eff}(z)=3\log_2 a(z)=-3\log_2(1+z),
\qquad
\dot N_{\rm eff}(z)=\frac{3H(z)}{\ln2}.
$$

That reconstruction is useful but tautological.  A real test requires a second,
independently measured complexity proxy $C_{\rm obs}(z)$—for example a carefully
defined gravitational-structure entropy, causal-network complexity, or
dependency measure—and a single redshift-independent conversion between it and
$N_{\rm eff}$.  The model should then be fitted on part of the redshift range and
predict the held-out BAO/SN expansion history.

The decisive tests are:

1. **Magnitude:** derive the approximately $0.25\,\mathrm{bits/Gyr}$ late-time rate
   without using $H_0$ or dark-energy density.
2. **History:** predict $H(z)$, not only $H_0$; the logarithmic law (2) is already
   known to be incomplete because it gives $q=0$.
3. **Transition:** predict when $\ddot a$ changes sign.
4. **Local/global split:** recover the turnaround scaling $R\propto
   M^{1/3}$ and its normalization.
5. **Geometry:** explain why the volume exponent is three and preserve observed
   homogeneity, isotropy, lensing, and relativistic causal structure.

## 6. First DESI DR2 shape test

The accompanying test uses the published 13-element DESI DR2 BAO data vector
and its covariance.  For every model the nuisance scale

$$
X=\frac{c}{H_0r_d}
$$

is fitted analytically.  The fixed-law comparisons therefore test the shape of
the distance-redshift relation without using $H_0$, the quantity the hypothesis
was originally meant to explain.

The power-law family follows from

$$
N(t)=d_{\rm eff}\log_2(t/t_*),
\qquad
a(t)\propto t^p,
\qquad
p=d_{\rm eff}/3.
$$

The results are:

| model | chi-square | degrees of freedom |
|---|---:|---:|
| matter-like history, $p=2/3$ | 1420.18 | 12 |
| spatial history, $p=1$ | 187.09 | 12 |
| literal 3D+time history, $p=4/3$ | 1364.93 | 12 |
| best constant $p=0.9125$ | 96.97 | 11 |
| two-channel model | 10.27 | 11 |
| flat $\Lambda$CDM benchmark, $\Omega_m=0.3027$ | 10.63 | 12 |

Thus no constant-dimensional history law describes the DESI expansion shape.
In a predeclared split, fitting the free power law below $z=1$ gives
$p=1.1185$, but its prediction for the six measurements at $z\geq1$ has
$\chi^2=670.72$.  The simple 3D+time proposal is therefore ruled out in
this form, despite producing the right rough Hubble scale.

### Independent-channel variant

There is a more successful extension that remains close to the neighborhoods
idea.  Let statistically independent distinguishing channels add in
quadrature, as independent variances do:

$$
\dot N_{\rm eff}^{2}
=\dot N_{m,0}^{2}a^{-3}+\dot N_h^{2}.
\tag{6}
$$

The first channel is proportional to the density of local material
interactions; the second is a constant history-consistency production rate.
Using equation (1),

$$
H^2(a)=\left(\frac{\ln2}{3}\right)^2
\left(\dot N_{m,0}^{2}a^{-3}+\dot N_h^{2}\right).
\tag{7}
$$

Fitting only the channel ratio and BAO scale gives

$$
\Omega_{m,\rm eff}=0.2975,
\qquad
\chi^2=10.27\quad(11\ {\rm dof}).
$$

The low-redshift fit gives $\Omega_{m,\rm eff}=0.3095$; without refitting, its six
high-redshift predictions have $\chi^2=4.74$.  With a reference BAO ruler
$r_d=147.09\,\mathrm{Mpc}$, the full fit corresponds to
$H_0=69.03\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$, a present matter-channel rate of
$0.1667\,\mathrm{bits/Gyr}$, and a history-channel rate of
$0.2561\,\mathrm{bits/Gyr}$.

This is a genuine empirical success for the *form* of the two-channel model,
but equation (7) is mathematically the flat-LambdaCDM background equation in a
new vocabulary.  It becomes explanatory only if the following can be derived
independently:

- why independent relational production rates combine quadratically;
- why the local channel's squared rate dilutes exactly as $a^{-3}$;
- why the history channel has a constant rate near $0.256\,\mathrm{bits/Gyr}$; and
- why the fitted channel ratio is about $0.30:0.70$.

The model predicts the acceleration transition when

$$
2\dot N_h^{2}=\dot N_{m,0}^{2}a^{-3},
$$

which for the fitted ratio is $z_{\rm acc}\simeq0.68$.  This is now a
precise target for an information-theoretic derivation rather than an analogy.

## 7. Relativistic causal-history test

Treating time as a fourth positive metadata coordinate is not covariant.  A
relativistic replacement uses an observer's causal past in flat FLRW spacetime.
For an observation event at proper time $\tau_o$, its causal-past four-volume is

$$
V_4(\tau_o)=\frac{4\pi}{3}\int_0^{\tau_o}d\tau'
\,a^3(\tau')
\left[\int_{\tau'}^{\tau_o}\frac{c\,d\tau''}{a(\tau'')}\right]^3.
\tag{8}
$$

The covariant expansion scalar of the cosmological congruence is

$$
\theta=\nabla_\mu u^\mu=3H.
$$

The test asks whether a redshift-independent constitutive coupling can satisfy

$$
3H=\kappa\dot C
\tag{9}
$$

when causal-history complexity is either $C=\log_2V_4$ or $C=V_4$ in fixed
units.  Equation (8) was evaluated along the DESI-calibrated two-channel
background from $z=2.33$ to today.

The required coupling for $C=V_4$ changes by a factor of **321.8**, decisively
excluding a constant raw-event-volume coupling.  For $C=\log_2V_4$ it changes by
a smaller but still material factor of **1.514**.  The present required value is
$\kappa=0.5373$, rather than $\ln2=0.6931$ supplied by the original direct
neighborhood-volume dictionary.

This corrects the interpretation of the power-law test.  The data exclude the
simple constant-coupling causal-volume model; they do not exclude every form of
causal-history complexity.  Total causal volume counts redundant history over
and over.  A more plausible next object is the *conditional new information*
added between neighboring causal slices, after conditioning on their shared
past.  Unlike total four-volume, that quantity could approach a constant rate
in a late de Sitter-like regime.

The causal-volume calculation is an internal consistency diagnostic because
the FLRW background used in (8) is inferred from cosmological data.  A future
independent test requires a microscopic or simulation-derived conditional
information measure, followed by a prediction of $H(z)$ through (9).

## 8. Conditional-neighborhood pilot in CAMELS

The next pilot uses three public late-time halo catalogs from the CAMELS
IllustrisTNG `CV_0` simulation ($z=0.1001$, $0.0485$, and $0$).  Resolved
subhalos are matched between adjacent snapshots using mutual-nearest positions
and a mass-consistency cut.  Among subhalos whose most-bound particle ID remains
available at the next snapshot, this geometric matcher recovers the exact
identity **96--97%** of the time.

For each matched subhalo, the analysis constructs a $k=16$ neighborhood.  The
conditional novelty is the surprisal of neighbor relations that appear at the
next snapshot given the previous neighbor set.  Local physical expansion is
measured from the volume change of the same Lagrangian neighbor patch, including
the background scale-factor change.  Low-density central subhalos define the
field sample; satellites or top-quartile-density objects define a bound-region
proxy.

| snapshots | neighbor retention | field novelty | bound novelty | field expansion/global | bound expansion/global |
|---|---:|---:|---:|---:|---:|
| `86 -> 88` | 96.5% | 0.000 bits | 13.084 bits | 0.982 | 0.460 |
| `88 -> 90` | 96.6% | 0.000 bits | 13.118 bits | 0.989 | 0.455 |

Conditional novelty is positively associated with density (Spearman
$\rho=+0.330$ and $+0.309$) and negatively associated with local expansion
($\rho=-0.121$ and $-0.126$).  Repeating with $k=8,32,64$ preserves the main
environmental result: field patches expand at roughly the background rate,
while the bound proxy expands at only about $0.44$--$0.50$ of it; denser regions
show more neighbor turnover, not less.

To check that hard top-k boundary crossings are not creating the result, a
second estimator assigns continuous Gaussian weights to 64 nearby candidates
and measures the Jensen--Shannon change of the neighborhood distribution.  It
strengthens the same sign:

| snapshots | JS change vs density | JS change vs expansion | field median JS | bound median JS |
|---|---:|---:|---:|---:|
| `86 -> 88` | $\rho=+0.826$ | $\rho=-0.437$ | 0.000035 bits | 0.000843 bits |
| `88 -> 90` | $\rho=+0.818$ | $\rho=-0.462$ | 0.000031 bits | 0.000767 bits |

The continuous neighborhood distribution therefore changes roughly 24 times
more in the bound proxy than in the low-density field sample.

This result is opposite to the simplest claim that *new relational information
locally produces expansion*.  Instead, new neighborhood information in this
simulation is concentrated in gravitationally structured regions, where
mergers, orbital rearrangement, and close-neighbor rank changes occur while
physical expansion is suppressed.  The result is compatible with the user's
second intuition: increasing relational dependency may be associated with
binding and history maintenance rather than with Hubble expansion.

### Phase-space conditioning and replication

The next control conditions explicitly on the old phase-space state.  CAMELS
stores `SubhaloPos` in comoving Mpc/$h$ and `SubhaloVel` as peculiar velocity in
km/s.  Holding the measured old peculiar velocity fixed over one snapshot
interval gives the first-order FLRW prediction

$$
\mathbf{x}_{\rm pred}(a_2)=\mathbf{x}(a_1)
+\frac{\mathbf{v}_{\rm pec}(a_1)}{100}
\int_{a_1}^{a_2}\frac{da}{a^2E(a)},
\qquad
E(a)=\sqrt{\Omega_m a^{-3}+\Omega_\Lambda},
\tag{10}
$$

where $\mathbf{x}$ is in comoving Mpc/$h$.  The phase-conditioned residual is
the Jensen--Shannon distance between the neighborhood distribution predicted
by (10) and the observed distribution at $a_2$.  This is deliberately only a
ballistic control: accelerations, mergers, halo-finder changes, and genuinely
irreversible information remain in the residual.

The test was repeated across five independent fixed-cosmology CAMELS
IllustrisTNG `CV` realizations, two adjacent intervals per realization.  The
reported uncertainty is the sample standard deviation across the ten
realization--interval measurements.

| quantity | field | bound proxy |
|---|---:|---:|
| physical expansion / global expansion | $0.992\pm0.021$ | $0.500\pm0.089$ |
| raw median JS change | $(3.80\pm0.85)\times10^{-5}$ bits | $(8.39\pm1.53)\times10^{-4}$ bits |
| phase-conditioned median JS residual | $(1.33\pm0.32)\times10^{-6}$ bits | $(1.40\pm0.64)\times10^{-4}$ bits |
| fraction of the median removed by velocity conditioning | 96.5% | 83.3% |

The residual remains negatively associated with expansion
($\rho=-0.304\pm0.047$) and positively associated with density
($\rho=+0.629\pm0.032$).  The sign and environmental split therefore replicate,
but most of the neighborhood change is predictable from ordinary peculiar
motion.

A matched gravity-only control uses three pairs of public `IllustrisTNG` and
`IllustrisTNG_DM` Latin-hypercube realizations with the same indices, cosmology,
and initial seed (six realization--interval measurements for each suite):

| quantity | hydrodynamic | gravity only |
|---|---:|---:|
| field expansion / global | $1.000\pm0.011$ | $0.988\pm0.017$ |
| bound expansion / global | $0.533\pm0.062$ | $0.542\pm0.049$ |
| raw field median JS | $(5.28\pm2.13)\times10^{-5}$ bits | $(5.49\pm2.11)\times10^{-5}$ bits |
| raw bound median JS | $(1.041\pm0.046)\times10^{-3}$ bits | $(1.241\pm0.116)\times10^{-3}$ bits |
| residual field median JS | $(2.45\pm2.03)\times10^{-6}$ bits | $(2.06\pm1.32)\times10^{-6}$ bits |
| residual bound median JS | $(2.32\pm1.34)\times10^{-4}$ bits | $(3.21\pm0.56)\times10^{-4}$ bits |

The hydrodynamic and gravity-only results have the same sign and comparable
magnitude.  The qualitative pattern therefore does not require baryonic
structural complexity; collisionless gravity alone produces it.

There are important limits.  CAMELS assumes a LambdaCDM background and cannot
show that information causes or replaces dark energy.  The ballistic residual
is not an entropy production measure: gravitational acceleration, imperfect
object matching, mergers, and halo identification can all contribute.  Even
so, the combined tests falsify the proposed *positive local
novelty--expansion* sign for the direct estimators and substantially narrow the
surviving idea.  What survives is a dependency/binding interpretation:
relational rearrangement is enhanced where gravity suppresses expansion.  A
new cosmological mechanism would require a covariant, coarse-grained
irreversible information functional whose stress--energy contribution is
derived independently, rather than inferred from the already-assumed
LambdaCDM expansion.

## 9. What is ruled out, what survives, and what remains untested

### Ruled out in the tested form

- A fixed effective dimension $d_{\rm eff}$ with
  $N=d_{\rm eff}\log_2(t/t_*)$ as the observed expansion history.
- Literal 3D plus accumulated time, corresponding to $p=4/3$.
- A redshift-independent coupling between $3H$ and the production rate of raw
  causal-past four-volume.
- The original direct local sign claim that more neighborhood novelty should
  accompany more physical expansion.
- The idea that the measured local novelty contrast specifically requires
  baryonic complexity.

### Still mathematically or empirically viable, but not derived

- The exact combinatorial thinning theorem and the kinematic dictionary in
  equation (1).
- A constant late-time conditional-information rate, provided that the
  information quantity can be defined independently and covariantly.
- The two-channel form in equations (6)--(7), provided its quadratic
  composition law, $a^{-3}$ scaling, constant channel, and normalization can be
  derived without fitting cosmology.
- A dependency/binding interpretation in which relational complexity tracks
  constrained gravitational structure rather than causing Hubble expansion.

### Not yet tested decisively

- A genuinely irreversible information measure after conditioning on full
  gravitational dynamics. The present ballistic residual still contains
  acceleration, mergers, disruption, and catalog-systematics effects.
- A covariant local information current or stress--energy tensor that can enter
  Einstein's equations while satisfying conservation and equivalence-principle
  constraints.
- An independently calculated information-production rate that predicts the
  absolute value of $H_0$, $\Omega_m:\Omega_\Lambda$, or $z_{\rm acc}$ before
  comparison with cosmological data.
- Observational signatures distinct from flat $\Lambda$CDM in growth, lensing,
  redshift-space distortions, or gravitational-wave propagation.

## 10. Restart map and paper plan

### Files and their roles

| file | purpose |
|---|---|
| `../relational_shape (2).tex` | untouched source of the original neighborhood-complexity paper |
| `../The_Relational_Shape_of_Structural_Complexity_and_Neighborhood_Density (2).pdf` | untouched compiled original paper |
| `../cosmology_time_note/accumulated_time_cosmology.tex` | focused single-column note on accumulated history, 3D, and why the construction fails |
| `../cosmology_time_note/accumulated_time_cosmology.pdf` | compiled four-page accumulated-time note |
| `COSMOLOGY_HYPOTHESIS.md` | master scientific checkpoint and current interpretation |
| `cosmology_toy_model.py` | age-scale estimate, constant-rate calibration, and turnaround calculation |
| `test_history_models.py` | DESI DR2 BAO shape fits and held-out-redshift tests |
| `test_relativistic_history.py` | causal-past four-volume and coupling-drift test |
| `test_conditional_information.py` | CAMELS matching, neighborhood estimators, FLRW phase propagation, replication, and gravity-only control |
| `make_paper_figures.py` | reproducibly generates the two manuscript figures |
| `complexity_cosmology.bib` | manuscript bibliography with primary observational, theoretical, and simulation sources |
| `neighborhood_thinning_cosmology.tex` | professional short-paper source |
| `neighborhood_thinning_cosmology.pdf` | compiled five-page short paper |
| `figures/` | publication figures in PDF and PNG formats |

The DESI likelihood vector and covariance are in this folder's `data/`
subdirectory. Downloaded CAMELS catalogs are cached under
`data/camels_cv0/` and `data/camels/`. These data directories are ignored by
Git; a fresh checkout must retain or redownload them. The CAMELS test downloads
missing catalogs automatically. The Python dependencies are recorded in the
repository-level `requirements.txt`; `h5py` was added for the HDF5 catalogs.

### Assumptions that must be stated in a paper

- The BAO analysis fits the nuisance scale $c/(H_0r_d)$, so its main result is a
  shape test rather than an absolute-$H_0$ measurement.
- The two-channel model is background-degenerate with flat $\Lambda$CDM.
- CAMELS assumes a LambdaCDM background; it tests local correlations and
  possible interpretations, not whether information creates the background.
- Subhalo identity matching is geometric with a mass cut and is validated only
  where a persistent most-bound particle anchor is available.
- “Bound proxy” means a satellite or an object in the upper local-density
  quartile; it is not a rigorous binding-energy classification.
- Reported replication uncertainties are sample standard deviations across
  realization--interval measurements, not errors on a cosmological parameter.
- The gravity-only comparison uses matched `LH_0`, `LH_1`, and `LH_2`
  IllustrisTNG/IllustrisTNG-DM indices. The five-volume fixed-cosmology result
  uses IllustrisTNG `CV_0` through `CV_4`.

### Best short-paper framing

The strongest defensible paper is an exploratory constraint or no-go study:

> An exact combinatorial relation suggests a possible bridge between relational
> complexity and spatial expansion. Its simplest cosmological dynamics match
> the present Hubble scale accidentally but fail the expansion history; a
> relativistic causal-volume replacement also lacks a constant coupling; and
> simulations place neighborhood change in bound regions rather than expanding
> ones. The remaining viable interpretation concerns relational dependency and
> binding, not a demonstrated origin of dark energy.

The manuscript and first figure set are complete. The next production step is
editorial rather than another parameter fit:

1. perform an author-level wording and emphasis review of the five-page draft;
2. select a target venue or preprint format and adapt its class/style;
3. add repository or archival-data links before public circulation;
4. reserve any new mechanism for future work until an irreversible covariant
   information functional is supplied.

## Reproduction

Run:

```powershell
python complexity_neighborhoods/neighborhood_thinning_cosmology/cosmology_toy_model.py
python complexity_neighborhoods/neighborhood_thinning_cosmology/test_history_models.py
python complexity_neighborhoods/neighborhood_thinning_cosmology/test_relativistic_history.py
python complexity_neighborhoods/neighborhood_thinning_cosmology/test_conditional_information.py
python complexity_neighborhoods/neighborhood_thinning_cosmology/test_conditional_information.py --realizations 0 1 2 3 4
python complexity_neighborhoods/neighborhood_thinning_cosmology/test_conditional_information.py --simulation-set LH --realizations 0 1 2
python complexity_neighborhoods/neighborhood_thinning_cosmology/test_conditional_information.py --suite IllustrisTNG_DM --simulation-set LH --realizations 0 1 2
```

Published calibration sources:

- Planck Collaboration, *Planck 2018 results VI: Cosmological parameters*,
  <https://arxiv.org/abs/1807.06209>
- DESI Collaboration, *DESI DR2 Results II*,
  <https://arxiv.org/abs/2503.14738>
- Riess et al., 2022 SH0ES distance ladder,
  <https://arxiv.org/abs/2112.04510>
- Pavlidou and Tomaras, maximum turnaround radius,
  <https://arxiv.org/abs/1310.1920>
- DESI DR2 BAO likelihood vector and covariance as distributed for Cobaya,
  <https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2>
- Villaescusa-Navarro et al., *The CAMELS Project: Public Data Release*,
  <https://arxiv.org/abs/2201.01300>
- Illustris data specifications (catalog field definitions and units),
  <https://www.illustris-project.org/data/docs/specifications/>
