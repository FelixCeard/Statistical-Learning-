// definitions
#let Binomial = "Binomial"

= Basic definition


Let $X_i$ denote one animal. Let $X_i^t in {0, 1}$ denote whether the animal is being caught at time step $t$.

We have that $Pr(X_i^t = 1) = p$ and $Pr(X_i^t = 0) = 1-p$ with $p tilde Pr(dots.c)$ from an arbitrary probability distirbution.

We can now define $f_k$ as follows:
$
  f_k = sum_(t=1)^N cases(1 & " if " sum_(t=1)^T X_i^t = k, 0 & " otherwise")
$
with 
$
  F_k = {X_i : i in [1, N] mid(|) (sum_(t=1)^T X_i^t) = k}
$

Therefore, we have that
$
  Pr(X_i in F_k mid(|) T=t) &= binom(t, k) dot Pr(X_i^t = 1)^k dot Pr(X_i^t = 0)^(t-k) \
  &= binom(t, k) dot p^k dot (1-p)^(t-k) \
  &= "Binomial"(t, k, p)
$

Therefore, we have that for $N$ independent $X_i$, we have the likelihood 

We now compute the expected number of times an animal is being caught after $T=t$ timesteps:
$
  EE[k : X_i in F_k mid(|) T=t] &= sum_(k=0)^t k dot Pr(X_i in F_k mid(|) T=t) \
  &= sum_(k=0)^t k dot binom(t, k) dot p^k dot (1-p)^(t-k)
$
which, given that we have four possible distributions for $p$, we compute the expected value for each distribution and report on the results in 

#figure(
  table(
    align: center,
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    rows: 2,
    [*Distribution*], [$U(0, 0.01)$], [$U(0, 0.02)$], [$0.02 dot "beta"(1, 3)$], [$0.01 dot "beta"(2, 2)$],
    [*Expected value*], [$0.4$], [$0.8$], [$0.8$], [$0.4$],
  ),
  caption: [Expected number of times an animal is being caught after $T=40$ timesteps]
)

We have, for $p tilde 0.02 dot "Beta"(1, 3)$:
$
  P(X_i^t = 1) &= integral_0^1 0.02 dot "Beta"(1, 3) \
  &= 0.02 dot integral_0^1 "Beta"(1, 3) \
  &= 0.02 dot 1.
$

We have
$
  Pr(f_2 = 0 mid(|) T=t, N=n) &= (1 - Pr(X_i in F_2 mid(|) T=t))^n \
  &= (1 - "Binomial"(t, 2, p))^n
$ 
For $p tilde (0, 0.01)$ and $0.01 dot "Beta"(1, 3)$, we have that
$
  Pr(f_2 = 0 mid(|) T=t, N=n) &= (1 - "Binomial"(t, 2, 0.01))^n 
$
and for $p tilde U(0, 0.02)$ and $0.02 dot "Beta"(1, 3)$, we have that
$
  Pr(f_2 = 0 mid(|) T=t, N=n) &= (1 - "Binomial"(t, 2, 0.02))^n 
$
Therefore, for our setting for $N=500$ and $T=40$, we have
#figure(
  table(
    align: center,
    columns: (2fr, 1fr, 1fr),
    [*Distribution*], [$U(0, 0.01)$], [$U(0, 0.02)$], 
    [$Pr(f_2 = 0 mid(|) T=40, N=500)$], [$1.3dot e^(-12)
$], [$1.1 dot e^(-34)$],
    [$Pr(f_2 = 0 mid(|) T=40, N=1000)$], [$1.7 dot e^(-24)$], [$1.17 dot e^(-68)$],
    [$Pr(f_2 = 0 mid(|) T=40, N=5000)$], [$1.5 dot e^(-119)$], [$0.0$],
  ),
  caption: [Expected number of times an animal is being caught after $T=40$ timesteps]
)

Furthermore, for an arbitrary $k$, we have that 
$
  Pr(f_k = i mid(|) T=t, N=n) &= "Binomial"(n, i, "Binomial"(t, k, p))
$

= another test (this works out I think)
$
  EE[n mid(|) T=t, N] &= EE[sum_(t=1)^T f_t mid(|) T=t, N] \
  &= sum_(t=1)^T EE[f_t mid(|) T=t, N] \
  &= sum_(t=1)^T sum_(i = 0)^N i dot Pr(f_t = i mid(|) T=t, N) \
  &= sum_(t=1)^T sum_(i = 1)^N i dot "Binomial"(N, i, Binomial(T, t, p)) 
$
which, for $T=40$ and $N in {500, 1000, 5000}$, we have to compute using python, as finding a closed form solution is not easily feasible, and might not bring us any additional insights.
We have:
#figure(
  table(
    align: center,
    columns: (2fr, 1fr, 1fr),
    [*Distribution*], [$U(0, 0.01)$], [$U(0, 0.02)$], 
    [$EE[n mid(|) T=40, N=500]$], [$165.51$], [$277.15$],
    [$EE[n mid(|) T=40, N=1000]$], [$331.03$], [$554.3$],
    [$EE[n mid(|) T=40, N=5000]$], [$1655.14$], [$2771.5$],
  ),
  caption: [Expected value of $n$ for $T=40$ and $N in {500, 1000, 5000}$]
)

== Other expected value

We distinguish between two cases:
- $f_2 = 0$
- $f_2 > 0$

$
  EE[hat(f_0) mid(|) T=t, N=n, f_2 > 0] &= EE[f_1^2 / (2 f_2) mid(|) T=t, N=n] \
  &= sum_(i=1)^N sum_(j=1)^(N-i) i^2 / (2j) dot Pr(f_1 = i, f_2 = j mid(|) T=t, N=n) \ &" "+sum_(j=1)^N sum_(i=1)^(N-j) i^2 / (2j) dot Pr(f_1 = i, f_2 = j mid(|) T=t, N=n)
$
Because we can model the distribution of $f_i$ as a multinomial distribution,#footnote(link("https://en.wikipedia.org/wiki/Multinomial_distribution")) we can compute the joint probability of $f_1$ and $f_2$ as follows:
$
  Pr(f_1 = i, f_2 = j mid(|) T=t, N=n) &= (i+j)!/(i! dot j!) Pr(f_1 = i)^i
$

// We have that
// $
//   Pr(f_1 = i, f_2 = j mid(|) T=t, N=n) &= Pr(f_1=i mid(|) T=t, N=n, f_2=j) dot Pr(f_2 = j mid(|) T=t, N=n)
// $
// where 
// $
//   Pr(f_1=i mid(|) T=t, N=n, f_2=j) &= 
// $