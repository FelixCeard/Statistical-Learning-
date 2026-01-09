# Part A
## Question 1
### (a)
$$P(X_i=1)=a\pi+(1-a)(1-\pi)=(1-a)+(2a-1)\pi.$$

### (b)
Joint Probability : $\prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i}$
Likelihood: $L(\pi)=\prod_{i=1}^n p(\pi)^{x_i}\,[1-p(\pi)]^{1-x_i}
      =[(1-a)+(2a-1)\pi]^m[a+(1-2a)\pi]^{n-m}$

###  (c)
$X_i \sim Benouli(X) $, so MLE of $p$ is $\hat p = m/n$

$(1-a)+(2a-1)\pi = m/n$
$\hat \pi_{ML} = \frac{\frac{m}{n}-(1-a)}{2a-1},
\qquad a\neq 0.5$

### (e)

$E(\hat\pi_{ML})
=\frac{E(\hat p)-(1-a)}{2a-1}
=\frac{(1-a)+(2a-1)\pi-(1-a)}{2a-1}
=\pi.$


### (f)
$
\sum_{i=1}^n X_i \sim \mathrm{Binomial}(n, p(\pi)).
$
Let $m=\sum_{i=1}^n X_i$ and $\hat p = m/n$,
$
\mathrm{Var}(\hat p)=\mathrm{Var}\!\left(\frac{m}{n}\right)
=\frac{1}{n^2}\mathrm{Var}(m)
=\frac{1}{n^2}\,n p(\pi)\big(1-p(\pi)\big)
=\frac{p(\pi)\big[1-p(\pi)\big]}{n}.
$

$
\mathrm{Var}(\hat \pi_{ML}) = \mathrm{Var}(\frac{\hat p+a-1}{2a-1})=\frac{\mathrm{Var}(\hat p)}{(2a-1)^2}=\frac{p(\pi)[1-p(\pi)]}{n(2a-1)^2}
$
Variance is dominated by the factor $(2a-1)^{-2}$, it is symmetric around a=0.5, and diverges as $a\to 0.5$. while $a$ moves away from 0.5 toward 0 or 1 it decreases.
