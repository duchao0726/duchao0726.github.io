---
title: Off-Policy Variance Reduction through Baseline Placement
date: November 2025
updated: October 2026
description: Is a single reward baseline the right structure for off-policy RL in LLMs? A theory of multiple baseline placements, their joint variance optimum, and the variance cost of estimating the baselines.
---

*These notes were written in the autumn of 2025 and last touched in November of that year, then left in a drawer. I am posting them in October 2026 essentially as they were, with a short final section on what has appeared in the meantime. There are no experiments here. The derivation is elementary, and that is rather the point: it falls out of writing the off-policy gradient down carefully and asking where to place the baseline.*

*I thank [Xiangxin Zhou](https://zhouxiangxin1998.github.io/), [Min Lin](https://linmin.me/), [Tianyu Pang](https://p2333.github.io/), and [Penghui Qi](https://scholar.google.com/citations?user=CLRsGEMAAAAJ) for helpful discussions back then, when we were all at Sea AI Lab.*

## <span id="setup">1. Setup</span>

A prompt $x$, a response $y=(y_1,\dots,y_L)$ generated token by token, and a scalar reward $R(x,y)$ revealed at the end. Two policies matter: the policy being trained, $\pit$, and the behavior policy $\pib$ that generated the data. They differ whenever rollouts are stale or come from another model, or when the inference and training engines disagree numerically.

For a fixed prompt $x$, the objective (omitting entropy and KL regularization) is $J(\theta;x)\triangleq\E_{y\sim\pit(\cdot|x)}[R(x,y)]$. Its gradient, written under the behavior distribution, is

$$
\begin{aligned}
\nabla_\theta J
&=\E_{y\sim\pib(\cdot|x)}\Big[\,{w(x,y)}\,R(x,y)\,\nabla_\theta\log\pit(y|x)\Big],\\
{w(x,y)}&{\triangleq}\frac{\pit(y|x)}{\pib(y|x)}=\prod_{u=1}^{L} r_u,
\qquad
r_u{\triangleq}\frac{\pit(y_u|x,y_{<u})}{\pib(y_u|x,y_{<u})}{,}
\end{aligned}
\tag{1}
$$

where $w(x,y)$ is the sequence importance weight and $r_u$ the token ratio. We take $L$ fixed for simplicity and assume ${\pib>0}$ wherever ${\pit>0}$.

Subtracting a constant from the reward leaves \eqref{1} unchanged, because $\E_{\pib}[\,w\,\nabla_\theta\log\pit(y|x)\,]=\E_{\pit}[\nabla_\theta\log\pit(y|x)]=0$. So for any ${b=b(x)}$ that is constant across responses to this prompt,

$$
\nabla_\theta J=\E_{y\sim\pib(\cdot|x)}\Big[\,{w(x,y)}\,\big(R(x,y)-b\big)\,\nabla_\theta\log\pit(y|x)\Big].
$$

This is the reward baseline. On-policy, ${w=1}$ and the group mean of rewards is a common choice [@shao2024;zheng2025]. Off-policy, $w$ is a product of many token ratios and can be highly variable, so in practice it is often truncated or masked [@yao2025;liu2025]. The question here is a different one from how to tame $w$: <strong>once the weight is there, is a single constant baseline still the right structure?</strong>

## <span id="weight-factorization">2. Splitting the weight along the sequence</span>

Write the score vector as a sum over positions, $\nabla_\theta\log\pit(y|x)=\sum_{t=1}^{L}s_t$ with $s_t{\triangleq}\nabla_\theta\log\pit(y_t|x,y_{<t})$. For a fixed $t$ the weight factorizes into a prefix part and a suffix part,

<div class="equation-group">
<div class="equation-parts">
<span class="equation-part">$\displaystyle {w(x,y)}=\rho_{\le t}\,\rho_{>t},$</span>
<span class="equation-part">$\displaystyle \rho_{\le t}{\triangleq}\prod_{u\le t}r_u=\frac{\pit(y_{\le t}|x)}{\pib(y_{\le t}|x)},$</span>
<span class="equation-part">$\displaystyle \rho_{>t}{\triangleq}\prod_{u>t}r_u=\frac{\pit(y_{>t}|x,y_{\le t})}{\pib(y_{>t}|x,y_{\le t})},$</span>
</div>
</div>

so that \eqref{1} becomes

$$
\nabla_\theta J=\sum_{t=1}^{L}\E_{y\sim\pib(\cdot|x)}\Big[\,\rho_{\le t}\,s_t\cdot\big(\rho_{>t}\,R\big)\Big].
\tag{2}
$$

Read \eqref{2} from the point of view of position $t$. The prefix weight $\rho_{\le t}$ corrects the distribution of $y_{\le t}$ from $\pib$ to $\pit$. Everything about the future sits in the bracket:

$$
\tilde R_t{\triangleq}\rho_{>t}\,R .
$$

Call $\tilde R_t$ the **equivalent reward** at position $t$: the return, rescaled by how likely the realized future is under the current policy relative to the behavior policy. The following one-line fact explains the name.

**<span id="lemma-1">Lemma 1.</span>** $\Qt(x,y_{\le t})\triangleq\E_{y_{>t}\sim\pit(\cdot|x,y_{\le t})}\big[R(x,y)\big]=\E_{\pib}\big[\tilde R_t \,\big|\, x,y_{\le t}\big]$.

*Proof.* Change of measure on the suffix: $\sum_{y_{>t}}\pib(y_{>t}|x,y_{\le t})\,\rho_{>t}\,R=\sum_{y_{>t}}\pit(y_{>t}|x,y_{\le t})\,R$. $\square$

So $\tilde R_t$ is a single-sample unbiased estimate of the action value of the prefix *under the policy being trained*, and \eqref{2} has the usual policy-gradient form (as in per-decision importance sampling with a terminal reward [@precup2000]).

*If one drops $\rho_{>t}$* and pairs the prefix weight $\rho_{\le t}$ with the raw $R$, this replaces $\Qt$ with $\Qb(x,y_{\le t})\triangleq\E_{\pib}[R\,|\,x,y_{\le t}]$, an exact substitution on-policy and an approximation otherwise.

## <span id="baseline-placements">3. Where does the baseline go?</span>

Now put the baseline back. With a constant baseline $b$, the contribution of token $t$ to the gradient estimator is $w(R-b)s_t$, which factorizes as

$$
{\rho_{\le t}\,s_t\cdot\rho_{>t}\,\big(R-b\big){:}}
\tag{3}
$$

the baseline is subtracted from the raw reward, *before* the suffix weight multiplies it. But \eqref{2} suggests another option, subtracting it from the equivalent reward, *after* the suffix weight:

$$
{\rho_{\le t}\,s_t\cdot\big(\rho_{>t}\,R-b\big).}
\tag{4}
$$

Both are unbiased, and the same construction applies at every position for a fixed score term. At ${L-1}$, the suffix is just the last token and the equivalent reward is $r_LR$. Applying both baseline placements gives $r_L(R-b_L)-b_{L-1}$. At ${L-2}$, this whole expression becomes the reward term: multiply it by $r_{L-1}$ and subtract $b_{L-2}$. Continuing backward to position $t$ gives

$$
{r_{t+1}\Big(r_{t+2}\Big(\cdots\Big(r_L\big(R-b_L\big)-b_{L-1}\Big)\cdots\Big)-b_{t+1}\Big)-b_t .}
$$

With only $b_L$, the baseline is subtracted from $R$, recovering \eqref{3}; with only $b_t$, it is subtracted from $\tilde R_t$, recovering \eqref{4}.

<aside class="intuition-box" aria-labelledby="backward-intuition">
<p class="intuition-title"><strong id="backward-intuition">Intuition.</strong></p>

<p>Consider two successful rollouts, both with ${R=1}$. With ${b_L=0.5}$, both initially have a positive advantage of $0.5$. Suppose their final-token importance ratios are $2$ and $0.2$. Reweighting, then subtracting ${b_{L-1}=0.2}$, gives:</p>

<figure class="baseline-placement-figure" id="baseline-placement-figure">
<div class="figure-scroll" tabindex="0" role="region" aria-label="Baseline placement diagram">
<img src="figures/baseline-placement.svg" width="1600" height="535" alt="Two successful rollouts propagate a learning signal backward across three token columns. At y_L, reward R=1 minus baseline b_L=0.5 gives advantage 0.5. Multiplication by token ratio r_L=2 or 0.2 gives 1.0 or 0.1. At y_(L−1), subtracting b_(L−1)=0.2 gives advantage +0.8 or −0.1. Multiplication by r_(L−1) carries each result to the next subtraction at y_(L−2). A horizontal Baselines row aligns each coefficient with its subtraction position. Multiplication and subtraction use circular operation nodes; arrows point to the result of each operation.">
</div>
<figcaption class="figure-scroll-hint">Scroll to view the full diagram.</figcaption>
</figure>

<p>Both rollouts succeeded, but the current policy is less likely than the behavior policy to reproduce the second rollout’s successful final token. After centering, that rollout gives the preceding token a negative advantage. Repeating this reweighting and centering carries the learning signal backward through the response.</p>
</aside>

The construction above considers a fixed score term $s_t$. More generally, each score term may use its own family of baselines, denoted by $b_{t,s}$, where $s$ specifies the baseline position. Define the partial suffix weights

$$
{\rho_{t,s}}{\triangleq}\prod_{u=t+1}^{s}r_u{=\frac{\pit(y_{t+1:s}\mid x,y_{\le t})}{\pib(y_{t+1:s}\mid x,y_{\le t})}},\qquad s=t,\dots,L,
$$

so that ${\rho_{t,t}=1}$ and ${\rho_{t,L}=\rho_{>t}}$. Expanding the nested expression and restoring the prefix factor $\rho_{\le t}s_t$ gives the token contribution

$$
{g_t}{\triangleq}\rho_{\le t}\,s_t\Big({\rho_{t,L}}R-\sum_{s=t}^{L}{\rho_{t,s}}{b_{t,s}}\Big){.}
\tag{5}
$$

**<span id="lemma-2">Lemma 2.</span>** For any coefficients $b_{t,s}$ that may depend on $x$, $t$, and $s$ but not on $y$, the token contributions in \eqref{5} satisfy

$$
{\sum_{t=1}^{L}\E_{\pib}\big[g_t\big]=\nabla_\theta J .}
$$

*Proof.* Since ${\rho_{\le t}\rho_{t,s}=\rho_{\le s}}$, changing measure on the prefix $y_{\le s}$ gives $\E_{\pib}[\rho_{\le t}s_t\rho_{t,s}]=\E_{\pib}[\rho_{\le s}s_t]=\E_{\pit}[s_t]=0$. Thus every baseline correction has zero expectation; summing the remaining reward terms gives \eqref{2}. $\square$

Hence, for a rollout ${y\sim\pib(\cdot|x)}$, $\hat g\triangleq\sum_{t=1}^L g_t$ is an unbiased estimator of the gradient $\nabla_\theta J$ in \eqref{2}.

On-policy, ${\rho_{t,s}=1}$ for all ${t\le s}$, so the baselines for each score term combine into a single baseline, $\sum_{s=t}^L b_{t,s}$. Only their sum matters; off-policy, their placement matters too.

## <span id="variance-optimal-baselines">4. The variance-optimal baselines</span>

We can choose the baselines to minimize the variance $\E_{\pib}\|\hat g-\nabla_\theta J\|^2$ of the gradient estimator $\hat g$, equivalently its second moment $\E_{\pib}\|\hat g\|^2$.

Summing \eqref{5} over $t$ and collecting the baseline terms gives

$$
{\hat g=\sum_{t=1}^L g_t=\hat g_0-\sum_{1\le t\le s\le L}b_{t,s}\rho_{\le s}s_t,}
$$

where $\hat g_0\triangleq wR\sum_t s_t$ is the estimator without baselines. Since $\partial\hat g/\partial b_{t,s}=-\rho_{\le s}s_t$, differentiating the second moment gives $\frac{\partial}{\partial b_{t,s}}\E_{\pib}\|\hat g\|^2=-2\E_{\pib}[\rho_{\le s}s_t^{\top}\hat g]$. Setting the derivative to zero gives

<div class="eq-scroll">

$$
{\sum_{1\le k\le v\le L}\E_{\pib}[\rho_{\le s}\rho_{\le v}s_t^{\top}s_k]\,b_{k,v}
=\E_{\pib}[\rho_{\le s}s_t^{\top}\hat g_0].}
\tag{6}
$$

</div>

Here $(t,s)$ identifies the baseline $b_{t,s}$ being optimized; $(k,v)$ runs over all baselines, with $k$ indexing the score term and $v$ the baseline position. This is a linear equation in the baselines $b_{k,v}$. Repeating the differentiation for each $b_{t,s}$ gives a <strong>linear system</strong>, retaining interactions between baseline placements and across score terms. Solving this system gives the jointly optimal baseline values that minimize the variance of $\hat g$ within this family (see the <a href="#general-system-solution">appendix</a>).

The minimum variance is non-increasing as more baseline placements are allowed: the original family is recovered by setting the new baselines to zero.

## <span id="practical-baseline-choices">5. Two baseline structures</span>

The linear system in \eqref{6} is defined by expectations involving rewards, importance ratios, and score inner products $s_t^\top s_k$. Rewards and ratios are available from rollout data and a forward pass, while the usual training backward pass does not give the individual score vectors $s_t$.

We can consider a regime in which squared score norms are approximately constant across tokens and rollouts, ${\|s_t\|^2\approx\kappa>0}$, while cross-token inner products are small relative to this scale, ${|s_t^\top s_k|\ll\kappa}$ for ${t\ne k}$ [@hao2025;chen2026holder;fang2026dynamo]. Under this diagonal approximation, $\kappa$ cancels from \eqref{6}, giving

<div class="equation-group equation-group-numbered equation-anchor" id="eq-7">
<div class="equation-parts">
<span class="equation-part">$\displaystyle \sum_{v=t}^L\E_{\pib}[\rho_{\le s}\rho_{\le v}]\,b_{t,v}=\E_{\pib}[\rho_{\le s}wR],$</span>
<span class="equation-part">$\displaystyle s=t,\dots,L.$</span>
</div>
<span class="equation-number">$(7)$</span>
</div>

This allows us to derive baseline formulas without computing score inner products (the resulting gradient estimators remain unbiased). We illustrate this with two baseline structures below.

### Two endpoints for each score term

For each ${t<L}$, keep only the raw-reward baseline $b_{t,L}$ and the equivalent-reward baseline $b_{t,t}$ in \eqref{5}:

<div class="equation-group">
<div class="equation-parts">
<span class="equation-part">$\displaystyle g_t=\rho_{\le t}s_t\left[\rho_{>t}(R-b_{t,L})-b_{t,t}\right].$</span>
</div>
</div>

Setting either baseline to zero recovers \eqref{3} or \eqref{4}. Optimizing them jointly amounts to solving a ${2\times2}$ system obtained from \eqref{7} at each ${t<L}$, with a closed-form solution. At ${t=L}$, the two endpoints coincide, leaving only $b_{L,L}$.

Under this approximation, the joint optimum has no higher variance than that obtained by optimizing either endpoint alone. The <a href="#practical-coefficient-derivations">appendix</a> gives the solution and variance reduction.

### One shared baseline at each position

Share each position’s baseline across score terms, ${b_{t,s}=b_s}$. The advantages follow the backward recursion from [Section 3](#baseline-placements):

<div class="equation-group equation-group-numbered equation-anchor" id="eq-8">
<div class="equation-parts">
<span class="equation-part">$\displaystyle a_L\triangleq R-b_L,$</span>
<span class="equation-part">$\displaystyle a_t\triangleq r_{t+1}a_{t+1}-b_t,$</span>
<span class="equation-part">$\displaystyle g_t=\rho_{\le t}s_ta_t.$</span>
</div>
<span class="equation-number">$(8)$</span>
</div>

The shared baselines also admit a closed-form solution; see the <a href="#shared-position-derivation">appendix</a>. An incremental view connects this structure to the usual reward baseline.

**Incremental construction.** Initialize ${b_1=\cdots=b_L=0}$; updating $b_L$ first recovers the optimal raw-reward baseline under our approximation:

$$
{b_L\leftarrow\frac{\E_{\pib}[w^2R]}{\E_{\pib}[w^2]}.}
$$

Moving backward, optimize $b_{L-1},\dots,b_1$ in turn while holding the other baselines fixed. Each nonzero update further reduces the approximate variance. Repeating these backward sweeps gives <strong>Gauss–Seidel iteration</strong>.

## <span id="estimating-from-rollouts">6. Estimating from a group of rollouts</span>

The preceding solutions involve intractable expectations, usually estimated from a group of rollouts. The resulting estimation error affects the gradient variance.

### Fitting and applying the baselines

Let $\mathcal D=\{y^{(1)},\dots,y^{(G)}\}$ contain $G$ independent rollouts from $\pib(\cdot|x)$. Fit the <strong>leave-one-out</strong> (LOO) baselines $\hat b_{t,s}^{(-i)}$ by replacing the expectations in \eqref{6} or \eqref{7} with averages over all rollouts except $i$. Substituting them into the token gradient contributions $g_t$ in \eqref{5} gives $g_t^{(-i)}$ and the group estimator

$$
{\hat g_G\triangleq\frac1G\sum_{i=1}^G\sum_{t=1}^L g_t^{(-i)}(y^{(i)}).}
\tag{9}
$$

Each baseline is independent of the rollout it corrects, so [Lemma 2](#lemma-2) gives ${\E_{\mathcal D}[\hat g_G]=\nabla_\theta J}$. This does not require the fitted baselines to be unbiased estimates of their population values; ratios of sample moments generally are not.

### The variance cost of baseline error

For a fixed baseline structure, let $\hat g^\star$ denote the single-rollout estimator with variance-optimal baselines determined by exact expectations (no LOO fitting is needed). Averaging it over the same group gives $\hat g_G^\star$. Then

$$
{\mathcal V_G\triangleq\E_{\mathcal D}\|\hat g_G-\nabla_\theta J\|^2
=\frac{V^\star}{G}+\E_{\mathcal D}\|\hat g_G-\hat g_G^\star\|^2,}
\tag{10}
$$

where $V^\star\triangleq\E_{\pib}\|\hat g^\star-\nabla_\theta J\|^2$ is the minimum single-rollout variance. The first term is the Monte Carlo variance with optimal baselines for the chosen structure; the second is the additional variance from using estimated baselines.

The identity is exact. Baseline error includes finite-sample estimation error (which can be amplified when solving for the coefficients) and error from approximating score inner products in [Section 5](#practical-baseline-choices). The <a href="#finite-group-variance">appendix</a> further separates the contributions of baseline bias and fitting noise.

A richer baseline structure can lower the first term while increasing the second.

*Whether these baselines help in RL training, and whether they let us train further off-policy, need to be tested empirically. I’ve since joined AMI Labs to work on world models, so I haven’t had much time to take this further. The following literature review was prepared with Astra’s assistance in October 2026.*

## <span id="subsequent-developments">7. What has appeared since (added October 2026)</span>

Several related works appeared while these notes remained unpublished. VCPO [@huang2026] derives an off-policy baseline weighted by squared importance ratios and squared sequence-score norms. In the notation here, this is a single baseline at the raw-reward end. Variance-aware scalar baselines have also been studied through covariance analysis [@han2026], with KL regularization [@oblr2025], and with reward–confidence corrections [@rcc2026].

The Optimal Token Baseline [@li2026] studies token-dependent baselines, with cumulative score-norm weighting and a forward-pass proxy for those norms. It addresses the same computational issue raised in [Section 5](#practical-baseline-choices): how to compute baselines without extracting individual token gradients. Related work on shrinkage and historical statistics [@shrinkage2025;bvblend2026] and learned or mixed baselines [@tether2026;evpo2026;poise2026] studies how baseline estimation affects variance, as does [Section 6](#estimating-from-rollouts).

Cumulative-token importance sampling [@zhangct2026] and PNPO [@zhangpn2026] revisit prefix correction: the former uses cumulative ratios, while the latter replaces each prefix product by a geometric mean. FIPO [@fipo2026] reweights token advantages using a clipped, discounted product of likelihood ratios, while BRACE and COPC [@brace2026;copc2026] correct return and advantage estimates from off-policy continuations. [Section 2](#weight-factorization) separates the prefix and continuation corrections.

There are also connections to earlier RL work: per-decision importance sampling [@precup2000], optimal reward baselines [@weaver2001;greensmith2004], and sequential control variates [@jiang2016;huangjiang2020]. TrajCV [@cheng2020] and its LLM reformulation ABC [@abc2026] provide precedents for trajectory-level controls and their variance analysis.

Several ingredients of the analysis above therefore also appear in the recent LLM literature, but I have not found an explicit treatment in that literature of multiple baseline placements along the suffix importance product for the same score term. The notes here develop this freedom into a family of unbiased off-policy gradient estimators, characterize its joint variance optimum, and give an exact account of the variance cost of estimating the baselines.

## <span id="derivations-and-extensions">Appendix: Derivations and extensions</span>

We keep the prompt $x$ and both policies $\pit$ and $\pib$ fixed, use exact importance ratios, and assume the same support and finite second moments. Expectations $\E$ below are taken under $\pib$. The results are exact unless the approximation in \eqref{7} is invoked.

### <span id="general-system-solution">Admissible control variates and variance minimization</span>

Equation \eqref{5} can be viewed as using a baseline function for each score term. Writing $f_t^{(b)}(x,y)\triangleq\sum_{s=t}^L b_{t,s}\rho_{t,s}$, where $b$ collects the baseline values, gives

<div class="eq-scroll">

$$
g_t=\rho_{\le t}s_t\bigl(\tilde R_t-f_t^{(b)}\bigr)
=w\,s_t\left(R-\frac{f_t^{(b)}}{\rho_{>t}}\right).
$$

</div>

Thus $f_t^{(b)}/\rho_{>t}$ is a raw-reward baseline function that can depend on the full response. The equivalent-reward form keeps $f_t^{(b)}$ linear in the partial suffix weights $\rho_{t,s}$.

More generally, any integrable $f_t$ satisfying ${\E[\rho_{\le t}s_tf_t]=0}$ is an admissible baseline function: subtracting its gradient correction preserves the expectation of the token contribution $g_t$. The full-gradient estimator $\hat g_f\triangleq\hat g_0-\sum_t\rho_{\le t}s_tf_t$ has variance $\mathcal V[f]\triangleq\E\|\hat g_f-\nabla_\theta J\|^2$, where $f\triangleq(f_1,\ldots,f_L)$. Under the inner product $\langle U,V\rangle\triangleq\E[U^\top V]$, a simple variational calculation gives the orthogonality condition for the optimal estimator $\hat g_{f^\star}$:

<div class="eq-scroll">

$$
\E\!\left[\hat g_{f^\star}^{\top}\sum_{t=1}^L\rho_{\le t}s_th_t\right]=0,
\tag{11}
$$

</div>

for every admissible $h\triangleq(h_1,\ldots,h_L)$. Taking ${h=f-f^\star}$ gives the exact variance decomposition for any admissible $f$:

<div class="eq-scroll">

$$
\mathcal V[f]=\mathcal V[f^\star]
+\E\left\|\sum_{t=1}^L\rho_{\le t}s_t(f_t-f_t^\star)\right\|^2.
$$

</div>

The family $f_t^{(b)}$ is a computable choice within this larger class: the ratios are available from forward passes through the two policies (by [Lemma 2](#lemma-2), every member satisfies the zero-mean condition and is therefore admissible). Within this family, varying $b_{t,s}$ changes $f_t$ in the direction $\rho_{t,s}$. The resulting control variates $\phi_{t,s}$ have zero mean, and their weighted sum $C_b$ gives the gradient correction:

<div class="eq-scroll">

$$
\phi_{t,s}\triangleq\rho_{\le s}s_t,
\qquad C_b\triangleq\sum_{1\le t\le s\le L}b_{t,s}\phi_{t,s}.
$$

</div>

Restricting the optimality condition \eqref{11} to these coefficient variations gives the linear system \eqref{6} for the jointly optimal baseline values. Its matrix is a positive semidefinite Gram matrix; a pseudoinverse handles redundant directions (as when placements for a fixed score term coincide on-policy). The optimal estimator is unique in mean square even when its coefficients are not. The corrections form $\mathcal C\triangleq\{C_b:b\}=\operatorname{span}\{\phi_{t,s}\}$. The optimal correction $C^\star$ is the projection of $\hat g_0$ onto this space, leaving ${\hat g^\star=\hat g_0-C^\star}$.

This also identifies when an additional placement helps. Given a new zero-mean control variate $\phi$, remove its projection onto $\mathcal C$, leaving $\phi^\perp$. Optimizing the coefficient of this remaining direction lowers the minimum variance by

<div class="eq-scroll">

$$
\Delta V=
\frac{\big(\E[(\hat g^\star)^\top\phi^\perp]\big)^2}
{\E\|\phi^\perp\|^2}.
\tag{12}
$$

</div>

A zero denominator means the new control is redundant. Otherwise, the gain is strict precisely when it correlates with the residual left by the existing baselines.

### <span id="martingale-placements">The martingale structure of baseline placements</span>

Write $\mathcal F_s$ for the information in $x,y_{\le s}$, with $\mathcal F_0$ containing only $x$. For fixed $t$, the partial suffix weights $\rho_{t,s}$ form a martingale in $s$:

<div class="eq-scroll" id="suffix-weight-martingale">

$$
\E[\rho_{t,s}\mid\mathcal F_{s-1}]=\rho_{t,s-1}
\qquad(s>t).
$$

</div>

Their successive differences reveal the structure of the joint problem. Set $c_{t,u}\triangleq\sum_{s=u}^L b_{t,s}$, with ${c_{t,L+1}=0}$, and define

<div class="eq-scroll">

$$
X_{t,u}\triangleq
\begin{cases}
\rho_{\le u}s_u,&t=u,\\
(\rho_{\le u}-\rho_{\le u-1})s_t,&t<u.
\end{cases}
\qquad
C_b=\sum_{u=1}^L\sum_{t=1}^u c_{t,u}X_{t,u}.
\tag{13}
$$

</div>

The second identity follows by telescoping. Each $X_{t,u}$ has conditional mean zero given $\mathcal F_{u-1}$: the diagonal case uses the score identity, and the other case uses the [martingale property above](#suffix-weight-martingale). For ${u<v}$, $X_{t,u}$ is known given $\mathcal F_{v-1}$, so $\E[X_{t,u}^{\top}X_{k,v}]=\E[X_{t,u}^{\top}\E[X_{k,v}\mid\mathcal F_{v-1}]]=0$. These inner products are the Gram-matrix entries in the $c_{t,u}$ coordinates. The joint system thus splits into time blocks of sizes $1,\ldots,L$, while score terms within each block remain coupled. The projections of $\hat g_0$ onto $\operatorname{span}\{X_{t,u}:1\le t\le u\}$ for each time block $u$ sum to the jointly optimal correction; the variance reductions are additive. On-policy, the ${t<u}$ increments vanish, recovering the coincident placements for each score term.

### <span id="baseline-structure-solutions">Solutions for the two baseline structures</span>

**<span id="practical-coefficient-derivations">Two endpoints for each score term.</span>** Under \eqref{7}, the joint optimum for ${t<L}$ is

<div class="eq-scroll">

$$
\begin{pmatrix}b_{t,L}^\star\\b_{t,t}^\star\end{pmatrix}
=
\begin{pmatrix}
\E[w^2]&\E[w\rho_{\le t}]\\
\E[w\rho_{\le t}]&\E[\rho_{\le t}^2]
\end{pmatrix}^{\dagger}
\begin{pmatrix}\E[w^2R]\\\E[wR\rho_{\le t}]\end{pmatrix}.
$$

</div>

Here $\dagger$ denotes the Moore–Penrose pseudoinverse. Let $V_t^{\mathrm{raw}},V_t^{\mathrm{equiv}},V_t^{\mathrm{joint}}$ denote the token-gradient variances with the respective optimized baselines. Equation \eqref{12} gives:

<div class="eq-scroll">

$$
\begin{aligned}
V_t^{\mathrm{raw}}-V_t^{\mathrm{joint}}
&\approx \kappa\,(b_{t,t}^\star)^2
\left(\E[\rho_{\le t}^2]
-\frac{\E[w\rho_{\le t}]^2}{\E[w^2]}\right)\ge 0,\\
V_t^{\mathrm{equiv}}-V_t^{\mathrm{joint}}
&\approx \kappa\,(b_{t,L}^\star)^2
\left(\E[w^2]
-\frac{\E[w\rho_{\le t}]^2}{\E[\rho_{\le t}^2]}\right)\ge 0.
\end{aligned}
$$

</div>

Summing over ${t<L}$ gives the full-gradient reductions. At ${t=L}$, or on-policy, the two controls coincide, so there is no gain from combining them.

**<span id="shared-position-derivation">One shared baseline at each position.</span>** For ${b_{t,s}=b_s}$, the control variate corresponding to $b_s$ is $H_s\triangleq\sum_{t\le s}\phi_{t,s}=\rho_{\le s}\sum_{t\le s}s_t$, giving $C_b=\sum_s b_sH_s$. To use the orthogonal time blocks from the [preceding subsection](#martingale-placements), consider the differences $\Delta H_s\triangleq H_s-H_{s-1}$, with ${H_0=0}$. Equation \eqref{13} gives $\Delta H_s=\sum_{t\le s}X_{t,s}$, the sum of control variates in time block $s$; hence these differences are mutually orthogonal. Writing $c_s\triangleq\sum_{u=s}^L b_u$ then gives $C_b=\sum_s c_s\Delta H_s$, so the exact shared solution is

<div class="eq-scroll">

$$
c_s^\star=
\frac{\E[(\Delta H_s)^\top\hat g_0]}{\E\|\Delta H_s\|^2},
\qquad b_s^\star=c_s^\star-c_{s+1}^\star,
\qquad c_{L+1}^\star=0,
$$

</div>

with ${c_s^\star=0}$ for a zero denominator. Using \eqref{7}, with $\delta_s\triangleq\rho_{\le s}-\rho_{\le s-1}$ and ${\rho_{\le0}=1}$, gives the ratio-only formula

<div class="eq-scroll">

$$
c_s^\star\approx
\frac{\E\left[wR\big(\rho_{\le s}+(s-1)\delta_s\big)\right]}
{\E\left[\rho_{\le s}^2+(s-1)\delta_s^2\right]}.
$$

</div>

Under this approximation, on-policy, the solution reduces to ${b_L^\star=\E[R]}$ with all earlier ${b_s^\star=0}$. These formulas use population orthogonality; replacing expectations with sample averages need not solve the empirical normal equations.

### <span id="finite-group-variance">The variance cost of fitted baselines</span>

For any fixed linear baseline structure, let $b^\star$ minimize the exact full-gradient variance within that structure. In the leave-one-out setup of [Section 6](#estimating-from-rollouts) with ${G\ge2}$, the baselines $\hat b^{(-i)}$ are fitted without rollout $y^{(i)}$; they respect this structure and have finite second moments, but may be approximate or biased. When applied to $y^{(i)}$, their correction differs from the optimal one by $e_i\triangleq C_{\hat b^{(-i)}-b^\star}(y^{(i)})$, giving $\hat g_G=\hat g_G^\star-\frac1G\sum_i e_i$.

Expanding the variance of $\hat g_G$ gives the two terms in \eqref{10} and the mixed term $-\frac{2}{G^2}\sum_{i,j}\E_{\mathcal D}[(\hat g^\star(y^{(i)})-\nabla_\theta J)^\top e_j]$. Despite the shared fitting data, every summand vanishes:

<div class="eq-scroll">

$$
\E_{\mathcal D}\left[
\big(\hat g^\star(y^{(i)})-\nabla_\theta J\big)^\top e_j
\right]=0
\qquad\text{for all }i,j.
$$

</div>

For ${i=j}$, conditioning on the fitting data fixes the coefficients of $e_i$. This zero-mean correction lies in the chosen structure, so \eqref{11}, restricted to variations within that structure, makes it orthogonal to $\hat g^\star(y^{(i)})$. For ${i\ne j}$, conditioning on all rollouts except $y^{(j)}$ fixes $\hat g^\star(y^{(i)})$ and the coefficients of $e_j$; $e_j$ then has conditional mean zero by [Lemma 2](#lemma-2). The reference average has variance $V^\star/G$, so the variance expansion gives \eqref{10}.

If the fits have a common finite mean $\bar b$, the remaining cost in \eqref{10} can be split into baseline bias and fitting noise. Define $e_{\mathrm{bias}}(y)\triangleq C_{\bar b-b^\star}(y)$ and $\eta_i\triangleq C_{\hat b^{(-i)}-\bar b}(y^{(i)})$. The bias–noise cross terms vanish by centering and conditioning, giving

<div class="eq-scroll">

$$
\mathcal V_G=
\frac{V^\star}{G}
+\frac1G\E\|e_{\mathrm{bias}}(y)\|^2
+\E_{\mathcal D}\left\|\frac1G\sum_{i=1}^G\eta_i\right\|^2.
\tag{14}
$$

</div>

The middle term is the variance cost of the mean coefficient error ${\bar b-b^\star}$, including finite-sample bias and any systematic error from [Section 5](#practical-baseline-choices)’s score approximation. The last term retains covariance between different $\eta_i$ because the fits share data; it need not vanish even when the baseline estimates are unbiased. A richer placement family can make the optimal correction more sensitive to the score approximation, increasing the bias cost. Estimating more coefficients from the same rollouts can increase fitting noise, especially when the control variates are nearly redundant.

## References

- <span id="ref-greensmith2004" data-cite="Greensmith et al., 2004"></span>Greensmith, E., Bartlett, P. L., and Baxter, J. (2004). [Variance reduction techniques for gradient estimates in reinforcement learning](https://www.jmlr.org/papers/v5/greensmith04a.html). *JMLR* 5.
- <span id="ref-hao2025" data-cite="Hao et al., 2025"></span>Hao, Y., Dong, L., Wu, X., Huang, S., Chi, Z., and Wei, F. (2025). [On-policy RL with optimal reward baseline](https://arxiv.org/abs/2505.23585). arXiv:2505.23585.
- <span id="ref-chen2026holder" data-cite="Chen et al., 2026"></span>Chen, Y., Liang, D., Chen, Y., et al. (2026). [Hölder policy optimisation](https://arxiv.org/abs/2605.12058). arXiv:2605.12058. Appendix I.3–I.4.
- <span id="ref-fang2026dynamo" data-cite="Fang et al., 2026"></span>Fang, Y., Lin, J., Fu, X., et al. (2026). [How to allocate, how to learn? Dynamic rollout allocation and advantage modulation for policy optimization](https://arxiv.org/abs/2602.19208). arXiv:2602.19208. Appendix C, Eq. (40).
- <span id="ref-huang2026" data-cite="Huang et al., 2026"></span>Huang, L., Zhang, Z., Hu, Q., Yang, S., and Han, S. (2026). [Stable asynchrony: variance-controlled off-policy RL for LLMs](https://arxiv.org/abs/2602.17616). arXiv:2602.17616.
- <span id="ref-jiang2016" data-cite="Jiang and Li, 2016"></span>Jiang, N. and Li, L. (2016). [Doubly robust off-policy value evaluation for reinforcement learning](https://arxiv.org/abs/1511.03722). *ICML*.
- <span id="ref-li2026" data-cite="Li et al., 2026b"></span>Li, Y., Xu, J., Li, Z., et al. (2026b). [The optimal token baseline: variance reduction for long-horizon LLM-RL](https://arxiv.org/abs/2602.07078). arXiv:2602.07078.
- <span id="ref-liu2025" data-cite="Liu et al., 2025"></span>Liu, J., Li, Y., Fu, Y., Wang, J., Liu, Q., and Shen, Y. (2025). [When speed kills stability: demystifying RL collapse from the training–inference mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda). Blog post.
- <span id="ref-precup2000" data-cite="Precup et al., 2000"></span>Precup, D., Sutton, R. S., and Singh, S. (2000). [Eligibility traces for off-policy policy evaluation](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf). *ICML*.
- <span id="ref-shao2024" data-cite="Shao et al., 2024"></span>Shao, Z., Wang, P., Zhu, Q., et al. (2024). [DeepSeekMath: pushing the limits of mathematical reasoning in open language models](https://arxiv.org/abs/2402.03300). arXiv:2402.03300. Introduces GRPO.
- <span id="ref-weaver2001" data-cite="Weaver and Tao, 2001"></span>Weaver, L. and Tao, N. (2001). [The optimal reward baseline for gradient-based reinforcement learning](https://arxiv.org/abs/1301.2315). *UAI*.
- <span id="ref-yao2025" data-cite="Yao et al., 2025"></span>Yao, F., Liu, L., Zhang, D., Dong, C., Shang, J., and Gao, J. (2025). [Your efficient RL framework secretly brings you off-policy RL training](https://fengyao.notion.site/off-policy-rl). Blog post.
- <span id="ref-zhangpn2026" data-cite="W. Zhang et al., 2026"></span>Zhang, W., Xie, Y., Wang, R., et al. (2026). [Reusing rollouts under policy lag: prefix-normalized policy optimization for LLM reinforcement learning](https://arxiv.org/abs/2608.01418). arXiv:2608.01418.
- <span id="ref-zhangct2026" data-cite="Y. Zhang et al., 2026"></span>Zhang, Y., Ye, C., Jin, S., et al. (2026). [Rethinking importance sampling in LLM policy optimization: a cumulative token perspective](https://arxiv.org/abs/2605.07331). arXiv:2605.07331.
- <span id="ref-zheng2025" data-cite="Zheng et al., 2025"></span>Zheng, C., Liu, S., Li, M., et al. (2025). [Group sequence policy optimization](https://arxiv.org/abs/2507.18071). arXiv:2507.18071.
- <span id="ref-oblr2025" data-cite="Z. Huang et al., 2025"></span>Huang, Z., Sheng, J., and Zheng, Z. (2025). [Variance-aware baselines and adaptive learning rates for reinforcement learning with verifiable rewards](https://arxiv.org/abs/2511.23310). arXiv:2511.23310.
- <span id="ref-rcc2026" data-cite="Li et al., 2026a"></span>Li, Y., Lan, T., and Qi, Z. (2026a). [When right meets wrong: bilateral context conditioning with reward-confidence correction for GRPO](https://arxiv.org/abs/2603.13134). arXiv:2603.13134.
- <span id="ref-cheng2020" data-cite="Cheng et al., 2019"></span>Cheng, C.-A., Yan, X., and Boots, B. (2019). [Trajectory-wise control variates for variance reduction in policy gradient methods](https://proceedings.mlr.press/v100/cheng20a.html). *CoRL*.
- <span id="ref-huangjiang2020" data-cite="J. Huang and Jiang, 2020"></span>Huang, J. and Jiang, N. (2020). [From importance sampling to doubly robust policy gradient](https://proceedings.mlr.press/v119/huang20b.html). *ICML*.
- <span id="ref-abc2026" data-cite="Pan et al., 2026b"></span>Pan, H.-R., Draye, F., and Schölkopf, B. (2026b). [ABC: advantage-based control variates for reinforcement learning with verifiable rewards](https://arxiv.org/abs/2609.36058). arXiv:2609.36058.
- <span id="ref-shrinkage2025" data-cite="Zeng et al., 2025"></span>Zeng, G., Zhou, Z., Arora, D., and Zanette, A. (2025). [Shrinking the variance: shrinkage baselines for reinforcement learning with verifiable rewards](https://arxiv.org/abs/2511.03710). arXiv:2511.03710.
- <span id="ref-bvblend2026" data-cite="Chang et al., 2026"></span>Chang, Y., Wu, Y., and Chang, Y. (2026). [BV-Blend: uncertainty-weighted historical baselines for stable critic-free RL with verifiable rewards](https://aclanthology.org/2026.findings-acl.84/). *Findings of ACL*.
- <span id="ref-tether2026" data-cite="Venkatraman et al., 2026"></span>Venkatraman, S., Dinot, M., and Aitchison, L. (2026). [Le Critique: privileged value functions for LLM reinforcement learning](https://arxiv.org/abs/2608.16739). arXiv:2608.16739.
- <span id="ref-evpo2026" data-cite="Pan et al., 2026a"></span>Pan, C., Liu, S., Lin, J., et al. (2026a). [EVPO: explained variance policy optimization for adaptive critic utilization in LLM post-training](https://arxiv.org/abs/2604.19485). arXiv:2604.19485.
- <span id="ref-poise2026" data-cite="Choi et al., 2026"></span>Choi, Y., Lim, J., Ahn, W., Oh, M., Shim, J., and Jo, Y. (2026). [Your language model is its own critic: reinforcement learning with value estimation from actor’s internal states](https://arxiv.org/abs/2605.07579). arXiv:2605.07579.
- <span id="ref-fipo2026" data-cite="Ma et al., 2026"></span>Ma, C., Yang, S., Huang, K., et al. (2026). [FIPO: eliciting deep reasoning with future-KL influenced policy optimization](https://arxiv.org/abs/2603.19835). arXiv:2603.19835.
- <span id="ref-brace2026" data-cite="Zhao et al., 2026"></span>Zhao, G., Xie, Z., Zheng, B., Lu, J., Gong, E., and Chen, Z. (2026). [BRACE: anchored Bellman-residual correction for stale critics in asynchronous RL](https://arxiv.org/abs/2609.09783). arXiv:2609.09783.
- <span id="ref-copc2026" data-cite="Hu et al., 2026"></span>Hu, Z., Zhou, Z., Zhang, X., et al. (2026). [COPC: coupled off-policy correction for asynchronous LLM reinforcement learning](https://arxiv.org/abs/2610.09597). arXiv:2610.09597.
- <span id="ref-han2026" data-cite="Han and Hu, 2026"></span>Han, S. and Hu, Z. (2026). [Baselines and score centering in policy gradient estimation](https://arxiv.org/abs/2610.22263). arXiv:2610.22263.
