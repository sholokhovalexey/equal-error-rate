**TL;DR: this note illustrates a rearly mentioned property of the equal error rate (ERR) metric in the context of Bayes decision theory and shows its geometric interpretation.**

For the interested reader refer to [^1], [^2].

## Definition
The **Equal Error Rate (EER)** is a performance metric commonly used to evaluate binary classifiers. 
The EER is defined as the point where the **False Acceptance Rate (FAR)** and **False Rejection Rate (FRR)** are equal, providing a single scalar value that balances the two types of errors.

The EER can be derived from the **Receiver Operating Characteristic (ROC)** or from the **Detection Error Tradeoff (DET)** curves. It is defined as the point on the curve for which 
(FAR, FRR) = (EER, EER).

<center><img src="roc_eer.png" width="500"></center>

Unlike accuracy, EER is less sensitive to class imbalance because it focuses on the trade-off between FAR and FRR rather than absolute counts.

### Background

Let's start from defining a binary classification problem in the framework of Bayes decision theory:
- **Class prior:**
  $P(Y=1) = \pi$,
  $P(Y=0) = 1 - \pi$.
- **Class-conditional densities:**
  $p(x|Y=1)$,
  $p(x|Y=0)$.
- **Loss function:** 0-1 loss.

The **Bayes-optimal decision rule** classifies $`x`$ as class 1 if the **likelihood ratio** exceeds the **Bayes threshold**:

$`\frac{p(x|Y=1)}{p(x|Y=0)} \geq t_\mathrm{Bayes}(\pi) = \frac{1-\pi}{\pi}`$

The total **probability of error** for the given threshold $`t`$ and prior $`\pi`$ is:
```math
P_\mathrm{error}(\pi, t) = \pi \cdot P_\mathrm{miss}(t) + (1 - \pi) \cdot P_\mathrm{fa}(t),
```
where

**FAR (False Positive Rate)**: 
$`P_\mathrm{fa}(t) = \int_{t}^{\infty}p(x|Y=0)dx`$

**FRR (False Rejection Rate)**: 
$`P_\mathrm{miss}(t) = \int_{-\infty}^{t}p(x|Y=1)dx`$

The **Bayes error rate (BER)** is obtained at the **optimal threshold** $`t_\mathrm{Bayes}`$ that minimizes $`P_\mathrm{error}`$:

```math
t_\mathrm{Bayes}(\pi) = \arg \min_{t} P_\mathrm{error}(\pi, t)
```
```math
\mathrm{BER}(\pi) = P_\mathrm{error}(\pi, t_\mathrm{Bayes}) = \pi \cdot P_\mathrm{miss}(t_\mathrm{Bayes}(\pi))+(1-\pi) \cdot P_\mathrm{fa}(t_\mathrm{Bayes}(\pi))
```

It is the minimum achievable classification error for a given $`\pi`$.


## 👉 EER is an upper bound of the Bayes error rate 👈

If the prior $`\pi`$ is **unknown**, we cannot compute $`t_\mathrm{Bayes}`$. Instead, we may seek the **worst-case minimal error**:
```math
\max_{\pi \in [0,1]} \min_{t} P_\mathrm{error}(\pi, t)
```

The inner minimization $`\min_t P_\mathrm{error}(\pi, t)`$ yields the **Bayes error rate (BER)** for a given $`\pi`$.
The outer maximization finds the prior $`\pi`$ that makes BER as large as possible.

### Geometric interpretation

Let's express BER as a dot product

```math
P_\mathrm{error}(\pi, t) = \pi \cdot P_\mathrm{miss}(t) + (1 - \pi) \cdot P_\mathrm{fa}(t) = [\pi, 1 - \pi] \cdot \begin{bmatrix} P_\mathrm{miss}(t) \\ P_\mathrm{fa}(t) \end{bmatrix}
```
To find the wors-case error $`\max_{\pi \in [0,1]} \min_{t} P_\mathrm{error}(\pi, t)`$, let's first note that the DET (or ROC) curve forms a convex set and serves as its boundary (see [^3], [^4] for details). 
Since a DET curve is convex, the minimum dot product will be achieved at a point $`(P_\mathrm{miss}(t), P_\mathrm{fa}(t))`$ where the hyperplane (line) orthogonal to $`[\pi, 1 - \pi]`$ supports the curve. 

<center><img src="det_curve.gif" width="1200"></center>


### Derivation using Sion's theorem

By **Sion’s minimax theorem**, if:
- $`P_\mathrm{error}(\pi, t)`$ is quasi-convex in $`t`$, 
- quasi-concave (or linear) in $`\pi`$,
  
then it is possible to **swap** maximization and minimization:
```math
\max_{\pi \in [0,1]} \min_{t} P_\mathrm{error}(\pi, t) = \min_{t} \max_{\pi \in [0,1]} P_\mathrm{error}(\pi, t)
```
The right-hand side $`\min_{t} \max_{\pi \in [0,1]} P_\mathrm{error}(\pi, t)`$ asks: for a **fixed threshold** $`t`$, what is the **worst** $`\pi`$?
 
Since $`P_\mathrm{error}(\pi, t) = \pi \cdot P_\mathrm{miss}(t) + (1-\pi) \cdot P_\mathrm{fa}(t)`$, it is **linear in $`\pi`$**, the maximum occurs at $`\pi = 0`$ or $`\pi = 1`$, depending on whether $`P_\mathrm{fa}(t)>P_\mathrm{miss}(t)`$:
```math
\max_{\pi \in [0,1]} P_\mathrm{error}(\pi, t) = \max(P_\mathrm{fa}(t), P_\mathrm{miss}(t))
```
Thus, the **minimax solution** is the threshold $`t_∗`$ where the graphs of $`P_\mathrm{fa}(t)`$ and $`P_\mathrm{miss}(t)`$ intersect:
```math
P_\mathrm{fa}(t_∗) = P_\mathrm{miss}(t_∗)
```
which is precisely the $`\mathrm{EER}`$.

Given that the **maximin = minimax**, we have:
```math
\underbrace{\max_{\pi}\mathrm{BER}(\pi)}_\text{worst-case BER} = \underbrace{\mathrm{EER}}_\text{minimax error}
```
Hence, **EER is the worst-case Bayes error when the prior $`\pi`$ is unknown**.

<center><img src="fpr_fnr.png" width="500"></center>


<details open>
<summary>Validity of the theorem's application</summary>
<br>
BER can be written as follows: $\mathrm{BER}(\pi) = \min_{t} \left( \pi \cdot P_\mathrm{miss}(t)+(1-\pi) \cdot P_\mathrm{fa}(t) \right)$. The pointwise minimum of linear functions is quasi-convex (since linear functions are convex and their minimum preserves quasi-convexity). This ensures that Sion’s theorem applies, allowing us to swap the min and max.
</details>


### Alternative derivation by differentiating the Bayes error rate

We derive the **Equal Error Rate (EER)** as the **worst-case Bayes error** by analyzing how the **Bayes-optimal decision threshold** affects the **False Acceptance Rate (FAR)** and **False Rejection Rate (FRR)** as the **class prior probability** $`\pi`$ varies. 

We seek the prior $`\pi`$ that **maximizes BER**, i.e., the worst-case scenario where the classifier performs the poorest.

Differentiating BER with respect to $`\pi`$:
```math
\frac{d}{d \pi} \mathrm{BER}(\pi) = P_\mathrm{miss}(\pi) - P_\mathrm{fa}(\pi) + \pi \cdot \frac{d}{d \pi} P_\mathrm{miss}(\pi) + (1 - \pi) \cdot \frac{d}{d \pi} P_\mathrm{fa}(\pi)
```
Using the chain rule on FAR and FRR and since $`t_\mathrm{Bayes}(\pi) = \frac{1 - \pi}{\pi}`$, we compute:

$`\frac{d}{d \pi} P_\mathrm{fa}(\pi) = \frac{d P_\mathrm{fa}}{dt} \Bigr|_{t=tt} \cdot \frac{d t_\mathrm{Bayes}}{d\pi}`$

$`\frac{d}{d \pi}P_\mathrm{miss}(\pi) = \frac{d P_\mathrm{miss}}{dt} \cdot \frac{d t_\mathrm{Bayes}}{d\pi}`$

Derivatives of $`P_\mathrm{fa}`$ and $`P_\mathrm{miss}`$:

Since $`P_\mathrm{fa}(t) = \int_{t}^{\infty} p(x|Y=0)dx`$, we have:

$`\frac{dP_\mathrm{fa}}{dt} = -p(t|Y=0)`$

Since $`P_\mathrm{miss}(t) = \int_{-\infty}^{t} p(x|Y=1)dx`$, we have:

$`\frac{dP_\mathrm{miss}}{dt} = p(t|Y=1)`$

Derivative of the Bayes threshold $`t_\mathrm{Bayes}(\pi)`$:

$`t_\mathrm{Bayes}(\pi) = \frac{1 - \pi}{\pi} \Rightarrow \frac{d t_\mathrm{Bayes}}{d\pi} = - \left( \frac{1}{\pi^2} \right)`$

Substituting Back into $`\frac{d}{d \pi} \mathrm{BER}(\pi)`$:

```math
\begin{aligned}
\frac{d}{d \pi} \mathrm{BER}(\pi) = P_\mathrm{miss}(\pi) - P_\mathrm{fa}(\pi) + \pi \cdot \left( p(t_\mathrm{Bayes}|Y=1) \cdot \left(-\frac{1}{\pi^2} \right) \right) + (1 - \pi) \cdot \left( -p(t_\mathrm{Bayes}|Y=0) \cdot \left(-\frac{1}{\pi^2} \right) \right) \\
= P_\mathrm{miss}(\pi)-P_\mathrm{fa}(\pi) - \frac{p(t_\mathrm{Bayes}|Y=1)}{\pi} + (1-\pi) \frac{p(t_\mathrm{Bayes}|Y=0)}{\pi^2} \\
\end{aligned}
```

At the **Bayes threshold** $`t_\mathrm{Bayes}(\pi)`$ we have:

$`p(t_\mathrm{Bayes}|Y=1) = \left( \frac{1 - \pi}{\pi} \right) p(t_\mathrm{Bayes}|Y=0)`$

Substituting this into the derivative we get:
```math
\frac{d}{d \pi }\mathrm{BER}(\pi) = P_\mathrm{miss}(\pi) - P_\mathrm{fa}(\pi) \\
```

The **maximum BER** occurs where $`\frac{d}{d \pi} \mathrm{BER}(\pi) = 0`$:
```math
P_\mathrm{miss}(\pi)-P_\mathrm{fa}(\pi) = 0  \Rightarrow  P_\mathrm{miss}(\pi) = P_\mathrm{fa}(\pi)
```
This is precisely the **Equal Error Rate (EER)** condition.


<details open>
<summary>Second derivative check (concavity of BER)</summary>
<br>
To confirm this is a **maximum**, we check the second derivative:
    
$`\frac{d^2}{d\pi^2} \mathrm{BER}(\pi) = \frac{d}{d \pi}(P_\mathrm{miss}(\pi) - P_\mathrm{fa}(\pi))`$

From earlier:

$`\frac{d}{d \pi}P_\mathrm{miss}(\pi) = - \frac{p(t_\mathrm{Bayes}|Y=1)}{\pi^2}`$

$`\frac{d}{d \pi}P_\mathrm{fa}(\pi) = \frac{p(t_\mathrm{Bayes}|Y=0)}{\pi^2}`$

Thus:

$`\frac{d^2}{d\pi^2} \mathrm{BER}(\pi) = - \frac{p(t_\mathrm{Bayes}|Y=1)}{\pi^2} - \frac{p(t_\mathrm{Bayes}|Y=0)}{\pi^2} < 0`$

This shows that $`\mathrm{BER}(\pi)`$ is **concave** in $`\pi`$, so the critical point is indeed a **maximum**.
</details>



#### Geometric interpretation

##### General result

<details open>
<summary>Minimizing a dot product over a convex set</summary>
<br>

Let have a convext set $`C`$ and a vector $`P`$ whose endpoint is on a line segment between points $`A`$ and $`B`$. For each $`P`$ we can compute the function $`f(P)`$ that is a dot product $`\langle P, Y \rangle`$, minimized over all points $`Y`$ belonging to the set $`C`$. Find a point $`P`$ that maximizes $`f(P)`$.

Let's start from expressing $`P`$ as:
$`P(\pi) = A + \pi \cdot (B - A)`$, 
where $`\pi`$ is a number between $`0`$ and $`1`$. 

Then, the problem can be formulated as follows:

$`\max_{\pi \in [0,1]}\min_{Y \in C} \langle P(t), Y \rangle`$

Since $`C`$ is convex, the minimum dot product over $`Y \in C`$ will be achieved at a point where the hyperplane orthogonal to $`P`$ supports the set $`C`$. 

The objective function for the outer optimization can be re-writte as:

$`f(\pi) = f(P(\pi)) = \min_{Y \in C} \langle A + \pi \cdot (B - A), Y \rangle = \min_{Y \in C} \langle A, Y \rangle + \pi \cdot \langle B - A, Y \rangle`$

For each fixed $`Y`$, the expression $`\langle A, Y \rangle + \pi \cdot \langle B - A, Y \rangle`$ is a straight line in $`\pi`$. The minimum of a family of straight lines is a **concave** function in $`\pi`$.
The maximum of $`f(\pi)`$ must occur at a point where the derivative with respect to $`\pi`$ is zero (if such a point exists in $`[0, 1]`$). So, the maximum occurs where $`\langle B - A, Y \rangle = 0`$. 

In 2D case the condition $`\langle Y, B - A \rangle = 0`$ means that the vector $`Y`$ is perpendicular to the line $`AB`$. 

</details>


##### Relation to EER 

Let's express BER as a dot product
```math
P_\mathrm{error}(\pi, t) = \pi \cdot P_\mathrm{miss}(t) + (1 - \pi) \cdot P_\mathrm{fa}(t) = [P_\mathrm{miss}(t), P_\mathrm{fa}(t)] \cdot \begin{bmatrix} \pi \\ 1 - \pi \end{bmatrix}
```

To find the wors-case error
```math
\max_{\pi \in [0,1]} \min_{t} P_\mathrm{error}(\pi, t)
```

Let's recall that a DET (or ROC) curve is convex (concave). Hence, the inner minimization over its epigraph (convex set) can be replaced by minimization over a scalar $`t`$. For a fixed $`t`$, we seek the point $`(P_\mathrm{fa}(t), P_\mathrm{miss}(t)`$ on the DET curve that minimizes this dot product. The outer maximization can be seen as finding a point $`P = (\pi, 1 - \pi)`$ on a line segment between the points $`(0, 1)`$ and $`(1, 0)`$. This formulation matches to the general result obtained before and allows to conclude that the optimal point is on the intersection of the DET curve with the line along the direction $`(1, 1)`$, which is exactly the EER point $`(\mathrm{EER}, \mathrm{EER})`$.



## References

[^1]: **Brummer, N.** (2010). *Measuring, refining and calibrating speaker and language information extracted from speech.*
[^2]: **Brummer, N., Ferrer, L., Swart, A.** (2021). *Out of a hundred trials, how many errors does your speaker verifier make?*
[^3]: **Cali, C., Longobardi, M.** (2015). *Some mathematical properties of the ROC curve and their applications.*
[^4]: **Gneiting, T., Vogel, P.** (2022). *Receiver operating characteristic (ROC) curves: equivalences, beta model, and minimum distance estimation.*
