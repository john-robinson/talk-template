class: middle, center, title-slide
count: false

# Video Frame Interpolation

Étude de cas

<br><br>

John Robinson<br>

---
## Contenu de cette presentation
- Comprendre le problème
- Revue d'articles
    - *Deep bayesian video frame interpolation*
    - *Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation*
    - *IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation*
    - *Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation*
    - *Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation*
- Que retenir de ces recherches ?
- Conclusion
- Prochaine étape

---
class: section
# Interpolation d'images
---

## Le problème

En se basant sur une série d'images $$\mathcal{I} = \\{I_{-k}, ... , I_0, I_1, ... I_k\\}$$

Construire un modèle $\mathcal{F}$ capable de générer une image intermédiaire.

$$I_t = \mathcal{F}(\mathcal{I}, t), \; \; 0 < t < 1$$

<!-- SCHEMA -->


$k$ paramétrise le modèle et le training set,

$$\mathcal{D} = \left\\{ \bigcup^{k}\_{l=1}I\_{i \pm l}, I\_i \right\\}^{N-k}\_{i=k}$$


- $k = 1$, triplets
- $k = 2$, quintuplets
- $k = 3$, septuplets 


Le **deep learning** nous permet d'approcher ce problème de **regression**.  

---
## Regression "pure"


Le modele $\mathcal{F}$ tente de capturer la relation directe entre l'output $I_t$ et les images adjacentes dans le dataset

<!-- 
SCHEMA -->

Cette formulation offre peu de flexibilité, $t = 0.5$

---
## Optic Flow

On considère ici une étape intermédiare, celle de l'**optic flow**, qui caractérise le mouvement apparent de la scène.

<!-- 
SCHEMA -->

La première étape consiste en l'estimation d'un certain nombre de d'optic flows, typiquement 2.

$$\phi = \\{F\_{i \rightarrow t}\\}\_i^{K}$$
où
$$F\_{i \rightarrow t} \approx g(\mathcal{I}, t)$$


Le modele $\mathcal{F}$ interpole donc en fonction des images et des flows.

$$I_t = \mathcal{F}(\mathcal{I}, \phi)$$

Cette approche permet une plus **grande flexibilité** quant a $t$. Cependant, approximer l'optic flow en ne se basant que sur les images reste **imprécis** (problème d'occlusion, etc...).



---
## Solution Actuelle, Modèle

Le modele actuel execute une régression "pure" et se base sur CAIN .footnote[Channel Attention Is All You Need for Video Frame Interpolation, 2020]

<p align = "center">
    <img src="/figures/CAIN/cain1.jpg"  width="90%">
    <img src="/figures/CAIN/cain2.jpg"  width="90%">
</p>


---
## Solution Actuelle, Challenges

Difficultés avec les **paternes répétitifs** et les situations **d'ambiguïté**.

<!-- Photos -->

Cette solution est donc clairement perfectible.
---
## Métriques et Evaluation

De nombreuse métriques telles que le **PSNR** et la **SSIM**
- **P**eak **S**ignal-to-**N**oise **R**atio (dB)
$$PSNR(I\_1, I\_2) = 10 \log\_{10} \left(\frac{MAX^2(I\_1)}{MSE(I\_1, I\_2)}\right)$$

Compare la qualité de $I\_2$ par rapport a $I_1$

- **S**tructural **SIM**ilarity (entre 0 et 1)
$$SSIM(x, y) = \frac{(2 \mu\_x\mu\_y + c\_1)(2 \sigma\_{xy} + c\_2)}{(\mu\_x^2 \mu\_y^2 + c\_1)(\sigma\_x^2 \sigma\_y^2 + c\_1)}$$

Compare la structure de l'image $x$ a celle de l'image $y$.

Ces métriques n'expliquent pas tout les aspects de la qualité d'une image, l'analyse **qualitative** reste donc de vigueur. 

---

class: section

# Revue d'articles

---

## Revue d'articles 
- *IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation*
- Deep bayesian video frame interpolation
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---


## IFRNet, le modèle

Cet article propose une approche **encodeur-decodeur** à plusieurs niveaux. Les images $I\_0$ et $I\_1$ sont encodée en une pyramide de features $\phi^{\\{1, ..., 4\\}}\_0$ et $\phi^{\\{1, ..., 4\\}}\_1$.

A chaque niveau $k$, le décodeur est chargé d'estimer les features interpolés $\hat{\phi\_t^{k}}$ et $F^k\_{t \rightarrow 0}$ et $F^k\_{t \rightarrow 1}$.

Ces flows permettent de **raffiner** les features encodés $\phi\_0^{k}$, $\phi\_1^{k}$ en $\tilde{\phi\_0^{k}}$, $\tilde{\phi\_1^{k}}$ afin de les décoder ensuite vers le prochain niveau $k-1$

Cette aproche utilise les flows de manière plus **holistique**. 
<p align = "center">
    <img src="/figures/IFRNet/ifrnet1.jpg"  width="95%">
</p>

---
## IFRNet, le modèle

L'encodeur construit une pyramide features.

$$\phi^{\\{1, ..., 4\\}}\_{0, 1} = \mathcal{E}(I\_0, I\_1)$$

Le premier décodeur $\mathcal{D}^4$ produit les premiers flows et features interpolés.

$$F^3\_{t \rightarrow 0}, F^3\_{t \rightarrow 1}, \hat{\phi}^3\_{t} = \mathcal{D}^4(\phi\_0^4, \phi\_1^4, T)$$

Les décodeurs intermédiares $\mathcal{D}^k, k = 2, 3$ raffinent les flows et les features.

$$F^{k-1}\_{t \rightarrow 0}, F^{k-1}\_{t \rightarrow 1}, \hat{\phi}^{k-1}\_{t} = \mathcal{D}^k(F^{k}\_{t \rightarrow 0}, F^{k}\_{t \rightarrow 1}, \hat{\phi}^k\_t, \tilde{\phi}\_0^k, \tilde{\phi}\_1^k)$$

Le dernier décodeur $\mathcal{D}^1$ calcule les flows ainsi que $M$ et $R$

$$F\_{t \rightarrow 0}, F\_{t \rightarrow 1}, M, R = \mathcal{D}^1(F^{1}\_{t \rightarrow 0}, F^{1}\_{t \rightarrow 1}, \hat{\phi}^1\_t, \tilde{\phi}\_0^1, \tilde{\phi}\_1^1)$$

---
## IFRNet, le modèle

L'output $\hat{I}\_t$ est ensuite calculée via un block **warp-merge-add**

$$\hat{I}\_t = M \odot \tilde{I}\_0 + (1-M) \odot \tilde{I}\_1 + R$$

où $\odot$ est le produit d'élément à élément et

$$\tilde{I}\_i = \text{BackwardWarping}(I\_i, F\_{t \rightarrow i})$$

$M$ est un masque ajustant la fusion de $\tilde{I}\_0$ et $\tilde{I}\_1$. $R$ est une image résiduelle compensant les erreurs du warping.

---
## IFRNet, entrainement



Le modèle est entrainé pour optimiser 3 **loss**
- $\mathcal{L}\_{r}$ pénalisant la **reconstruction** de l'image.
- $\mathcal{L}\_{d}$ pénalisant **l'estimation des flows**.
- $\mathcal{L}\_{g}$ pénalisant les **changements de structure** à l'échelle des features.

on considère alors l'optimisation jointe

$$\mathcal{L} = \mathcal{L}\_{r} + \lambda \mathcal{L}\_{d} + \eta \mathcal{L}\_{g}$$

avec $\lambda = \eta = 0.01$.

---
## IFRNet, entrainement
$\mathcal{L}\_{r}$ est calculée pour être la somme des **Charbonnier** et **Census** loss entre $\hat{I}\_t$ et $I\_{GT}$

$$\mathcal{L}\_{r} = \rho(\hat{I}\_t - I\_t^{GT}) + \mathcal{L}\_{cen}(\hat{I}\_t, I\_t^{GT})$$

- $\rho(x) = (x^2 + \epsilon^2)^\alpha$ avec $\alpha = 0.5$ et $\epsilon = 10^{-3}$ est la loss de **Charbonnier** substitue la loss $L1$ en étant plus flexible.
- $\mathcal{L}\_{cen}$ calcule la distance de **Hamming** entre des patches de 7x7 transformés suivant la transformée de **Census**. Conserve les propriétés géométriques de l'image.


<!-- <p align = "center">
    <img src="/figures/IFRNet/ifrnet3.jpg"  width="60%">
</p> -->

---
## IFRNet, entrainement

IFRNet distille la connaissance d'un réseau de neurone **externe** pré-entrainé pour **estimer les flows**. Ses prédiction $F^p\_{t \rightarrow 0}, F^p\_{t \rightarrow 1}$ servent de pseudo label pour les décodeurs.

Pour ajuster la robustesse de cette distillation, les masques $P\_0$ et $P\_1$ sont calculés

$$P\_l = exp(-\beta ||F\_{t \rightarrow l} - F^p\_{t \rightarrow l}||)$$

Desquels $p \in [0, 1)$ est déterminé, $p$ ajuste les paramètres de $\rho$, $\alpha = p/2$ et $\alpha = 10^{-(10p - 1)/3}$ 

La loss est ensuite calculée comme

$$\mathcal{L}\_{d} = \sum\_{k=1}^3\sum\_{l=0}^1 \rho(F^{k \uparrow 2^k}\_{t \rightarrow l} - F^{p}\_{t \rightarrow l})$$

Cette formulation permet au modèle **d'apprendre** du réseau tiers tout en ajustant un degré de **confiance** en ce réseau grâce à $\beta$.
---
class: middle

<p align = "center">
    <img src="/figures/IFRNet/ifrnet3.jpg"  width="60%">
</p>

---

## IFRNet, entrainement  

IFRNet supervise aussi le calcul des features intermédiares $\hat{\phi}\_t^k$ en encodant les features de $I\_t^{GT}$ avec l'encodeur $\mathcal{E}$.

La similarité entre ces features est calculée avec la census loss sur des patches 3x3

$$\mathcal{L}\_{g} = \sum^{3}\_{k = 1} \mathcal{L}\_{cen}(\hat{\phi}\_t^k, \phi\_t^k)$$

avec 
$$\phi\_t^{\\{1, ..., 3\\}} = \mathcal{E}(I^{GT}\_t)$$



---
## IFRNet

<br>

<p align = "center">
    <img src="/figures/IFRNet/ifrnet2.jpg"  width="105%">
</p>



---
## IFRNet, performances

Résultats





## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- *Deep bayesian video frame interpolation*
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---

## Deep Bayesian VFI
---

## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- Deep bayesian video frame interpolation
- *Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation*
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---


## Exploring Motion Ambiguity and Alignment
---

## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- Deep bayesian video frame interpolation
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- *Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation*
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---

## Uncertainty Guided Spatial Pruning
---

## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- Deep bayesian video frame interpolation
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- *Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation*
---

## Clearer Frames, Anytime
---

# References
- Liste des articles
    - *Yu, Zhiyang, & al. "Deep bayesian video frame interpolation." Oct 2022.*
    - *Choi, Kim, & al. "Channel Attention Is All You Need for Video Frame Interpolation" 2020.*
    - *Zhou, Li, & al. "Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation" Mar 2022*
    - *Kong, Jiang, & al. "IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation" May 2022*
    - *Cheng, Jiang, & al. "Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation" Oct 2023*
    - *Zhong, Krishnan, & al. "Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation" Nov 2023*
---
- Autres références
---

