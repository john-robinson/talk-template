class: middle, center, title-slide
count: false

# Video Frame Interpolation


<br><br>

John Robinson<br>

---
## Contenu de cette presentation


- Comprendre le problème
- Revue d'articles
    - *Deep Bayesian Video Frame Interpolation*
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

Cette solution est donc perfectible.
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
- Deep Bayesian Video Frame Interpolation
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---


## IFRNet (Mai 2022)

Cet article propose une approche **encodeur-decodeur** à plusieurs niveaux. Les images $I\_0$ et $I\_1$ sont encodée en une pyramide de features $\phi^{\\{1, ..., 4\\}}\_0$ et $\phi^{\\{1, ..., 4\\}}\_1$.

A chaque niveau $k$, le décodeur est chargé d'estimer les features interpolés $\hat{\phi\_t^{k}}$ et $F^k\_{t \rightarrow 0}$ et $F^k\_{t \rightarrow 1}$.

Ces flows permettent de **raffiner** les features encodés $\phi\_0^{k}$, $\phi\_1^{k}$ en $\tilde{\phi\_0^{k}}$, $\tilde{\phi\_1^{k}}$ afin de les décoder ensuite vers le prochain niveau $k-1$

Cette aproche utilise les flows de manière plus **holistique**. 
<p align = "center">
    <img src="/figures/IFRNet/ifrnet1.jpg"  width="95%">
</p>

---
## IFRNet, la méthode

L'encodeur construit une pyramide features.

$$\phi^{\\{1, ..., 4\\}}\_{0, 1} = \mathcal{E}(I\_0, I\_1)$$

Le premier décodeur $\mathcal{D}^4$ produit les premiers flows et features interpolés.

$$F^3\_{t \rightarrow 0}, F^3\_{t \rightarrow 1}, \hat{\phi}^3\_{t} = \mathcal{D}^4(\phi\_0^4, \phi\_1^4, T)$$

Les décodeurs intermédiares $\mathcal{D}^k, k = 2, 3$ raffinent les flows et les features.

$$F^{k-1}\_{t \rightarrow 0}, F^{k-1}\_{t \rightarrow 1}, \hat{\phi}^{k-1}\_{t} = \mathcal{D}^k(F^{k}\_{t \rightarrow 0}, F^{k}\_{t \rightarrow 1}, \hat{\phi}^k\_t, \tilde{\phi}\_0^k, \tilde{\phi}\_1^k)$$

Le dernier décodeur $\mathcal{D}^1$ calcule les flows ainsi que $M$ et $R$

$$F\_{t \rightarrow 0}, F\_{t \rightarrow 1}, M, R = \mathcal{D}^1(F^{1}\_{t \rightarrow 0}, F^{1}\_{t \rightarrow 1}, \hat{\phi}^1\_t, \tilde{\phi}\_0^1, \tilde{\phi}\_1^1)$$

---
## IFRNet, la méthode

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
## IFRNet, récapitulatif

<br>

<p align = "center">
    <img src="/figures/IFRNet/ifrnet2.jpg"  width="100%">
</p>

---
## IFRNet, récapitulatif

Quantitativement, l'article présente 3 modèles, **IFRNet small**, **IFRNet** et **IFRNet large**, tous sont plus performants que CAIN.

<p align = "center">
    <img src="/figures/IFRNet/ifrnet4.jpg"  width="100%">
</p>

---

## IFRNet, récapitulatif

Qualitativement, IFRNet montre de bons résultats

<p align = "center">
    <img src="/figures/IFRNet/ifrnet5.jpg"  width="100%">
</p>



---

## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- *Deep Bayesian Video Frame Interpolation*
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---

## DBVFI (Oct 2022)

L'approche suggérée est de considérer l'interpolation comme un problème de **maximisation**.
L'image interpolée $I\_t^\*$ maximise une **distribution d'images** conditionnée par les données

$$I\_t^\* = \underset{I\_t}{\text{argmax}} P(I\_t | I\_0, I\_1, F\_{0 \rightarrow t}, F\_{1 \rightarrow t})$$

Ce modèle ensuite relaxé puis consolidé en considérant les possibles erreurs d'estimation de $F\_{0 \rightarrow t}$ et $F\_{1 \rightarrow t}$
L'interpolation est alors **itérative** s'apparant à une descente de **gradient** tirant part des réseaux de neurones.

---

## DBVFI, la méthode

Le modèle est **relaxé**, les paires d'images et de flows $I\_0, F\_{0 \rightarrow t}$ et $I\_1, F\_{1 \rightarrow t}$ sont indépendantes.
$$P(I\_t | I\_0, I\_1, F\_{0 \rightarrow t}, F\_{1 \rightarrow t}) = \prod\_{i \in \\{0, 1\\}} P(I\_t | I\_i, F\_{i \rightarrow t})$$

Comme l'estimation des flows est basée sur un **framerate bas**, on présume donc une erreur $\Delta F\_{i \rightarrow t}$ comme étant une variable latente du modèle. En intégrant pour toute les erreurs possible, 

$$P(I\_t | I\_i, F\_{i \rightarrow t}) = \int\_{\Delta F\_{i \rightarrow t}} P(I\_t | I\_i, F\_{i \rightarrow t}, \Delta F\_{i \rightarrow t}) P(\Delta F\_{i \rightarrow t} | I\_i, F\_{i \rightarrow t}) d\Delta F\_{i \rightarrow t}$$

Cette intégrale est **incalculable**, on accepte alors une approximation avec $$\Delta \hat{F}\_{i \rightarrow t} = \underset{\Delta F\_{i \rightarrow t}}{\text{argmax}} P(\Delta F\_{i \rightarrow t} | I\_i, F\_{i \rightarrow t})$$

$$P(I\_t | I\_i, F\_{i \rightarrow t}) \approx P(I\_t | I\_i, F\_{i \rightarrow t}, \Delta \hat{F}\_{i \rightarrow t}) P(\Delta F\_{i \rightarrow t} | I\_i, F\_{i \rightarrow t})$$

---
## DBVFI, la méthode

Avec ces changements, prendre le **logarithme négatif** donne l'expression d'une loss 
$$\mathcal{L} = -  \sum\_{i \in \\{0, 1\\}} \\left(\log P(I\_t | I\_i, F\_{i \rightarrow t} \\right)) + \log P(\Delta F\_{i \rightarrow t} | I\_i, F\_{i \rightarrow t})$$

Permettant une descente de gradient sur les images et les erreurs

$$I\_t^{(k+1)} = I\_t^{(k)} - \lambda\_I \frac{\partial \mathcal{L}}{\partial I\_t}$$
$$\Delta \hat{F}\_{i \rightarrow t}^{(k+1)} = \Delta \hat{F}\_{i \rightarrow t}^{(k)} - \lambda\_F \frac{\partial \mathcal{L}}{\partial \Delta \hat{F}\_{i \rightarrow t}}$$

Les modules Flow/Image Gradient estiment ces gradients.

-  $\frac{\partial \mathcal{L}}{\partial I\_t}$ est formulé **explicitement** se basant sur le warping de $I\_t^{(k)}$ par le flow $F\_{i \rightarrow t} + \Delta \hat{F}\_{i \rightarrow t}^{(k)}$
-  $\frac{\partial \mathcal{L}}{\partial \Delta \hat{F}\_{i \rightarrow t}}$ est formulé **implicitement** avec un réseau de neurones.

---
## DBVFI, la méthode

Afin de réduire le nombre d'updates nécéssaire, la méthode proposée approche l'optimisation en estimant l'update à apporter avec **un réseau de neurones**

$$I\_t^{(k+1)} = I\_t^{(k)} + \mathcal{G}\_I \\left( \\left\\{ \frac{\partial \mathcal{L}}{\partial I\_t}\\right\\}, I\_t^{(k)}   \\{F\_{i \rightarrow t}\\}, \\{\Delta \hat{F}\_{i \rightarrow t}\\} \\right)$$

$\\{\cdot\\}$ indique l'ensemble des évaluations pour chaque image $I\_0$ et $I\_1$.

$$\Delta \hat{F}\_{i \rightarrow t}^{(k + 1)} = \Delta \hat{F}\_{i \rightarrow t}^{(k)} + \mathcal{G}\_F \\left(  \frac{\partial \mathcal{L}}{\partial \Delta \hat{F}\_{i \rightarrow t}}\\, I\_i , \Delta \hat{F}\_{i \rightarrow t}^{(k)}, F\_{i \rightarrow t} \\right)$$

Considérant que $\mathcal{G}\_I$ et $\mathcal{G}\_F$ partagent certains inputs, ces deux reseaux de neurones sont implémentés avec un CNN **commun** 


---

## DBVFI, l'entrainement
Toute chose confondue, exécuter une étape d'optimisation implique l'utilisation de 2 réseaux de neurones. Entrainer ce modèle consiste à réaliser $K$ étapes d'optimisation, 
$$I\_t^{(1)}, ..., I\_t^{(K)}$$
Et d'optimiser les paramètres de ses réseaux en considérant la reconstruction de l'image
$$\mathcal{L}\_r = \sum\_{k=1}^K \alpha\_k ||I\_t^{GT} - I\_t^{(k)}||_1$$

Les $\alpha\_k$ sont déterminés empiricallement et **augmentent** avec les itérations


---

## DBVFI, récapitulatif

Le modèle fonctionne celon ce pipeline.


<p align = "center">
    <img src="/figures/DBVFI/dbvfi1.jpg"  width="100%">
</p>



---
## DBVFI, récapitulatif

Le réseau de neurones englobant $\mathcal{G}\_I$ et $\mathcal{G}\_F$ a cette structure.
<br>
<br>
<p align = "center">
    <img src="/figures/DBVFI/dbvfi2.jpg"  width="100%">
</p>



---

## DBVFI, récapitulatif

Quantitativement, ce modèle performe mieux que de nombreux autres, y compris CAIN.
<br>
<br>
<p align = "center">
    <img src="/figures/DBVFI/dbvfi3.jpg"  width="100%">
</p>



---

## DBVFI, récapitulatif


Qualitativement, les résultats sont satisfaisants, notamment au niveaux des structures **répétitives**

<p align = "center">
    <img src="/figures/DBVFI/dbvfi4.jpg"  width="90%">
</p>



---





## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- Deep Bayesian Video Frame Interpolation
- *Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation*
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---


## Exploring Motion Ambiguity and Alignment (Mars 2022)

L'approche proposée se base également sur une décomposition des deux images en **pyramide de features**

$$\phi^{\\{0, 1, 2\\}}\_0 \text{ et } \phi^{\\{0, 1, 2\\}}\_1 $$

Ces features sont ensuite concaténés et alignés à travers leurs différents niveaux. **Cross Scale Pyramid Alignment**.

<p align = "center">
    <img src="/figures/cspa/cspa1.jpg"  width="100%">
</p>

 

<!-- SCHEMA -->


---
## Exploring Motion Ambiguity and Alignment, la méthode

L'alignement de la pyramide se fait d'un niveau au niveau inférieur.

Au sommet, $\phi^{2}\_0$ et $\phi^{2}\_1$ sont alignés puis fusionnés

$$\phi^2\_{0 \rightarrow 0.5} = \text{Align}(\phi^{2}\_0, \phi^{2}\_1)$$


$$\tilde{\phi}^1\_{0 \rightarrow 0.5} = \text{Fuse}(\phi^{2 \uparrow 2}\_{0 \rightarrow 0.5}, \phi^{1}\_0)$$


La prochaine étape aligne ce résultat et fusionne avec **tout** les features antérieurs.

$$\phi^1\_{0 \rightarrow 0.5} = \text{Align}(\tilde{\phi}^{1}\_0, \phi^{1}\_1)$$

$$\tilde{\phi}^0\_{0 \rightarrow 0.5} = \text{Fuse}(\phi^{2 \uparrow 4}\_{0 \rightarrow 0.5}, \phi^{1 \uparrow 2}\_{0 \rightarrow 0.5}, \phi^{0}\_0)$$

Le feature final $\phi^0\_{0 \rightarrow 0.5}$

$$\phi^0\_{0 \rightarrow 0.5} = \text{Align}(\tilde{\phi}^0\_{0 \rightarrow 0.5}, \phi^0\_{1} )$$

Ce procédé est répété pour le calcul de ${\phi}^0\_{1 \rightarrow 0.5}$

---

## Exploring Motion Ambiguity and Alignment, la méthode

Le module **CSF** implémente Fuse commune une concaténation suivi d'une convolution.

Le module **AB** implémente Align en calculant un masque d'**offset** $O^l\_{k \rightarrow 0.5}$ et de **weight** $W^l\_{k \rightarrow 0.5}$.

Le feature aligné est calculé par une convolution

$$\phi^{l}\_{k \rightarrow 0.5}(x) = \sum\_i \tilde{\phi}^l\_{k \rightarrow 0.5} (x + O^l\_{k \rightarrow 0.5, i}(x)) * W^l\_{k \rightarrow 0.5, i}(x)$$

$i$ indique l'$i$ème élément du champ réceptif de cette convolution.

Le feature interpolé est calculé grâce à un masque d'attention.

$$\phi\_{0.5} = M * \phi^0\_{0 \rightarrow 0.5} + (1 - M) \phi^0\_{1 \rightarrow 0.5}$$

$$M = \sigma(\phi^0\_{0 \rightarrow 0.5} * \phi^0\_{1 \rightarrow 0.5})$$

$\hat{I}\_{0.5}$ est ensuite reconstruite à partir de $\phi\_{0.5}$ par un module composé de blocs résiduels et d'une convolution.

---
## Exploring Motion Ambiguity and Alignment, la méthode

<br>
<br>
<br>

<p align = "center">
    <img src="/figures/cspa/cspa2.jpg"  width="100%">
</p>

---


## Exploring Motion Ambiguity and Alignment, entrainement

Pour l'entrainement, on pénalise le modèle 
- Sur la **reconstruction** de l'image $$\mathcal{L}\_1 = ||\hat{I}\_{0.5} - I\_{0.5}^{GT}||\_1$$
- Sur la **cohérence des textures** de l'image interpolée comparée aux inputs. $\mathcal{L}_{TCL}$

L'interpolation est alors formulée 

$$\hat{I}\_{0.5} = \underset{I\_{0.5}}{\text{argmin}} \mathcal{L}\_1 (\hat{I}\_{0.5}, I\_{0.5}^{GT}) + \alpha\mathcal{L}\_{TCL}(\hat{I}\_{0.5} , I\_0, I\_1)$$

---
## Exploring Motion Ambiguity and Alignment, entrainement

Comparer la texture d'$\hat{I}\_{0.5}$ avec celle de $I\_0$ et  $I\_1$ se fait part patches.

Pour un patch $\hat{f}\_x$ centré en $x$, on recherche le patch $f^{t^\*}\_{y^\*}$ lui **correspondant** le plus.

$$t^\*, y^\* =  \underset{t \in \\{0, 1\\}, y}{\text{argmin}} ||\hat{f}\_x - f^t\_y||\_2$$

$t^\*$ et $y^\*$ indexe l'image et la position.

On pénalise alors la reconstruction de ce patch

$$\mathcal{L}\_{TCL}(\hat{I}\_{0.5},I\_0, I\_1) = ||\hat{f}\_x - f^{t^\*}\_{y^\*}||\_1$$$$


---

## Exploring Motion Ambiguity and Alignment, récapitulatif

Quantitativement, cette solution est la plus performante sur les métriques classiques. Elle est cependant **nettement plus lente** que CAIN.

<br>
<br>
<p align = "center">
    <img src="/figures/cspa/cspa3.jpg"  width="100%">
</p>



---

## Exploring Motion Ambiguity and Alignment, récapitulatif

<p align = "center">
    <img src="/figures/cspa/cspa4.jpg"  width="100%">
</p>



---

## Exploring Motion Ambiguity and Alignment, récapitulatif

On note également que les structures répétitives évoluent de manière cohérente.

<br>
<p align = "center">
    <img src="/figures/cspa/cspa5.jpg"  width="100%">
</p>



---

## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- Deep Bayesian Video Frame Interpolation
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- *Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation*
- Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation
---

## Uncertainty Guided Spatial Pruning (Oct 2023)

Lorsqu'on interpole deux images, l'essence est de concentrer le calcul sur les zones de mouvement.

Cet article présente une méthode permettant de **déterminer ces zones**, réduisant la complexité de nos modèles 
<p align = "center">
    <img src="/figures/pruning/pruning1.jpg"  width="40%">
</p>



<!-- Part du principe que certaine zones ne requiert pas de calcul intensif dans les images. Propose de les ignorer avec un masque d'incertitude (NN), emploie les sparse convolution.

Propose une architecture de VFI -->

---
## Uncertainty Guided Spatial Pruning, la méthode

La méthode proposée utilise deux réseaux de neurones
- Un réseau **d'incertitude** (UEN), responsable d'estimer les zones de mouvement.
- Un réseau **d'interpolation** (VFI), responsable de l'interpolation des images.

L'architecture du VFI est très similaire à celle d'**IFRNet**.

Les images sont encodées en pyramide 
$$\phi^{\\{1, ..., 3\\}}\_{0, 1} = \mathcal{E}(I\_0, I\_1)$$

Pour être ensuite décodées en **features** et **flows** sur plusieurs niveaux.

$$F^{k-1}\_{t \rightarrow 0}, F^{k-1}\_{t \rightarrow 1}, \hat{\phi}^{k-1}\_{t} = \mathcal{D}^k(F^{k}\_{t \rightarrow 0}, F^{k}\_{t \rightarrow 1}, \hat{\phi}^k\_t, \tilde{\phi}\_0^k, \tilde{\phi}\_1^k)$$

---
## Uncertainty Guided Spatial Pruning, la méthode

L'entrainement du VFI est similaire, on ne distille pas la connaissance des flows d'un réseau tiers.

A chaque niveau, les décodeurs produisent additionnellement un **masque d'incertitude** $P\_k$ indiquant au décodeur $k-1$ les zones sur lesquelles convoluer.

La génération de ces masques est supervisée par l'UEN.

L'entrainement du modèle se fait alors en **deux parties**, premièrement l'entrainement de l'UEN puis du VFI.

Ignorer les zones redondantes implique l'utilisation de **sparse convolution**

- Durant l'**entrainement**, pour permettre la propagation des gradients, le résultat d'une convolution dense est masqué.
- Durant l'**inférence**, la convolution est exécutée en n'appliquant le kernel que sur les zones spécifiées par le masque.

---

## Uncertainty Guided Spatial Pruning, entrainement


L'UEN taché d'estimer la variance de l'image interpolée est pénalisé par 
$$\mathcal{L}\_{su} = \exp(-U) ||I\_t - f(I\_0, I\_1)||\_1 + 2U$$

Le VFI est lui pénalisé sur
- La **reconstruction** de l'image $$\mathcal{L}\_{rec} = ||I\_t - I^{GT}\_t||\_1$$
- Le degré **d'omission** controlé par $S\_t$ $$\mathcal{L}\_{s} = \left|\left | \frac{1}{\sum\_{k=1}^3 H\_k, W\_k} \left( \sum\_{k=1}^3 \sum\_{h=1}^{H\_k} \sum\_{w=1}^{W\_k} P\_{k,h,w} \right) - S\_t \right| \right|\_1$$
- La **prédiction des masques** $$\mathcal{L}\_{ugm} = ||P\_k^{u\downarrow 2^{k+1}} - P\_{k + 1}||_1$$

$P\_{k + 1}$ est le masque prédit par le VFI et $P\_k^{u\downarrow 2^{k+1}}$ et le masque estimé par l'UEN. 
---

## Uncertainty Guided Spatial Pruning, entrainement

Additionellement, le VFI utilise une branche auxiliaire n'ometant aucune zone à chaque niveau qui output $I\_t^{sc}$ et $\hat{\phi}\_t^{sc, k}$.

On peut alors pénaliser le modèle sur la reconstruction des features et images non masqués

$$\mathcal{L}\_{sc} = ||I^{sc}\_t - I^{GT}\_t||\_1 + \sum\_{k = 1}^{3} \mathcal{L}\_{cen}(\hat{\phi}\_t^{sc, k}, \hat{\phi}\_t^{k})$$

On entraine alors le VFI en optimisant

$$\mathcal{L} = \mathcal{L}\_{rec} + \lambda\_{s}\mathcal{L}\_{s} +  \lambda\_{ugm}\mathcal{L}\_{ugm} +  \lambda\_{sc}\mathcal{L}\_{sc} $$

---
## Uncertainty Guided Spatial Pruning, recapitulatif

L'hyper-paramètre $S\_t$ permet d'ajuster l'utilisation des resources.

<br>

<p align = "center">
    <img src="/figures/pruning/usgp1.jpg"  width="100%">
</p>



---

## Uncertainty Guided Spatial Pruning, recapitulatif

<br>

<p align = "center">
    <img src="/figures/pruning/ugsp2.jpg"  width="100%">
</p>

---

## Uncertainty Guided Spatial Pruning, recapitulatif

Le modèle affiche de bons résulats et améliore l'efficacité.

<br>
<br>

<p align = "center">
    <img src="/figures/pruning/ugsp3.jpg"  width="100%">
</p>

---
## Uncertainty Guided Spatial Pruning, recapitulatif

<br>

<p align = "center">
    <img src="/figures/pruning/ugsp4.jpg"  width="100%">
</p>

---

## Revue d'articles 
- IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
- Deep Bayesian Video Frame Interpolation
- Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
- Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
- *Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation*
---

## Clearer Frames, Anytime (Nov 2023)

Developpe le concept d'ambiguité
Propose le time indexing pour y remedier
Developpement mathématique (input output, regression est une moyenne)
Approche plug and play et résultats sur des modèles connus




---

# References
- Liste des articles
    - *Yu, Zhiyang, & al. "Deep Bayesian Video Frame Interpolation." Oct 2022.*
    - *Choi, Kim, & al. "Channel Attention Is All You Need for Video Frame Interpolation" 2020.*
    - *Zhou, Li, & al. "Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation" Mar 2022*
    - *Kong, Jiang, & al. "IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation" May 2022*
    - *Cheng, Jiang, & al. "Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation" Oct 2023*
    - *Zhong, Krishnan, & al. "Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation" Nov 2023*
---
- Autres références
---

