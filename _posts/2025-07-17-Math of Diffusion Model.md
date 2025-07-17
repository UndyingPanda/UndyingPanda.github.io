---
title: Diffusion Model数学原理推导
tags: AI
---

给定真实图片$$x_0$$~$$q(x)$$，通过diffusion前向过程T次，每次添加高斯噪声得到$$x_1,x_2,...,x_T$$。整体上是一个Markov chain的过程。

具体来说，添加噪声公式为：$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \mu_t = \sqrt{1 - \beta_t} x_{t-1}, \Sigma_t = \beta_t \mathbf{I}\right)$$。其中分布概率为：$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$。

reparameterization trick：$$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

注意这里{$$\beta_t \in(0,1)$$}$$_{t=1}^T$$是一组hyper parameters，可以为constant numbers，当然也可以为learnable parameters（事实上我看到后续的Paper中有在这个方向优化做尝试）。这是一组逐渐增大的值（usually）。

$$q(x_t \mid x_{t-1}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_t|}} \exp\left( -\frac{1}{2}(x_t - \mu_t)^\top \Sigma_t^{-1}(x_t - \mu_t) \right)$$

Reverse Process简单来说就是需要求出逆向分布$$q(x_{t-1}\mid x_t)$$。这样就可以从$$x_T \sim \mathcal{N}(0,\mathbf{I})$$还原出原图分布$$x_0$$。

论文中已证明：若$$q(x_t\mid x_{t-1})$$定义为高斯分布概率密度函数，那么在$$\beta_t$$足够小的情况下，$$q(x_{t-1}\mid x_t)$$仍然是一个高斯分布。

无法简单推断$$q(x_{t-1}\mid x_t)$$的情况下，使用深度学习模型$$p_\theta$$来预测逆向分布，也就是$$p_\theta(X_{0:T})=p(X_T)\prod_{t=1}^Tp_\theta(x_{t-1}\mid x_t)$$;$$p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$$

为了方便描述，定义$$\alpha_t=1-\beta_t$$，且$$\bar \alpha_t=\prod_{i=1}^t\alpha_i$$

进一步的数学推导中，可以得出$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,\mathbf{I})$$，也就是$$x_t$$可由$$x_0$$和高斯噪声$$\epsilon$$表示出。

同时，虽然直接从$$x_t$$得到$$q(x_{t-1}\mid x_t)$$是不容易的，且直接使用Bayes公式的话也难以解决，但是如果知道$$x_0$$，通过Bayes公式可以从Forward Process的角度得到分布概率密度函数$$q(x_{t-1}\mid x_t,x_0)$$，这个表达式有比较好看的数学公式表达。

$$q(x_{t-1}\mid x_t,x_0)=\mathcal{N}(x_{t-1};\tilde\mu(x_t,x_0),\tilde\beta_t\mathbf{I})$$

其中$$q(x_{t-1}\mid x_t,x_0)=q(x_t\mid x_{t-1},x_0)\frac{q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}$$，公式右边各项的概率密度函数都是正向的，前面已经推导过，将高斯概率密度函数代入再重新整理可以得到$$\tilde \mu_\theta=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_t),\quad \epsilon_t\sim\mathcal{N}(0,1)$$,$$\tilde \beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$。

这说明我们只需要用我们的深度学习网络（一般是U-net + Attention）来预测step $$t$$时的噪声$$\epsilon_t=\epsilon_\theta(x_t,t)$$就可以得到$$\mu_\theta(x_t,t)$$，进一步得到$$\Sigma_\theta(x_t,t)=\tilde \beta_t=\frac{1-\bar \alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$。然后得到模型预测的$$p_\theta(x_{t-1}\mid x_t)$$并认为其为真实分布$$q(x_{t-1}\mid x_t)$$，最后reparameterize得到$$x_{t-1}$$，递推下去即可得到$$x_0$$。

如何设置loss使得模型预测到的分布，也即$$\mu_\theta(x_t,t)$$和$$\Sigma_\theta(x_t,t)$$尽可能地接近真实数据分布呢？这就需要在对真实数据分布下，最大化模型预测分布的对数似然，也即优化$$x_0 \sim q(x_0)$$下的$$p_\theta(x_0)$$交叉熵$$\mathcal{L}=\mathbb{E}_{q(x_0)}[-\log p_\theta(x_0)]$$。

经过数学推导最终得到$$L_t=\mathbb{E}[\frac{\beta_t^2}{2\alpha_t(1-\bar\alpha_t\left\|\Sigma_\theta\right\|_2^2}\left\|\epsilon_t-\epsilon_\theta(\sqrt{\bar \alpha_t}x_0+\sqrt{1-\bar \alpha_t}\epsilon_t,t)\right\|^2]$$

$$\sigma$$的强度控制了forward process的随机性，当$$\sigma \rarr 0$$时，随机性完全丧失，变成了一个deterministic过程，给定$$x_0$$和$$t$$，我们每次的foward process得到的都是同样的$$x_t$$也就是$$x_{t-1}$$是fixed的。

$$\sigma_t(\eta)^2=\eta \cdot \tilde \beta_t=\eta \cdot \frac{1-\bar \alpha_{t-1}}{1-\bar \alpha_t}\beta_t$$

假设总采样步$$T=1000$$，间隔为$$Q$$，那么DDIM采样的步数为$$S=T/Q$$，DDIM论文中对于不同的$$S$$和$$\eta$$实验结果如下（FID结果）
