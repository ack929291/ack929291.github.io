---
layout: post
title: "手撕Transformer part5 RMSNorm"
date: 2026-09-04 00:00:00 +0800
categories: Transformer
tags: [transformer]
---

<figure class="diagram-image" style="--diagram-max-width: 1150px;">
  <button class="diagram-image__trigger" type="button" aria-label="放大查看5个token经过RMSNorm前后的数值变化">
    <img src="{{ '/assets/images/transformer_part5/Transformer_part5_1_中.png' | relative_url }}" alt="The capital of France is这5个token经过RMSNorm前后的RMS、最小值和最大值，以及共享的Gamma参数统计">
  </button>
</figure>

<section class="numbered-sections" markdown="1">

## RMSNorm的运算原理

part4中，Embedding把`The capital of France is`转换成了5个2048维向量，组合起来得到形状为`(5, 2048)`的输入。进入第一个Transformer Block后，RMSNorm会分别处理这5个token向量。先取其中一个token，它的2048维向量可以写成：

\\[
\boldsymbol{x}=(x_1,x_2,\ldots,x_d),\qquad d=2048
\\]

RMS是Root Mean Square的缩写，也就是均方根。RMSNorm的均方根计算如下：

\\[
\operatorname{RMS}(\boldsymbol{x})
=
\sqrt{
\frac{1}{d}\mathop{\Large\sum}\limits_{i=1}^{d}x_i^2
+
\varepsilon
}
\\]

这个公式从根号内部开始计算。首先让`d`个分量分别平方，再把平方结果相加，平方会消除正负号对求和的抵消：

\\[
\mathop{\Large\sum}\limits_{i=1}^{d}x_i^2
=
x_1^2+x_2^2+\cdots+x_d^2
\\]

接下来，用平方和除以维度数`d`：

\\[
\frac{1}{d}\mathop{\Large\sum}\limits_{i=1}^{d}x_i^2
\\]

这一步得到`d`个分量的平方平均值。对于当前token，`d=2048`，所以这里就是把2048个平方的和除以2048。

平方平均值算出后，再加上一个极小的正数`ε`，最后开平方。`ε`是预先设定的固定常数，不参与训练，在Qwen2.5-3B中设定的值是 `1e-6`。假如一个token的2048个分量全是0，平方平均值也会是0；加入`ε`可以让开方得到的结果大于0，避免下一步除以0。正常输入的平方平均值远大于`ε`时，它对结果的影响非常小。

### 每个分量除以RMS

得到当前token的RMS以后，2048个分量都会除以RMS，也就是将数值归一成当前分量是RMS的多少倍：

\\[
\hat{x}_i
=
\frac{x_i}{\operatorname{RMS}(\boldsymbol{x})}
\\]

所有分量都除以同一个RMS，因此这一步只会整体缩放token向量。分量之间原有的比例和正负号都会保留下来。

### 与Gamma逐元素相乘

完成上述归一化以后，模型还会把结果与一个可训练向量`γ`逐元素相乘。`γ`与token向量的维度相同，其中每个数都对应token向量中同一位置的分量：

\\[
\boldsymbol{\gamma}
=
(\gamma_1,\gamma_2,\ldots,\gamma_d)
\\]

对于Qwen2.5-3B，`d=2048`，所以`γ`也是一个2048维向量。最终输出为：

\\[
\boldsymbol{y}
=
\boldsymbol{\gamma}\odot\hat{\boldsymbol{x}}
=
\boldsymbol{\gamma}\odot
\frac{\boldsymbol{x}}
{\operatorname{RMS}(\boldsymbol{x})}
\\]

`γ_i`决定归一化后的第`i`个分量还要缩放多少倍。这2048个缩放系数从训练中学习得到，使模型能够在统一整体尺度以后，继续分别调整2048个分量的数值幅度。

开头的图中，5个token在进入RMSNorm前各有不同的RMS，算RMS并除以RMS之后，每个token的RMS约为1。乘上`γ`之后，最终RMS落在约`0.33`到`0.40`之间，这就能看出`γ`的作用幅度。

### 5个token分别归一化，共享同一份Gamma

回到形状为`(5, 2048)`的完整输入，RMSNorm会为每一行单独计算RMS：

```text
The      的2048维向量 → 计算自己的RMS → 2048个分量分别除以自己的RMS
capital  的2048维向量 → 计算自己的RMS → 2048个分量分别除以自己的RMS
of       的2048维向量 → 计算自己的RMS → 2048个分量分别除以自己的RMS
France   的2048维向量 → 计算自己的RMS → 2048个分量分别除以自己的RMS
is       的2048维向量 → 计算自己的RMS → 2048个分量分别除以自己的RMS
```

这一步会得到5个不同的RMS，每个token只使用自己的2048个分量计算。随后，这5个归一化结果都会与当前层的同一个2048维`γ`逐元素相乘：

```text
输入                         (5, 2048)
每个token各自计算RMS          (5, 1)
每行除以自己的RMS            (5, 2048)
共享的γ（广播时可看作）       (1, 2048)
逐元素相乘后的输出            (5, 2048)
```

至此，一次RMSNorm可以概括成两步：每个token先使用自己的RMS统一整体尺度，再使用这一层共享的`γ`分别调整2048个分量。整个过程不会改变输入形状，5个2048维向量经过处理后仍然是`(5, 2048)`。

## 为什么需要归一化

part2中每个Block在Attention和FFN之前各有一次RMSNorm。为什么需要归一化，且在每一层Transformer Block中反复调整数值尺度？这需要从神经网络怎样训练说起。

模型会根据当前输出计算一个损失`L`（损失可以理解为模型输出和正确答案之间的差距），再通过反向传播求出损失相对于每个参数的梯度。假设模型中的一个参数是`θ`，它的梯度为：

\\[
\frac{\partial L}{\partial \theta}
\\]

这个梯度描述的是：在当前位置附近，参数`θ`发生一点变化时，损失`L`大约会变化多少。对于一个很小的参数改变量`Δθ`，可以近似写成：

\\[
\Delta L
\approx
\frac{\partial L}{\partial \theta}\Delta\theta
\\]

训练使用梯度下降更新参数：

\\[
\Delta\theta
=
-\eta\frac{\partial L}{\partial\theta}
\\]

其中`η`是学习率。梯度提供当前位置附近使损失增大最快的方向，因此取它的反方向来减小损失；学习率则控制这一步走多远。

### 深层网络中的梯度需要连续相乘

神经网络由许多层函数连接而成。先用只有一个数的简单情况表示这个过程：

\\[
h_1=f_1(h_0),\qquad
h_2=f_2(h_1),\qquad
\ldots,\qquad
L=f_{n+1}(h_n)
\\]

最终的损失`L`由`h_n`计算得到，`h_n`又来自上一层的`h_{n-1}`。要知道最前面的`h_0`怎样影响最终损失`L`，根据求导的链式法则，需要把沿途每一层的局部导数乘起来：

\\[
\frac{\partial L}{\partial h_0}
=
\frac{\partial L}{\partial h_n}
\frac{\partial h_n}{\partial h_{n-1}}
\cdots
\frac{\partial h_2}{\partial h_1}
\frac{\partial h_1}{\partial h_0}
\\]

每一个局部导数都表示：这一层的输入发生一点变化时，输出会跟着变化多少。反向传播每经过一层，从后往前传的梯度就会再乘上当前层的局部导数。网络越深，需要连乘的项就越多，因此每一层看似轻微的放大缩小也可能不断累积。

假设每一层都会把经过它的梯度放大2倍，连续经过10层以后，整体放大倍数是：

\\[
2^{10}=1024
\\]

如果每一层只让梯度保留一半，连续经过10层以后则只剩：

\\[
0.5^{10}=\frac{1}{1024}
\\]

真实的Transformer在各层之间传递的是向量，每一层也会同时改变其中的许多分量，实际连乘的是一组导数组成的矩阵。但作用仍然相同：它可能沿某些方向连续放大梯度，也可能沿另一些方向连续削弱梯度。

### 梯度爆炸

如果这些局部导数的连乘结果过大，最终传到前面各层的梯度就会异常巨大，这称为梯度爆炸。

假设某个参数收到的梯度是：

\\[
\frac{\partial L}{\partial\theta}=100000
\\]

即使学习率只有`0.001`，这次参数更新仍然会达到：

\\[
\Delta\theta
=
-0.001\times100000
=
-100
\\]

梯度描述的是当前位置附近使损失增大最快的方向，训练则沿着它的反方向，也就是负梯度方向更新参数。如果一次更新跨得太远，到达新位置以后，在原位置计算出的负梯度方向可能已经不再适用。参数可能直接越过损失较低的区域，而下一步又向相反方向跨回去，造成损失反复震荡。实际只需要一小步到达损失最小的局部点，但却因为步长太大而来回反复调整。

```text
梯度过大
    ↓
参数一次更新得太远
    ↓
越过损失较低的区域
    ↓
参数来回震荡，训练难以收敛
```

继续放大时，参数、激活值或者梯度还可能超出浮点数能够表示的有限范围，产生`inf`。`inf`参与某些无效运算后，还可能进一步产生`NaN`，表示无法得到有效的数值结果。即使梯度爆炸没有造成数值溢出，训练也已经受到影响。

### 梯度消失

如果局部导数的连乘结果越来越小，最终传到前面各层的梯度就会接近0，这称为梯度消失。

假设某个参数收到的梯度只有：

\\[
\frac{\partial L}{\partial\theta}=0.000001
\\]

学习率为`0.001`时，参数更新量是：

\\[
\Delta\theta
=
-0.001\times0.000001
=
-0.000000001
\\]

这个参数几乎没有发生变化。梯度消失的问题不等于梯度数值很小，模型已经接近最优结果时，较小的梯度是正常现象。而梯度消失是：模型的损失仍然较高，但损失发出的学习信号在经过许多层以后被连续削弱，传到前面各层时已经几乎消失。后面的层可能仍在更新，前面的层却很难继续学习。

梯度只描述当前位置附近的小幅变化。梯度接近0不代表无论怎样修改参数都无法改变损失，而是梯度下降在当前位置无法得到足够的更新幅度和有效的前进信号。

### 前向数值与反向梯度的联系

RMSNorm作用于前向传播中的输入值，而梯度爆炸和梯度消失发生在反向传播阶段。接下来进一步说明它们之间的联系。

反向传播也需要使用前向传播的结果和中间值。在一次前向传播完成后，现在开始反向传播。继续用只有一个数的简化情况：假设当前层的输入为`x`，参数为`w`，前向计算为：

\\[
y=wx
\\]

现在调整参数`w`。根据链式法则，损失`L`对参数`w`的梯度为：

\\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial w}
=
\frac{\partial L}{\partial y}x
\\]

可以看到，前向传播的输入`x`会直接参与参数梯度的计算。在后面传回来的梯度保持不变时，`x`过大会放大参数梯度，`x`过小也会使参数梯度变小。

因此，RMSNorm虽然不直接规定梯度应该有多大，但它会控制前向输入`x`的数值尺度，使参数梯度中由`x`引起的放大或缩小更加稳定，减少输入尺度波动对训练的干扰。

## RMSNorm与LayerNorm的区别

RMSNorm是在LayerNorm的基础上简化而来的。为了理解它省略了什么，先看LayerNorm的计算过程。

### LayerNorm的计算过程

对于一个包含`d`个分量的token向量：

\\[
\boldsymbol{x}=(x_1,x_2,\ldots,x_d)
\\]

LayerNorm首先计算这些分量的均值：

\\[
\mu
=
\frac{1}{d}\mathop{\Large\sum}\limits_{i=1}^{d}x_i
\\]

然后计算每个分量与均值的差，并用这些差计算方差：

\\[
\sigma^2
=
\frac{1}{d}
\mathop{\Large\sum}\limits_{i=1}^{d}(x_i-\mu)^2
\\]

接下来，每个分量减去均值，再除以标准差：

\\[
\hat{x}_i
=
\frac{x_i-\mu}
{\sqrt{\sigma^2+\varepsilon}}
\\]

最后，LayerNorm还会让结果逐元素乘上可训练向量`γ`，再加上可训练向量`β`：

\\[
y_i
=
\gamma_i\hat{x}_i+\beta_i
\\]

`γ`决定每个分量缩放多少倍，`β`决定每个分量向正方向或负方向移动多少。这两个向量都包含`d`个可训练参数，并由当前LayerNorm层的所有token共享。

### LayerNorm归一化后的分布

在乘`γ`、加`β`之前，减去均值完成了重新居中，使这些分量的均值变为0；除以标准差完成了尺度控制，使它们的方差接近1。乘`γ`、加`β`以后，最终输出不再需要保持上述均值和方差。

### RMSNorm省去了重新居中

RMSNorm的作者将LayerNorm的作用分成了两部分：减去均值带来的重新居中，以及除以标准差带来的重新缩放。作者提出，LayerNorm帮助模型稳定训练的关键可能是重新缩放带来的尺度控制，而重新居中的作用没那么大。

基于这个判断，RMSNorm去掉了均值的计算和减去均值的操作，直接使用RMS调整向量的尺度：

\\[
y_i
=
\gamma_i
\frac{x_i}
{\operatorname{RMS}(\boldsymbol{x})}
\\]

因此，在乘`γ`之前，RMSNorm处理后的向量满足：

\\[
\operatorname{RMS}(\hat{\boldsymbol{x}})\approx1
\\]

由于没有减去均值，它的均值不一定为0，方差也不一定为1。RMSNorm不会专门把分布中心移动到0，而是让分布中心和各个分量一起按相同比例缩放。

[RMSNorm论文](https://proceedings.neurips.cc/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf)的实验结果显示，RMSNorm通常能够取得与LayerNorm相近的模型效果，支持了作者提出的假设：在这些实验覆盖的模型和任务中，重新居中可以省略，尺度控制更为关键。这个结论并不意味着重新居中在所有模型中都没有作用。

### 理论上的计算开销

从公式包含的运算看，LayerNorm需要计算均值和方差两个统计量，还要让每个分量减去均值、除以标准差、乘`γ`并加`β`。RMSNorm只需要计算平方平均值并开方，然后让每个分量除以RMS并乘`γ`。

与LayerNorm相比，RMSNorm省去了均值的计算，也省去了“先求均值，再根据均值计算方差”的依赖关系。本文所观察的Qwen RMSNorm没有`β`这组可训练参数。因此，无论是前向计算还是反向求导，RMSNorm的计算过程都更加简单。


至此，part2中原本被概括成一步的RMSNorm已经完整展开。Embedding产生的向量经过第一次RMSNorm以后，就会进入Attention，开始计算Q、K和V。
</section>
