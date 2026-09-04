---
layout: post
title: "手撕Transformer part4 Embedding"
date: 2026-08-30 22:00:00 +0800
categories: Transformer
tags: [transformer]
---

part3中，Tokenizer把`The capital of France is`转换成了5个token ID：

```text
The | capital | of | France | is
785 | 6722    | 315| 9625   | 374
```

这些整数只是token在词表中的编号，还不能直接进入后面的Transformer Block。Embedding接在Tokenizer后面，负责把每个编号换成一个2048维向量。这篇文章先看推理时怎样完成这次转换，再看训练怎样让一张最初随机初始化的表逐渐形成能够承载语义的向量空间。

<section class="numbered-sections" markdown="1">

## Embedding就是一张查找表

Qwen2.5-3B的Embedding权重可以记作矩阵`E`。它一共有151936行、2048列：

```text
E.shape = (151936, 2048)
```

151936行对应模型可以接收的151936个token ID。第0行属于ID 0，第1行属于ID 1，依次直到最后一行。每一行包含2048个数，也就是这个token对应的2048维向量。

因此，token ID在这里直接充当行号：

```text
token ID = i
Embedding向量 = E[i]
```

以`The capital of France is`为例，Tokenizer输出的5个ID会分别取出下面5行：

```text
785   → E[785]   → 2048维向量
6722  → E[6722]  → 2048维向量
315   → E[315]   → 2048维向量
9625  → E[9625]  → 2048维向量
374   → E[374]   → 2048维向量
```

把这5个向量按照原来的token顺序排列起来，就得到形状为`(5, 2048)`的结果。5表示当前输入有5个token，2048表示每个token都由2048个数表示。

这就是Embedding在推理阶段完成的全部工作：根据token ID取出对应行，再把得到的`(5, 2048)`交给第一个Transformer Block继续处理。至此，Embedding的推理过程结束。

## Embedding也会参与训练

模型开始训练之前，Embedding表中的数值会被随机初始化。此时每个token已经拥有自己固定的一行，但这一行里的2048个数还不能准确地表示这个token。

Embedding表中的数值和Transformer Block中的权重一样，都是模型需要训练的参数。随着模型在大量文本上训练，这些数值也会被不断调整。模型的预测越来越准确时，每个token对应的向量也逐渐形成对预测有用的表示，并开始承载这个token的语义信息。

训练完成后，Embedding便知道应该用怎样的2048维向量表示每个token。

## 只有ID，语言关系为什么仍然存在

token ID本身没有语言含义。785不比374更接近某种语义，它们只是两个不同的编号。

可以把一段文本想成一部剧本，把其中的每个token想成一个角色。Tokenizer为每个角色取了一个固定的数字外号，例如`France`的外号是9625，`is`的外号是374。在指向角色这件事上，叫它的名字和叫它的外号是一回事；只要这个对应关系保持不变，看到9625就能够确定它指的是`France`。

因此，把剧本中的角色名全部换成数字外号，不会改变角色的出场顺序，也不会改变它们原有的关系。文本转换成token ID以后，token出现的顺序、上下文以及预测目标都保留了下来。

例如，训练语料中可能反复出现下面两种结构：

```text
The capital of France is Paris
The capital of Germany is Berlin
```

转换成ID以后，模型看到的仍然是两串具有相同结构的序列：`France`和`Germany`经常出现在相似的位置，`Paris`和`Berlin`也经常成为相似上下文中的预测目标。模型利用这些数字外号在大量文本中怎样共同出现，逐渐学到它们所指角色之间的关系。

随着训练继续进行，经常出现在相似上下文、承担相似作用的token会收到具有共同规律的调整。它们的向量不必完全相同，但在2048维空间中的位置和方向会逐渐形成结构。整数ID由此只负责稳定地找到某一行，真正被学习和使用的是这一行中的2048个数。

## 2048维语义空间

一个token的2048个数，可以看成2048维空间中的一个点。

这里的单个维度并不直接符合人的直觉。例如，第1维不会被预先规定为“国家”，第2维也不会被预先规定为“城市”。各个维度的含义是在训练中形成的，一个token的表示由这2048个维度共同决定。

不过，训练后的向量之间也会形成一些可以被人理解的空间关系。在Qwen2.5-3B的Embedding实测中，用`v`表示一个token的Embedding向量，可以得到下面的关系：

\\[
\boldsymbol{v}\_{\text{Paris}} - \boldsymbol{v}\_{\text{France}} + \boldsymbol{v}\_{\text{Germany}} \approx \boldsymbol{v}\_{\text{Berlin}}
\\]

`Paris`减去`France`，可以理解为从这组向量中取出“国家到首都”的方向；再把这个方向加到`Germany`上，得到的结果与`Berlin`非常接近。这说明Embedding不仅保存了每个token对应的数值，也在训练中形成了能够表示语言关系、并且具有一定可解释性的空间结构。

## 回到整体流程

从part3的最后一步开始，Tokenizer已经得到`[785, 6722, 315, 9625, 374]`。Embedding把这些ID依次当作行号，从形状为`(151936, 2048)`的权重表中取出5行，于是输入从5个整数变成了5个2048维向量。

得到的`(5, 2048)`就是第一个Transformer Block的输入。至此，part1中从token ID到向量的过程已经完整展开。

</section>
