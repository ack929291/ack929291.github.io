---
layout: post
title: "手撕Transformer part1 整体流程"
date: 2026-08-22 20:52:00 +0800
categories: Transformer
tags: [transformer]
---

<figure class="diagram-image" style="--diagram-max-width: 794px;">
  <button class="diagram-image__trigger" type="button" aria-label="放大查看Transformer整体流程示意图">
    <img src="{{ '/assets/images/transformer_part1/Transformer_整体_中等.png' | relative_url }}" alt="Transformer整体流程示意图">
  </button>
</figure>

Transformer整体流程示意如上图所示。这篇文章先把模型当成一条完整的流水线，看看从输入文本到生成一个token，中间大概有哪些部分、它们分别负责什么。Transformer Block内部的计算暂时省略，图中的维度以Qwen2.5-3B为基础，并做了一些更易读的修改。

直观地看，Tokenizer负责把文字翻译成模型认识的编号，Embedding把编号变成可以计算的向量，Transformer Blocks结合上下文处理这些向量，LM Head再把处理结果翻译回词表上的候选分数，最后由采样过程选出一个token。

以"The Capital of France is"作为输入，完整的一轮生成如下。

## Tokenizer和Embedding：准备模型的输入

模型不能直接接收一段字符串。Tokenizer先按照自己的切分规则把文本拆成token，再从词表中查到每个token对应的ID。图中的句子被转换成5个ID，这些数字只是token在词表中的编号。

Embedding继续把每个ID转换成一个2048维向量，得到形状为`(5, 2048)`的结果。到这里，原始文字已经变成模型能够进行矩阵运算的形式；每一行对应一个token的初始表示。

## Transformer Blocks：结合上下文处理表示

Embedding的输出会依次经过36个Transformer Block，这是模型处理输入的主体。每个Block接收上一层的结果并继续更新，36个Block各自拥有独立的参数，并不是把同一个模块重复调用36次。

输入和输出的形状始终保持`(5, 2048)`，变化的是向量中的内容。经过逐层处理后，每个token的表示不再只对应它自身，还包含了模型从当前上下文中提取的信息。至于Block内部如何做到这一点，后面再分别展开。

## 整理表示并生成词表分数

最后一个Transformer Block输出的仍然是`(5, 2048)`。模型先对这些表示做一次归一化，使不同位置的表示保持在相对稳定的数值尺度，同时不改变张量形状。Qwen2.5-3B在这里使用的归一化方法叫作RMSNorm。

随后，模型通过输出层把每个位置的2048维表示转换成词表中所有token的候选分数，这个输出层通常叫作LM Head。Qwen2.5-3B的词表包含151936个token，因此得到形状为`(5, 151936)`的logits：5表示输入中的5个位置，151936表示每个位置都要对词表中的全部token打分。

这里的5行不是对同一个位置重复预测5次，而是分别对应5段逐渐变长的上下文。以图中的输入为例，第1行表示模型只看到`The`后，对下一个token的预测；第2行表示看到`The Capital`后的预测；随后依次增加一个token，直到第5行表示看到完整的`The Capital of France is`后，对下一个token的预测。每一行都有151936个分数，也就是说词表中的每个token都是候选项，只是分数大小不同。

## 从候选项中选出下一个token

LM Head给出的是一组大小不一的分数，还不是概率。这一轮要续写的是完整输入，而不是`The`或`The Capital`等中间位置，因此只需要第5行，也就是最后一个位置产生的logits。

取出这一行后，模型使用Softmax将151936个分数转换成总和为1的概率分布，再由采样策略从中选择一个token，例如图中的`Paris`。

这里的输出不是一整句话，而只是一个token。模型将它追加到原输入末尾，序列从5个token变成6个token（"The Capital of France is Paris"），然后进入下一轮预测。

因此，一段完整回答实际是逐个token生成的。单次循环负责完成`文本 -> token ID -> 隐藏表示 -> 词表分数 -> 新token`，新token接回输入后，同样的生成过程继续进行。
