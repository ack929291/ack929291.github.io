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

直观地看，Tokenizer先确定文本中有哪些token，并找到它们在词表中的编号；Embedding再把每个编号换成能够承载信息的向量；Transformer Blocks根据上下文逐层更新这些向量；LM Head根据最终的表示给词表中的所有候选token打分，最后由采样过程选出一个token。

以"The Capital of France is"作为输入，完整的一轮生成如下。

## Tokenizer：token ID只是编号

模型不能直接接收一段字符串。Tokenizer先按照自己的切分规则把文本拆成token，再从词表中查到每个token对应的ID。图中的句子被转换成5个ID，这些数字只是token在词表中的编号。

这些ID只用于从Embedding中查找每个token对应的向量。

## Embedding：把编号换成向量

Embedding可以先理解为一张随模型一起训练的查找表，词表中的每个token都对应一个2048维向量。输入一个token ID，模型就取出它对应的那一行。图中的5个ID因此变成5个向量，组合起来得到形状为`(5, 2048)`的结果。

每个token的向量都有2048个维度，这些向量共同位于一个2048维的表示空间中。训练会不断调整Embedding表中的数值，使这个空间逐渐形成对预测有用的结构。经典词向量例子`king - man + woman ≈ queen`表达的就是这种直觉：向量的距离、方向和线性运算可以承载模型学到的关系与知识。

到这里，每一行只是对应token的初始表示，还没有结合当前句子的上下文。同一个token在查表时会得到同一个初始向量，它在这句话里具体表示什么，要由后面的Transformer Blocks继续处理。

## Transformer Blocks：结合上下文处理表示

Embedding的输出会依次经过36个Transformer Block，这是模型处理输入的主体。这里的“结合上下文”，指的是每个位置不再只保留自己的初始表示，而是从当前可见的其他位置取得信息，再更新自己的向量。

以最后一个位置的`is`为例，刚进入第一个Block时，这一行主要来自`is`本身的Embedding；经过逐层处理后，它还包含了模型从`The Capital of France is`这段前缀中提取的信息。后面的LM Head正是根据这个带有上下文的表示，判断下一个token更可能是`Paris`还是词表中的其他候选项。

处理过程中，输入和输出的形状始终保持`(5, 2048)`，变化的是5个向量中的内容。每个Block接收上一层的结果并继续更新，36个Block各自拥有独立的参数，并不是把同一个模块重复调用36次。至于Block内部如何完成信息交换，后面再分别展开。

## 整理向量并给词表中的token打分

最后一个Transformer Block输出的是5个2048维向量，仍然记作`(5, 2048)`。模型先对每个向量分别做一次归一化，使它们以相对稳定的数值尺度进入输出层。归一化不会改变向量的长度，处理后仍然是5个2048维向量。Qwen2.5-3B在这里使用的归一化方法叫作RMSNorm。

随后，每个2048维向量都会进入一个叫作LM Head的输出层。这个向量包含了模型从当前位置及其上下文中提取的信息，LM Head根据这些信息，给词表中的每个token分别打分。分数越高，表示模型认为这个token越适合接在当前位置后面，经过后续处理时也越容易被选中。

LM Head不会直接给出一个确定的token，而是先把词表中的所有token都当作选项。每个选项得到的原始分数叫作logit。

Qwen2.5-3B的词表包含151936个token。LM Head分别处理前面的5个2048维向量，每个向量都产生151936个分数，因此最终得到5行logits，记作`(5, 151936)`。

这5行分别预测5个位置之后应该接哪个token：

- 第1行来自`The`这个位置处理后的向量，用来预测第2个token，也就是`The`后面应该接什么。在当前输入中，实际接在后面的是`Capital`。
- 第2行来自`Capital`这个位置处理后的向量，它已经包含`The Capital`这段上下文，用来预测第3个token。在当前输入中，下一个token是`of`。
- 第3行根据`The Capital of`预测第4个token，当前输入中是`France`。
- 第4行根据`The Capital of France`预测第5个token，当前输入中是`is`。
- 第5行根据完整的`The Capital of France is`预测第6个token。这个token还没有出现在输入中，正是这一轮需要生成的内容。

每一行都会给词表中的151936个token分别打分，区别只在于它们看到的上下文长度不同。

## 从候选项中选出下一个token

虽然模型同时计算了5行logits，但前4行预测的`Capital`、`of`、`France`和`is`都已经包含在输入中，这一轮不需要再从这些位置选择token。当前任务是续写完整的`The Capital of France is`，因此只取第5行，也就是最后一个位置产生的logits。

取出这一行后，模型使用Softmax将151936个分数转换成总和为1的概率分布，再由采样策略从中选择一个token，例如图中的`Paris`。

这里的输出不是一整句话，而只是一个token。模型将它追加到原输入末尾，序列从5个token变成6个token（"The Capital of France is Paris"），然后进入下一轮预测。

回头看这一轮生成，模型一直在改变同一段信息的表示形式：文字先变成token ID，ID再换成初始向量，Transformer Blocks把它们更新为包含上下文的向量，LM Head则把最后一个位置的向量变成整个词表的分数。选出的新token接回输入后，这条流程再次开始，一段完整的回答就是这样逐个token生成的。
