---
layout: post
title: "手撕Transformer part8 LM Head与采样"
date: 2026-09-15 19:30:00 +0800
categories: Transformer
tags: [transformer]
---

<figure class="diagram-image" style="--diagram-max-width: 794px;">
  <button class="diagram-image__trigger" type="button" aria-label="放大查看LM Head与采样流程图">
    <img src="{{ '/assets/images/transformer_part8/transformer_part8.png' | relative_url }}" alt="Qwen2.5-3B从最后一层隐藏状态经过Final RMSNorm和LM Head得到Logits，再通过贪心解码或随机采样选择下一个token">
  </button>
</figure>

经过36层Transformer Block加工之后，中间表示记作`x^36`。最后再经过一次归一化，中间表示记作`h`。

`h`的形状是(5,2048)，5行分别表示5个token的中间表示（我们的原始输入是"The capital of France is"）。生成时，前5个token已经确定，我们现在只需要预测第6个token，所以只取`h`的最后一行。

<section class="numbered-sections" markdown="1">

## LM Head怎样产生Logits

LM Head就是一个线性层，它把`h`的最后一行从2048维映射成151936个Logits，形状是（1，151936）。其中每一个元素对应词表中的一个token，分数越高，表示模型认为它越适合作为下一个token。

Logit可以是正数、负数或0，也不需要落在0到1之间。它表示候选token之间的相对倾向，还不是概率。一个Logit单独是8还是80并不能说明这个token有多大概率，还要看它与其他候选项之间的差距。

### 复用Embedding矩阵

Qwen2.5-3B使用形状是（151936，2048）的Embedding矩阵，把每个token ID转换成2048个数。到了LM Head，这里对应需要形状（2048，151936）的权重矩阵，把2048个数重新映射为151936个Logits。

这两个矩阵形状互为转置，可以选择合并为同一个矩阵。Qwen2.5-3B就是直接用Embedding矩阵的转置来作为LM Head：

\\[
(1,2048)\times E^T_{(2048,151936)}
\longrightarrow(1,151936)
\\]

这个Embedding矩阵包含大约3.11亿个参数：

\\[
151936\times2048=311164928
\\]

按BF16计算，Embedding矩阵约占590MiB。Qwen2.5-3B的总参数量约为30.9亿，对应占显存约5.8GiB，Embedding占模型权重显存约10.1%，所以合并后节省的显存占用还是很可观的。

#### 和Qwen2.5-72B对比

同一系列的Qwen2.5-72B采用了不同的选择。它的词表大小是152064，隐藏维度是8192，因此一张Embedding矩阵包含约12.5亿个参数：

\\[
152064\times8192=1245708288
\\]

按BF16计算约占2.3GiB。Qwen2.5-72B的总参数量约为727亿，对应的权重显存约为135.4GiB，所以一张Embedding矩阵约占模型权重显存的1.7%。

Qwen2.5-72B没有复用Embedding矩阵，而是为LM Head单独训练了一个单独的输出矩阵。因为占比小，所以可以多一些参数增加表达。

## 先在Logits上应用生成规则

模型本身在产生原始Logits后不会加额外的规则。不过推理程序实际运行时可以根据需要添加规则，常见的是三类惩罚参数，它们都在压低已经生成过的token的分数，用于缓解重复。

出现惩罚（presence penalty）：某个token只要在已生成的内容中出现过，就在它的Logit上减去一个固定值。

频率惩罚（frequency penalty）把出现次数也算进来：出现过多少次，就减去多少次固定值。

重复惩罚（repetition penalty）采用缩放的做法：出现过的token，Logit为正时除以一个大于1的系数，为负时乘以这个系数。两种处理都会让分数变得更低。

这三类参数在常见的推理框架中是三个独立的设置，可以单独使用，也可以组合使用。

## 关闭采样直接选择最高分

关闭随机采样时，就会使用`argmax`处理Logits。`argmax`返回Logit最大的位置，也就是选择当前分数最高的token：

每一步都拿分数最高的token，不会选择其他候选项。在环境相同的情况下，通常会得到确定的输出。

## Temperature调整候选项之间的差距

开启随机采样后，第一项常见操作是Temperature。它让每个Logit除以正数\\(T\\)，\\(T\\)就是人为设置的参数（要求\\(T>0\\)）：

\\[
z_i'=\frac{z_i}{T}
\\]

假设三个候选项的Logits是：

\\[
[4,2,1]
\\]

当\\(T=0.5\\)时：

\\[
[4,2,1]\div0.5=[8,4,2]
\\]

候选项之间的差距被放大，Softmax后的概率会更多地集中到第一名，生成结果更稳定。

当\\(T=2\\)时：

\\[
[4,2,1]\div2=[2,1,0.5]
\\]

候选项之间的差距被压缩，排名靠后的token Softmax后会分配更多概率，生成结果更容易变化。\\(T=1\\)时Logits保持不变。

## Top-k只保留固定数量的候选项

Temperature只调整分数差距，没有删除候选项。Top-k按照Logit从高到低排序，只保留前\\(k\\)个候选项。例如：

```text
token A：4.0
token B：3.0
token C：2.0
token D：1.0
token E：0.0
```

当`top_k=2`时，只保留A、B。C、D、E的Logit被设为负无穷，后面经过Softmax时概率成为0：

```text
[4, 3, 2, 1, 0]
        ↓ Top-k=2
[4, 3, -∞, -∞, -∞]
```

## Top-p根据累计概率决定候选数量

Top-p将候选项按照概率从高到低排列，然后从第一名开始累加，保留累计概率达到阈值\\(p\\)所需要的最少候选项。

假设当前候选项的临时概率是：

<div class="compact-table" markdown="1">

| 候选token | 概率 | 累计概率 |
| --- | ---: | ---: |
| A | 0.45 | 0.45 |
| B | 0.25 | 0.70 |
| C | 0.15 | 0.85 |
| D | 0.08 | 0.93 |
| E | 0.07 | 1.00 |

</div>

当`top_p=0.8`时，A和B的累计概率只有0.70，加入C以后第一次达到0.85，所以保留A、B和C，淘汰D和E。



Top-p在最终Softmax之前，属于候选筛选阶段。不过Top-p必须知道概率才能计算累计值，它在内部会根据当时的Logits临时计算一次概率。筛选后，最终Softmax还会对留下的候选项重新计算，得到真正用于抽样的概率。

## Softmax把剩余分数变成概率

经过修正、Temperature和候选筛选后，保留下来的仍然是Logits。Softmax把它们转换成总和为1的概率：

\\[
P_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}
\\]

例如剩余三个候选项的Logits是：

\\[
[2,1,0]
\\]

指数转换得到：

\\[
[e^2,e^1,e^0]\approx[7.39,2.72,1]
\\]

它们的总和约为11.11，因此最终概率约为：

\\[
[0.665,0.245,0.090]
\\]

## 采样按照概率抽出一个token

得到概率分布以后，采样过程按照每个token的概率随机抽取一次。

## 把next token接回输入继续生成

argmax或采样后得到的是一个token ID。程序将它追加到序列末尾，再检查新结果是否满足停止条件：

- 生成了EOS等结束token；
- 命中了调用方设置的Stop Sequence；
- 达到了`max_tokens`等长度限制。

例如这一轮选择了`Paris`：

```text
The capital of France is
              ↓ 选择Paris
The capital of France is Paris
```

随后模型再次根据更新后的序列预测下一个token。完整回答就是这样逐个token生成的。
</section>
