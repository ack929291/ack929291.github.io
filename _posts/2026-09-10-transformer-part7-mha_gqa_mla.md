---
layout: post
title: "手撕Transformer part7 MHA GQA MLA"
date: 2026-09-13 00:00:00 +0800
categories: Transformer
tags: [transformer]
---

<figure class="diagram-image" style="--diagram-max-width: 1400px;">
  <button class="diagram-image__trigger" type="button" aria-label="放大查看MHA完整计算流程图">
    <img src="{{ '/assets/images/transformer_part7/Transformer_part7_1_MHA.png' | relative_url }}" alt="MHA中每个Query头分别使用一个独立的KV头，并将各头结果拼接后完成输出投影">
  </button>
</figure>

<figure class="diagram-image" style="--diagram-max-width: 1400px;">
  <button class="diagram-image__trigger" type="button" aria-label="放大查看GQA完整计算流程图">
    <img src="{{ '/assets/images/transformer_part7/Transformer_part7_2_GQA.png' | relative_url }}" alt="GQA将多个Query头划分为查询组，每个查询组共享一个KV头">
  </button>
</figure>

<section class="numbered-sections" markdown="1">

## 从MHA的KV Cache说起

大语言模型通过预测下一个词生成内容，每次根据前面的token序列产生一个新的token。生成新的token时，当前Query需要查询前面所有token的Key和Value，计算注意力分数。因为每一个新token都需要查询前面所有token的Key和Value，而每次需要时才重新计算会浪费算力，所以推理引擎会把它们缓存在显存中，这部分数据就是KV Cache。

历史Query完成当时的查询后不会再次使用，KV Cache只需保存Key和Value。上下文每增加一个token，每层Attention都要为各个注意力头多保存一份Key和Value；模型层数、上下文长度或者请求并发增加时，KV Cache都会增大。

以Qwen2.5-3B的形状为例。它的隐藏维度是2048，Attention有16个注意力头，每个头是128维。

MHA是原始Transformer中的注意力结构。如果使用MHA，经过投影得到的Query、Key和Value都会被拆分到16个注意力头中：

```text
Q：16个头，每头128维
K：16个头，每头128维
V：16个头，每头128维
```

文中第一张图是简化示意图，只画了4个注意力头，完整结构中的16个头以同样的方式计算。

对于每个token、每一层，MHA需要缓存的KV Cache数值个数是：

\\[
16\\times128
+
16\\times128
=4096
\\]

Qwen2.5-3B共有36层Attention，总共需要缓存的KV Cache数值个数是：
\\[
4096\\times36
=147456
\\]

Qwen2.5-3B默认使用BF16，每个数值占2字节，每个token对应的KV Cache占0.28 MiB。如果请求序列有1000个token，这个请求的KV Cache就要占280 MiB。

KV Cache按照上下文长度线性增长，长对话和并发请求会进一步放大显存占用。MHA为每个Query头保留一个对应的KV头，也就是一份独立的Key和Value，因此保存了最多的KV数据。

## GQA让多个Query头共享一个KV头

Qwen2.5-3B实际使用GQA。它仍然保留16个Query头，但只有2个KV头。对于每个token，每个KV头会产生一份Key和Value：

```text
Query头：16个
KV头：    2个

第1～8个Query头  → 查询组1 → 共享KV头1
第9～16个Query头 → 查询组2 → 共享KV头2
```

GQA会把Query头划分成查询组：16个Query头被划分为2个查询组（划分方式可变，此处为Qwen2.5-3B的实现），每个查询组包含8个Query头，并共享一个KV头。文中第二张图将同样的关系简化成了4个Query头和2个KV头，也就是2个查询组，每个查询组包含2个Query头。

共享KV头以后，同一查询组中的Query头仍然能够得到不同结果。以查询组1为例，8个Query头分别由不同的投影参数产生，所以它们的数值不同。不同的Query与这个KV头的Key计算，会得到不同的注意力分数；经过Softmax以后，也会使用不同的权重读取它的Value。

```text
不同的Query
    ↓
与同一个KV头的Key得到不同的注意力分数
    ↓
使用不同的权重读取它的Value
    ↓
各个Query头仍然得到不同结果
```

GQA节省缓存的直接原因，是需要保存的KV头从16个减少到了2个。每个token、每层只需缓存：

\\[
2\\times128
+
2\\times128
=512
\\]

因此它的KV Cache是前面假设的MHA的：

\\[
\\frac{512}{4096}=\\frac{1}{8}
\\]

同样按照BF16、36层和1000个token计算，KV Cache会从约280MiB降到约35MiB。模型从一开始就按照这种共享关系训练，Query、Key和Value会共同适应这一结构。

## MLA缓存KV的潜在表示

<figure class="diagram-image" style="--diagram-max-width: 1500px;">
  <button class="diagram-image__trigger" type="button" aria-label="放大查看MLA完整计算流程图">
    <img src="{{ '/assets/images/transformer_part7/Transformer_part7_3_MLA.png' | relative_url }}" alt="DeepSeek-V3的MLA将KV压缩为512维潜在表示，单独保留64维RoPE部分，再为各注意力头生成内容Key和Value">
  </button>
</figure>

GQA已经减少了KVCache的数量，不过它保存的仍然是完整的Key和Value。MLA希望继续压缩：既然历史Key和Value都由隐藏状态得到，能否不保存运算的结果，只保存一份更小、并且足以参与后续计算的中间表示？

这就是MLA中Latent的含义。MLA为每个token生成一个低维潜在向量，把原本分散在许多注意力头中的Key和Value信息通过训练压缩进这个向量，推理时也主要缓存它。等到计算Attention时，再利用训练得到的另外的权重从中计算出各个头需要的信息。

上图以DeepSeek-V3为例展示了这套流程。它的隐藏维度是7168，共有128个注意力头；为了让连线保持清楚，图中只展开其中4个头。下面先沿着一个token观察它怎样变成潜在表示，再扩展到整段序列。

### 把生成K和V的信息压进512维

当前token进入Attention的隐藏状态表示为\\(h_t\\)，它有7168维。MLA先将它乘一个降维权重矩阵\\(W^{DKV}\\)：

\\[
C_t^{KV}=h_tW^{DKV}
\\]

\\[
(1,7168)\\times(7168,512)
\\longrightarrow(1,512)
\\]

\\(W^{DKV}\\)是训练得到的权重矩阵；计算出来的\\(C_t^{KV}\\) 才是当前token的KV潜在向量。这个向量经过RMSNorm以后记作\\(C_{N,t}^{KV}\\)。

如果当前序列共有`S`个token，每个token都会产生自己的512维潜在向量。把它们逐行排列起来，就得到整段序列的潜在矩阵：

\\[
C_N^{KV}\\in\\mathbb{R}^{S\\times512}
\\]

为什么一个512维向量能够代替128个头的Key和Value？因为模型从训练开始就受到这个512维瓶颈的约束。\\(W^{DKV}\\)负责压缩隐藏状态，各个头的\\(W_i^{UK}\\)和\\(W_i^{UV}\\)负责从中读取信息，这些权重在训练中一起更新。为了完成训练任务，它们逐渐学会在512维中保留后面计算Attention真正需要的内容。

对于第`i`个注意力头，从潜在表示展开内容Key和Value的关系是：

\\[
K_i^C=C_N^{KV}W_i^{UK},
\\qquad
V_i=C_N^{KV}W_i^{UV}
\\]

完整的\\(W^{UK}\\)和\\(W^{UV}\\)会把512维潜在表示分别展开成\\(128\\times128=16384\\)维，其中包含128个注意力头各自需要的内容Key或Value。图中按照注意力头将完整矩阵拆开，因此每个头对应的\\(W_i^{UK}\\)和\\(W_i^{UV}\\)形状都是`(512,128)`。同一个\\(C_N^{KV}\\)经过不同的矩阵块以后，就会为各个头产生不同的128维内容Key和128维Value：

```text
同一个C_NKV (S,512)
    ├── 产生K1_content和V1
    ├── 产生K2_content和V2
    ├── ……
    └── 产生K128_content和V128
```

图中画出的完整Key和Value就是这样得到的。不过这张图首先表达的是它们之间的生成关系，并不意味着推理时必须把所有历史Key和Value都展开并保存在显存中，这样就失去了节省显存的意义。具体的区别在下文中介绍。

### Query仍然要分成128个头

对于Query，DeepSeek-V3也采用了先降维、再升维的路线：先通过\\(W^{DQ}\\)得到1536维的\\(C^Q\\)，经过RMSNorm得到\\(C_N^Q\\)，再通过\\(W^{UQ}\\)升维。对于长度为`S`的序列，两次投影的形状变化是：

\\[
(S,7168)\\times(7168,1536)
\\longrightarrow(S,1536)
\\]

\\[
(S,1536)\\times(1536,24576)
\\longrightarrow(S,24576)
\\]

1536维的\\(C^Q\\)也可以叫作Query的潜在表示。它的作用是减少Query投影所需的参数和计算量，不负责节省KVCache：历史Query不会被后面的token再次查询，所以这个中间结果用完就可以丢弃。

升维得到的24576维来自：

\\[
24576=128\\times(128+64)
\\]

它会被整理成128个Query头，每个头共有192维。这192维全部由\\(C_N^Q\\)通过\\(W^{UQ}\\)生成，然后再按照用途分成两部分：前128维直接参与内容计算，后64维专门留给RoPE处理。

```text
Q_i_content：(S,128)
Q_i_pos：    (S,64)，RoPE处理前的Query特征
```

图中把后64维简写为`Q_i_pos`，但它并不是根据位置编号直接生成的固定位置向量。它和前128维一样，也来自当前token的Query潜在表示，因此不同token会产生不同的数值。区别在于它不会直接参与内容点积，而是先根据当前token所在的位置经过RoPE旋转，得到\\(Q_i^R\\)，再负责注意力分数中与位置有关的计算。

“后64维是位置部分”说的是它承担的计算任务。更准确地说，它是Query中用于RoPE的64维特征：模型先生成它，再经过RoPE加工。到这里，Query一侧的内容部分和RoPE部分才都准备完成。

### 为什么还要单独保存64维RoPE Key

如果只考虑内容，512维的\\(C_N^{KV}\\)已经足够生成各个头的Key和Value。但part6文章介绍过，Attention还需要通过RoPE获得相对位置信息。如果直接对展开后的内容Key做RoPE，位置相关的旋转就会夹在潜在表示与内容Key的投影之间，使\\(W_i^{UK}\\)无法通过调整乘法顺序被吸收到Query一侧。推理时只能在每一步重新展开并旋转所有历史Key，或者缓存这些展开并旋转后的完整Key：前一种方式会带来大量重复计算，后一种方式则会失去低维缓存的显存优势。

MLA因此把Key拆成两条路线。128维内容Key仍然由\\(C_N^{KV}\\)产生，并且不做RoPE，以便后面进行矩阵吸收；用于RoPE的64维Key特征则从隐藏状态中单独投影出来。\\(U\\)是经过归一化的隐藏状态：

\\[
C^{KR}=UW^{KR}
\\]

\\[
(S,7168)\\times(7168,64)
\\longrightarrow(S,64)
\\]

这64维不只是位置信息。它先由当前token的隐藏状态生成，再根据这个token所在的位置经过RoPE，得到\\(K^{R}\\)。对于同一个token，128个注意力头共享这一个RoPE Key；不同token仍然有各自的\\(K^{R}\\)，因为它们生成的特征和所在位置都可能不同。

现在，第`i`个头可以用内容Query查询内容Key，再用经过RoPE的Query部分查询共享的RoPE Key：

\\[
S_i
=
\\frac{
Q_i^{C}(K_i^{C})^{\\mathsf T}
+
Q_i^{R}(K^{R})^{\\mathsf T}
}{\\sqrt{192}}
\\]

分母中的192就是128维内容部分加64维位置部分。看起来进行了两次点积，其实它和把两段向量拼接起来再做一次192维点积完全相同：

\\[
[Q_i^C;Q_i^R][K_i^C;K^R]^{\\mathsf T}
=
Q_i^C(K_i^C)^{\\mathsf T}
+
Q_i^R(K^R)^{\\mathsf T}
\\]

这样，MLA既保留了`QKᵀ`原本的注意力计算，又把位置相关的64维单独拿了出来，使512维内容潜在表示可以继续走后面的优化路线。两个分数相加后，再经过因果Mask和Softmax，就得到了当前头的注意力权重\\(A_i\\)。

### 让潜在表示直接参与Attention

到这里还有一个问题。如果每生成一个新token，都先从缓存中的\\(C_N^{KV}\\)重新展开所有历史token的完整Key和Value，然后再计算Attention。这样每一步都需要重新展开全部历史KV，计算开销很大。MLA真正巧妙的地方，是利用矩阵乘法的结合律改变运算顺序，让潜在表示直接参与Attention。

先看内容Key。它原本由潜在表示升维得到：

\\[
K_i^C=C_N^{KV}W_i^{UK}
\\]

代入Query和Key的点积：

\\[
Q_i^C(K_i^C)^{\\mathsf T}
=
Q_i^C(C_N^{KV}W_i^{UK})^{\\mathsf T}
\\]

矩阵乘积转置以后，内部顺序需要颠倒：

\\[
(AB)^{\\mathsf T}=B^{\\mathsf T}A^{\\mathsf T}
\\]

因此，内容注意力分数可以改写成：

\\[
Q_i^C(K_i^C)^{\\mathsf T}
=
\\left(Q_i^C(W_i^{UK})^{\\mathsf T}\\right)(C_N^{KV})^{\\mathsf T}
\\]

原来的顺序是先把所有历史潜在向量展开成内容Key，再让当前Query去查询它们。调整以后，当前Query先乘\\(W_i^{UK}\\)的转置，从128维变成512维，再直接查询缓存中的\\(C_N^{KV}\\)。

在生成阶段，当前头的内容Query形状是`(1,128)`，整个换序后的过程可以写成：

\\[
(1,128)\\times(128,512)\\times(512,S)
\\longrightarrow(1,S)
\\]

最终仍然得到当前Query对`S`个历史token的`S`个注意力分数，但中间没有展开历史内容Key。

Value一侧也能使用同样的结合律。按照图中的生成关系，原本需要先计算\\(V_i=C_N^{KV}W_i^{UV}\\)，再用注意力权重读取它：

\\[
A_iV_i=A_i(C_N^{KV}W_i^{UV})
\\]

改变结合顺序后：

\\[
A_iV_i
=
A_i(C_N^{KV}W_i^{UV})
=
(A_iC_N^{KV})W_i^{UV}
\\]

于是，形状为`(1,S)`的注意力权重先对`(S,512)`的潜在矩阵加权，得到一个512维结果，再通过\\(W_i^{UV}\\)生成当前头的128维输出。历史Value同样不需要提前展开：

\\[
(1,S)\\times(S,512)\\times(512,128)
\\longrightarrow(1,128)
\\]

图中的升维权重并没有消失，它们已经被吸收到Query和注意力输出两侧的运算中，这个过程通常被称为矩阵吸收。128个头分别得到128维输出，拼接成16384维以后，再经过输出投影回到7168维，并与残差分支相加。

现在回头看KV Cache，MLA为每个历史token、每层真正需要保存的是：

```text
归一化后的C_KV：512个数
经过RoPE的K_rope：64个数
合计：             576个数
```

512维\\(C_N^{KV}\\)承载内容信息，64维\\(K^R\\)承载位置信息。Query只在当前计算中临时产生，各个头的完整内容Key和Value则通过矩阵吸收绕开。这样，MLA并不是简单地把K和V压缩后再反复解压，而是从训练结构到推理公式都围绕这份潜在表示重新组织了注意力计算。

## 三种注意力结构的主线

MHA、GQA和MLA仍然都在完成同一件事：使用Query与历史Key计算注意力权重，再用这些权重读取Value。它们改变的是Key和Value如何产生，以及推理时需要保存什么：

| 注意力结构 | Query与KV的关系 | 推理时保存的内容 |
| --- | --- | --- |
| MHA | 每个Query头对应一个独立KV头 | 所有注意力头的完整K/V |
| GQA | 每个查询组共享一个KV头 | 数量更少的完整K/V |
| MLA | 各头的K/V由共享潜在表示产生 | KV潜在表示和单独的RoPE Key |

### 放到同一个模型中比较KV Cache

前面介绍GQA时使用的是Qwen2.5-3B，介绍MLA时使用的是DeepSeek-V3。这里统一使用DeepSeek-V3的举例，计算它如果分别采用三种注意力结构会产生多少KVCache。

DeepSeek-V3的隐藏维度是7168；它共有61层Attention和128个注意力头。本次对比采用下面的条件：

```text
请求数量：1
序列长度：1000个token
Attention层数：61
Query头数量：128
Key/Value头维度：128
缓存数据类型：BF16，每个数值2字节
```

如果使用MHA，128个Query头分别拥有自己的Key和Value。每个token、每层需要缓存的数值个数：

\\[
128\\times128+128\\times128=32768
\\]

如果使用GQA，沿用前文8个Query头共享一个KV头的比例。128个Query头对应16个KV头，所以每个token、每层需要缓存的数值个数：

\\[
16\\times128+16\\times128=4096
\\]

DeepSeek-V3实际使用的MLA则保存512维\\(C_N^{KV}\\)和64维RoPE Key，每个token、每层共缓存的数值个数：

\\[
512+64=576
\\]

将它们都乘以1000个token、61层和每个数值2字节，得到下面的结果：

| 注意力结构 | 每个token、每层缓存的数值 | 1000个token的KV Cache | 相当于MHA |
| --- | ---: | ---: | ---: |
| MHA | 32768 | 约4 GB（3.72 GiB） | 1 |
| GQA（假设16个KV头） | 4096 | 约500 MB（476.56 MiB） | 1/8 |
| MLA | 576 | 约70 MB（67.02 MiB） | 约1/56.9 |

从MHA到GQA，减少的是KV头的数量；从GQA到MLA，进一步改变了KV的保存形式。它们优化的目标都指向推理阶段不断增长的KV Cache。
</section>
