---
layout: post
title: "理解vLLM 02 API Server 接收请求"
date: 2026-10-06 00:00:00 +0800
categories: vLLM
tags: [vllm]
---

<figure class="diagram-image" style="--diagram-max-width: 1120px;">
  <button class="diagram-image__trigger" type="button" aria-label="放大查看vLLM请求接入与发送流程图">
    <img src="{{ '/assets/images/vllm/02/vLLM_02.png' | relative_url }}" alt="vLLM请求接入与发送：api_router.py校验JSON声明、监听断连并按配置跟踪在飞请求；serving.py校验模型与引擎、渲染聊天模板并分词，提供流式生成器；路由返回StreamingResponse，框架先发送响应头，再迭代生成器，驱动async_llm.py装配引擎请求、分配内部ID并准备输出收集器和请求状态；core_client.py附加前端编号，将请求编码后交给ZMQ，并确保结果接收任务已启动">
  </button>
</figure>

本篇介绍HTTP请求从接收到发送ZMQ给Engine Core进程中间的环节。上图是简化后的流程图，这些步骤都发生在API Server进程中。正文里的源码路径以`vllm/`包目录为起点。

这段流程同时推进两件事：把对话加工成引擎可以执行的请求，以及准备接收这条请求的生成结果。到ZMQ发送时，输入数据、请求标识和结果接收状态都已经就绪。

<section class="numbered-sections" markdown="1">

## api_router.py

客户端向`/v1/chat/completions`发送POST请求，设置`Content-Type: application/json`，请求体例如：

```json
{
  "model": "Qwen2.5-3B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "The capital of France is"}
  ],
  "max_tokens": 256,
  "stream": true
}
```

`model`是服务对外提供的模型名称，`messages`保存按顺序排列的对话，`max_tokens`限制最多生成多少个token，`stream: true`要求逐步返回生成内容。

### 校验JSON声明

这一步会查看请求头中的媒体类型，要求是`application/json`。只检查请求头，不检查实际内容。

### 监听disconnect事件

服务处理请求期间，客户端可能关闭页面或主动取消。这一步负责监听处理请求的handler和`http.disconnect`事件。

如果客户端没有断联，路由处理完成，就返回响应对象。如果客户端断联，就取消正在处理的任务。返回流式响应对象后，后续输出期间的断连处理由响应层接手。

### 跟踪在飞请求

进入API Server后，没有完全处理完成并返回给客户端前的请求，叫做在飞请求，也就是有多少正在处理的请求。

这一步记录并跟踪在飞请求数。默认关闭。

## serving.py

### 校验模型是否存在、引擎进程是否存活

这一步检查请求的模型是否存在，接着检查引擎状态，如果引擎进程故障，就报错返回，不进行后续的加工了。

### 用模板组装prompt

`messages`中的每条消息都带有角色和正文。模型按照训练时使用的格式读取这些消息，因此要把它们组成一段完整输入文本。这段文本称为prompt，排列规则称为聊天模板。

本例使用模型目录中`tokenizer_config.json`的`chat_template`字段。它是一段Jinja模板：Jinja按模板中的条件和循环，把消息内容填入指定位置。

前面的请求经过Qwen2.5模板处理后，得到：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
The capital of France is<|im_end|>
<|im_start|>assistant
```

`<|im_start|>`标记一条消息的开始，后面跟着角色；`<|im_end|>`标记这条消息的结束。最后的assistant起始段是留出助手回复的开头，让模型从这里继续生成。

模板渲染完成时，数据仍是一段字符串。它包含消息正文，也包含标明对话结构的特殊标记。

### 分词得到token ID

分词器根据模型目录中保存的规则，把完整prompt划分成token，并转换成对应的整数ID列表。

模板渲染和分词通过线程池执行。请求任务提交工作后，在`await`处等待。本例最终调用Hugging Face tokenizers库的Rust实现，完成实际切分。执行Rust分词时，线程释放GIL锁，API Server的事件循环可以继续服务其他请求。

### 创建生成器对象

分词完成后，serving层创建生成器对象并返回。api_router.py将它包装进StreamingResponse，框架先发送响应头，再开始迭代生成器，进入后续的请求装配和发送。客户端可以先拿到响应头，之后再等待正文中的生成内容。

## async_llm.py

### 装配引擎请求

这一步把输入token ID、生成参数等信息装配成引擎请求，并为请求分配内部ID。

### 准备输出收集器

输入装配完成后，启动`output_handler`后台任务。这是`AsyncLLM`持续取得引擎结果并进行处理的任务，首次调用时创建，后续请求复用。

然后为当前请求创建`RequestOutputCollector`，用于把处理好的结果交给等待这条请求的`generate()`。

收集器的核心是一个输出槽位和一个`asyncio.Event`。Event用来通知等待者“结果已就绪”：没有结果时，`generate()`可以等待这个事件；结果被放入槽位后，等待者被唤醒，取走结果，并清空槽位。

流式增量模式下，如果上一份结果还没取走，下一份又到达，收集器会合并增量。消费者随后取得的是合并后的内容。这让结果处理任务和请求的消费任务可以按各自的进度运行。

### 登记每条请求的输出状态

有了收集器，还需要知道收到某条请求的结果时该如何加工。这一步维护以内部请求ID为索引的记录。

记录和刚才创建的收集器绑定，并创建这条请求专用的增量反分词器。

反分词把token ID还原成文字，在新ID到来时继续解码，并提取新增内容。每条请求独立保存这个状态，分别衔接自己的回答。

至此，API Server知道了三件事：收到哪个内部ID时查哪条记录，用哪个反分词器处理它，以及把处理好的结果交给哪个收集器。

这些登记发生在发送之前。这样，发送请求后无论结果多快返回，上游都已经有对应的处理状态。

## core_client.py

### 给请求附上回邮地址

这一步给请求附上API Server的编号`client_index`。引擎生成结果后，根据这个编号把结果发回对应的API Server。本例只有一个API Server，编号为0。

结果回到API Server后，再根据内部请求ID找到对应的请求记录，继续反分词并返回给客户端。

### 经ZMQ发送给引擎进程

将编码后的信息通过ZMQ发送给引擎进程。

### 确保结果接收任务已启动

这一步检查接收引擎结果的后台任务是否已经启动。如果没有，就启动它；后续请求共用这个任务。

这个任务持续从ZMQ接收引擎返回的信息，解码后放进结果队列。前面的`output_handler`从队列取出结果，根据内部请求ID找到对应的记录，进行增量反分词，再把处理好的内容放进这条请求的输出收集器，供`generate()`取出并向上返回。

</section>
