---
layout: post
title: "概述：从回车到生成"
date: 2026-07-10 10:00:00 +0800
categories: 一次LLM请求中间都发生了什么
tags: [LLM, 深度学习]
---

当你对着 ChatGPT 敲下回车，到屏幕上开始蹦出第一个字，中间发生了什么？
这个系列从工程和原理两个角度把整条链路拆开。

## 整体流程一览

一个完整的 LLM 推理请求，大致经过这些阶段：

1. **客户端 → 服务端**：HTTP 请求带着 prompt、采样参数等到达推理服务
2. **Tokenization**：把输入文本切成 token（分词）
3. **Chat Template**：拼上系统提示词、角色标记等特殊 token
4. **Embedding**：token id → 词向量
5. **Transformer 前向**：多层 self-attention + FFN
6. **KV Cache**：增量推理时复用历史的 K/V，避免重复计算
7. **Logits → 概率**：最后一层输出词表上的得分，softmax 归一
8. **采样**：temperature / top-k / top-p 决定下一个 token
9. **循环生成**：重复 4–8，直到遇到 EOS 或达到长度上限
10. **Detokenization**：token 序列还原成文本
11. **流式回传**：SSE / WebSocket 把 token 逐个推回客户端

## 后续篇章打算拆成

- (二) Tokenization：BPE / SentencePiece 到底在切什么
- (三) KV Cache：为什么预填充和解码要分开算
- (四) 采样策略：temperature、top-p 到底在调什么
- (五) ...

<!-- TODO: 每个环节单独开一篇文章，都打上同一个分类，这里给个总览导航 -->
